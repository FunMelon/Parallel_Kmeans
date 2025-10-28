#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include "error.cuh"
#include "helper_cuda.h"
#include "kmeansGPU.h"

#define MAX_STREAMS 32
#define BLOCK_SIZE 256

// 构造函数
KmeansGPU::KmeansGPU(
    int numClusters,    // 类的数量
    int numFeatures,    // 特征数量
    float* clusters,    // [numClusters, numFeatures]，类中心点坐标
    int nsamples        // 样本数量
) : Kmeans(numClusters, numFeatures, clusters, nsamples) {}


// 复杂构造函数
KmeansGPU::KmeansGPU(
    int numClusters,
    int numFeatures,
    float* clusters,
    int nsamples,
    int maxIters,
    float epsilon
) : Kmeans(numClusters, numFeatures, clusters, nsamples, maxIters, epsilon) {}

// 模板：将数组x的前N个元素初始化为value
template <typename T>
__global__ void init(
    T* x,           // 指向设备端数组的指针
    const T value,  // 初始化值
    const int N     // 元素数量
) {
    const int n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n < N) {
        x[n] = value;
    }
}

// 计算每个样本到指定聚类中心的距离
__global__ void calDistKernel(
    const float* d_data,        // [nsamples, numFeatures]，样本数据
    const float* d_clusters,    // [numClusters, numFeatures]，类中心点坐标
    float* d_distance,          // [nsamples, numClusters]，样本到各个类的距离，这里只计算指定类那一列
    const int numClusters,      // 类的数量
    const int clusterNo,        // 当前计算的类编号
    const int nsamples,         // 样本数量
    const int numFeatures       // 特征数量
) {
    int sample_idx = blockIdx.x;    // 不同的块处理不同的样本
    int feature_idx = threadIdx.x;  // 同一块内不同的线程处理同一样本的不同特征
    
    extern __shared__ float s_c[];
    
    // 初始化共享内存
    s_c[feature_idx] = 0.0f;
    
    // 计算平方差
    if (sample_idx < nsamples && feature_idx < numFeatures) {
        int data_idx = sample_idx * numFeatures + feature_idx;
        int cluster_idxx = clusterNo * numFeatures + feature_idx;
        float diff = d_data[data_idx] - d_clusters[cluster_idxx];
        s_c[feature_idx] = diff * diff;
    }
    __syncthreads();
    
    // 同一个块内的线程规约求和
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (feature_idx < offset && (feature_idx + offset) < numFeatures) {
            s_c[feature_idx] += s_c[feature_idx + offset];
        }
        __syncthreads();
    }
    
    // 写入结果，更新d_distance的特定列
    if (feature_idx == 0 && sample_idx < nsamples) {
        d_distance[sample_idx * numClusters + clusterNo] = sqrtf(s_c[0]);
    }
}

// 寻找每个样本到各个类的最小距离，并记录下对应的类编号，同时统计每个类的样本数量
__global__ void assignToNearestClusterKernel(
    const float* d_distance,      // [nsamples, numClusters]，样本到各个类的距离
    int* d_sampleClasses,         // [nsamples, ]，样本所属的类编号
    int* d_clusterSize,           // [numClusters, ]，存储每个类的样本数量
    const int numClusters,        // 类的数量
    const int nsamples,           // 样本数量
    float* d_minDist              // [nsamples, ]，每个样本到其所属类的最小距离
) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;  // 每个线程处理一个样本
    if (idx < nsamples) {
        float minDist = d_distance[idx * numClusters];
        int minIdx = 0;
        
        for (int i = 1; i < numClusters; ++i) {
            float tmp = d_distance[idx * numClusters + i];
            if (tmp < minDist) {
                minDist = tmp;
                minIdx = i;
            }
        }
        d_sampleClasses[idx] = minIdx;
        d_minDist[idx] = minDist;
        
        // 原子计数每个聚类的样本数
        atomicAdd(&d_clusterSize[minIdx], 1);
    }
}

// 累加损失，并行数组求和
__global__ void accumulateLossKernel(
    const float* input, // 输入数组
    float* output,      // 输出数组
    const int size      // 输入数组大小
) {
    extern __shared__ float s_data[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    
    // 每个线程加载数据到共享内存
    s_data[tid] = (i < size) ? input[i] : 0.0f;
    __syncthreads();
    
    // 规约求和
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }
    
    // 将块的结果写入全局内存，每个块处理一个输出元素
    if (tid == 0) {
        output[blockIdx.x] = s_data[0];
    }
}

// 累加样本到聚类中心
// 把每个样本的特征值累加到其所属类的特征和中，方便后续计算均值
__global__ void accumulateClustersKernel(
    const float* d_data,        // [nsamples, numFeatures]，样本数据
    float* d_clusterSums,       // [numClusters, numFeatures]，类中心点的特征和
    const int* d_sampleClasses, // [nsamples, ]，样本所属的类编号
    const int nsamples,       // 样本数量
    const int numFeatures     // 特征数量
) {
    int sample_idx = blockIdx.x;    // 每个块处理一个样本
    int feature_idx = threadIdx.x;  // 每个块内的线程处理该样本的不同特征
    
    if (sample_idx < nsamples && feature_idx < numFeatures) {
        int cluster_idx = d_sampleClasses[sample_idx];
        int data_idx = sample_idx * numFeatures + feature_idx;
        int cluster_feature_idx = cluster_idx * numFeatures + feature_idx;
        
        atomicAdd(&d_clusterSums[cluster_feature_idx], d_data[data_idx]);
    }
}

// 更新聚类中心
// 通过给出的该类的特征和与样本数量，计算新的类中心点坐标
__global__ void updateClusterCentersKernal(
    float* d_clusters,          // [numClusters, numFeatures]，类中心点坐标
    const float* d_clusterSums, // [numClusters, numFeatures]，类中心点的特征和
    const int* d_clusterSize,   // [numClusters, ]，每个类的样本数量
    const int numClusters,      // 类的数量
    const int numFeatures       // 特征数量
) {
    int cluster_idx = blockIdx.x;   // 每个块处理一个聚类
    int feature_idx = threadIdx.x;  // 每个块内的线程处理该聚类的不同特征
    
    if (cluster_idx < numClusters && feature_idx < numFeatures) {
        int idx = cluster_idx * numFeatures + feature_idx;
        int count = d_clusterSize[cluster_idx];
        
        if (count > 0) {
            d_clusters[idx] = d_clusterSums[idx] / count;
        }
    }
}

// 计算总损失
float computeTotalLoss(
    float* d_minDist,   // [nsamples, ]，每个样本到其所属类的最小距离
    int nsamples        // 样本数量
) {
    float* d_partialSums;   // 设备端部分和数组
    float* h_partialSums;   // 主机端部分和数组
    
    // 分配所需的内存
    int blockSize = BLOCK_SIZE;
    int numBlocks = (nsamples + blockSize - 1) / blockSize;  // 向上取整
    
    CHECK(cudaMalloc(&d_partialSums, numBlocks * sizeof(float)));   // 分配设备端部分和数组
    h_partialSums = new float[numBlocks];
    
    // 第一次规约，计算每个块的部分和
    accumulateLossKernel<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(
        d_minDist, d_partialSums, nsamples);
    
    // 第二次规约，多个块的部分和累加，由于部分和可能长度不止一个块，这里需要判断
    if (numBlocks > 1) {
        float* d_finalSum;
        CHECK(cudaMalloc(&d_finalSum, sizeof(float)));
        accumulateLossKernel<<<1, blockSize, blockSize * sizeof(float)>>>(
            d_partialSums, d_finalSum, numBlocks);
        
        float finalLoss;
        CHECK(cudaMemcpy(&finalLoss, d_finalSum, sizeof(float), cudaMemcpyDeviceToHost));
        CHECK(cudaFree(d_finalSum));
        
        CHECK(cudaFree(d_partialSums));
        delete[] h_partialSums;
        
        return finalLoss;
    } else {
        float finalLoss;
        CHECK(cudaMemcpy(&finalLoss, d_partialSums, sizeof(float), cudaMemcpyDeviceToHost));
        CHECK(cudaFree(d_partialSums));
        delete[] h_partialSums;
        return finalLoss;
    }
}

// 更新聚类中心
void updateClusterWithCuda(
    const float* d_data,    // [nsamples, numFeatures]，样本数据
    float* d_clusters,      // [numClusters, numFeatures]，类中心点坐标
    float* d_clusterSums,   // [numClusters, numFeatures]，类中心点的特征和
    int* d_sampleClasses,   // [nsamples, ]，样本所属的类编号
    int* d_clusterSize,     // [numClusters, ]，每个类的样本数量
    const int nsamples,     // 样本数量
    const int numClusters,  // 类的数量
    const int numFeatures   // 特征数量
) {
    // 初始化聚类的特征和以及样本数量计数器
    init<float><<<(numClusters * numFeatures + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        d_clusterSums, 0.0f, numClusters * numFeatures);
    
    // 统计每个聚类的样本数量
    // int blockSize = BLOCK_SIZE;
    // int gridSize = (nsamples + blockSize - 1) / blockSize;
    // countCluster<<<gridSize, blockSize>>>(d_sampleClasses, d_clusterSize, nsamples);
    
    // 累加样本到对应的聚类中心
    accumulateClustersKernel<<<nsamples, numFeatures>>>(
        d_data, d_clusterSums, d_sampleClasses, nsamples, numFeatures);
    
    // 计算新的聚类中心（平均值）
    updateClusterCentersKernal<<<numClusters, numFeatures>>>(
        d_clusters, d_clusterSums, d_clusterSize, numClusters, numFeatures);
    
    // CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
}

// 完整的距离计算流程 
void calDistWithCuda(
    const float* d_data,
    float* d_clusters,
    float* d_distance,
    int* d_sampleClasses,
    float* d_minDist,
    float* d_loss,
    int* d_clusterSize,
    const int numClusters,
    const int nsamples,
    const int numFeatures
) {
    // 重置聚类大小计数器
    init<int><<<(numClusters + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_clusterSize, 0, numClusters);
    
    // 使用多个流并行计算距离
    cudaStream_t streams[MAX_STREAMS];
    int smem = sizeof(float) * numFeatures;
    
    for (int i = 0; i < numClusters; i++) { // 不同类别使用不同流来进行聚类
        CHECK(cudaStreamCreate(&streams[i]));
        calDistKernel<<<nsamples, numFeatures, smem, streams[i]>>>(
            d_data, d_clusters, d_distance, numClusters, i, nsamples, numFeatures);
    }
    
    // 等待所有流完成
    for (int i = 0; i < numClusters; i++) {
        CHECK(cudaStreamSynchronize(streams[i]));
        CHECK(cudaStreamDestroy(streams[i]));
    }
    
    // 寻找最小距离并分类样本
    int blockSize = BLOCK_SIZE;
    int gridSize = (nsamples + blockSize - 1) / blockSize;

    init<int><<<(numClusters + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        d_clusterSize, 0, numClusters);

    assignToNearestClusterKernel<<<gridSize, blockSize>>>(
        d_distance, d_sampleClasses, d_clusterSize, numClusters, nsamples, d_minDist);
    
    // CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
}

// 对calDistWithCuda函数的封装，因为核函数不能作为类的成员函数
void KmeansGPU::getDistance(const float* d_data) {
    calDistWithCuda(d_data, d_clusters, d_distances, d_sampleClasses, d_minDist,
                   d_loss, d_clusterSize, m_numClusters, m_nsamples, m_numFeatures);
}

// 更新聚类中心
void KmeansGPU::updateClusters(const float* d_data) {
    // 根据上一步的d_sampleClasses，获取新的d_clusterSums和d_clusterSize，然后更新d_clusters
    updateClusterWithCuda(d_data, d_clusters, d_clusterSums, d_sampleClasses, 
                         d_clusterSize, m_nsamples, m_numClusters, m_numFeatures);
    
    // 拷贝结果回主机
    CHECK(cudaMemcpy(m_clusters, d_clusters, 
                    sizeof(float) * m_numClusters * m_numFeatures, 
                    cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(m_sampleClasses, d_sampleClasses, 
                    sizeof(int) * m_nsamples, 
                    cudaMemcpyDeviceToHost));
}

// 完整的fit函数
void KmeansGPU::fit(const float* v_data) {
    // 1.准备变量
    float* d_data;  // [nsamples, numFeatures]，样本数据
    int datamem = sizeof(float) * m_nsamples * m_numFeatures;       // 数据内存大小
    int clustermem = sizeof(float) * m_numClusters * m_numFeatures; // 聚类内存大小
    int sampleClassmem = sizeof(int) * m_nsamples;                   // 样本分类内存大小
    int distmem = sizeof(float) * m_nsamples * m_numClusters;            // 距离内存大小
    
    // 调试信息
    std::cout << "GPU KMeans: " << m_nsamples << " samples, " 
              << m_numFeatures << " features, " << m_numClusters << " clusters" << std::endl;

    // 2.分配设备端内存
    CHECK(cudaMalloc((void**)&d_data, datamem));
    CHECK(cudaMalloc((void**)&d_clusters, clustermem));
    CHECK(cudaMalloc((void**)&d_clusterSums, clustermem)); 
    CHECK(cudaMalloc((void**)&d_sampleClasses, sampleClassmem));
    CHECK(cudaMalloc((void**)&d_distances, distmem));
    CHECK(cudaMalloc((void**)&d_minDist, sizeof(float) * m_nsamples));
    CHECK(cudaMalloc((void**)&d_loss, sizeof(float) * m_nsamples));
    CHECK(cudaMalloc((void**)&d_clusterSize, sizeof(int) * m_numClusters));
    
    // 3.将数据从主机端复制到设备端
    CHECK(cudaMemcpy(d_data, v_data, datamem, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_clusters, m_clusters, clustermem, cudaMemcpyHostToDevice));

    // 4.kmeans迭代核心
    float lastLoss = 0;
    for (int i = 0; i < m_maxIters; ++i) {
        // 计算距离并分类样本
        this->getDistance(d_data);
        
        // 计算当前损失
        float currentLoss = computeTotalLoss(d_minDist, m_nsamples);
        
        // 更新聚类中心
        this->updateClusters(d_data);
        
        // 输出迭代信息
        std::cout << "Iter: " << i + 1 << " Loss: " << currentLoss 
                  << " Change: " << std::abs(lastLoss - currentLoss) << std::endl;
        
        // 检查收敛条件
        if (std::abs(lastLoss - currentLoss) < this->m_epsilon // 收敛条件1：损失变化小于阈值
            || i == m_maxIters - 1) { // 收敛条件2：达到最大迭代次数
            std::cout << "Converged at iteration " << i + 1 << std::endl;
            break;
        }

        lastLoss = currentLoss;
    }

    // 5.将最终结果从设备端复制到主机端
    CHECK(cudaMemcpy(m_clusters, d_clusters, clustermem, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(m_sampleClasses, d_sampleClasses, sampleClassmem, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(m_distances, d_distances, distmem, cudaMemcpyDeviceToHost));

    // 6.释放设备端内存
    CHECK(cudaFree(d_data));
    CHECK(cudaFree(d_clusters));
    CHECK(cudaFree(d_clusterSums));
    CHECK(cudaFree(d_sampleClasses));
    CHECK(cudaFree(d_distances));
    CHECK(cudaFree(d_minDist));
    CHECK(cudaFree(d_loss));
    CHECK(cudaFree(d_clusterSize));
    
    std::cout << "GPU KMeans completed" << std::endl;
}