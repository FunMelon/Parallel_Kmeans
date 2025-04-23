#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include "MACRO.h"
#include "helper_cuda.h"
#include "kmeansGPU.h"

// 构造函数
KmeansGPU::KmeansGPU(
    int numClusters,    // 类的数量
    int numFeatures,    // 特征数量
    float* clusters,    // [numClusters, numFeatures]，类中心点坐标
    int nsamples        // 样本数量
)
    : Kmeans(numClusters, numFeatures, clusters, nsamples) {
}

// 复杂构造函数
KmeansGPU::KmeansGPU(
    int numClusters,    // 类的数量
    int numFeatures,    // 特征数量
    float* clusters,    // [numClusters, numFeatures]，类中心点坐标
    int nsamples,       // 样本数量
    int maxIters,       // 最大迭代次数
    float epsilon       // 迭代停止条件（损失函数的变化小于该值时停止迭代）
)
    : Kmeans(numClusters, numFeatures, clusters, nsamples, maxIters, epsilon) {
}

// 模板：将数组x的前N个元素初始化为value
template <typename T>
__global__ void init(
    T* x,           // 指向device上数组的指针
    const T value,  // 初始化值
    const int N     // 数组大小
) {
    const int n =
        threadIdx.x +
        blockIdx.x * blockDim.x;  // 将2D/3D的线程布局映射为一维数组索引n（线程在bolck中的编号
    // + block在grid中的编号 * block中线程的数量）
    if (n < N)  // 当前线程下标合法时，执行赋值
        x[n] = value;
}

// 定义全局函数包装核函数（由于核函数不能作为成员函数）
// 计算每个样本到每个类的距离
__global__ void calDistKernel(
    const float* d_data,      // [nsamples, numFeatures]，样本数据
    const float* d_clusters,  // [numClusters, numFeatures]，类中心点坐标
    float* d_distance,        // [nsamples, numClusters]，每个样本到每个类的距离
    const int numClusters,    // 类的数量
    const int clusterNo,      // 当前计算的类的编号
    const int nsamples,       // 样本数量
    const int numFeatures     // 特征数量
) {
    int n =
        threadIdx.x +
        numFeatures * blockIdx.x;  // n 指向 d_data 中：样本索引是 blockIdx.x，维度是 threadIdx.x
    int m =
        threadIdx.x +
        numFeatures * clusterNo;  // m 指向 d_clusters 中：类中心是 clusterNo，维度是 threadIdx.x
    extern __shared__ float
        s_c[];  // 声明一个共享内存数组 s_c，长度在 kernel 启动时动态指定。
    s_c[threadIdx.x] = 0.0;  // 每个线程初始化自己的那一位为 0
    // 计算每个样本到当前类的距离
    if (n < numFeatures * nsamples && threadIdx.x < numFeatures) {
        s_c[threadIdx.x] = powf(d_data[n] - d_clusters[m], 2);
    }
    __syncthreads();
    // 开始进行规约
    for (int offset = blockDim.x >> 1; offset >= 32;
        offset >>= 1) {  // 标准归d约操作，从两两合并开始，逐层减少线程
        if (threadIdx.x < offset)
            s_c[threadIdx.x] += s_c[threadIdx.x + offset];
        __syncthreads();
    }
    // blockDim.x <= 32 时，__syncthreads() 不再有效率，因此进入 warp 范围归约
    for (int offset = 16; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset)
            s_c[threadIdx.x] += s_c[threadIdx.x + offset];
        __syncwarp();
    }

    if (threadIdx.x == 0)  // 将最终结果写入全局内存，开根号变为欧式距离
        d_distance[blockIdx.x * numClusters + clusterNo] = sqrt(s_c[0]);  // 第 blockIdx.x 个样本到第 clusterNo 个类的距离在一维数组中的位置
}

// 寻找每个样本到各个类的最小距离，并记录下对应的类编号
__global__ void reduceMin(
    float* d_distance,      // [nsamples, numClusters]，每个样本到每个类的距离
    int* d_sampleClasses,   // [nsamples, ]，每个样本对应的类编号
    int* d_clusterSize,     // [numClusters, ]，每个类的样本数量
    int numClusters,        // 类的数量
    int nsamples,           // 样本数量
    float* d_minDist        // [nsamples, ]，每个样本到各个类的最小距离
) {
    int n = threadIdx.x + blockDim.x * blockIdx.x;  // 每个线程处理一个样本
    if (n < nsamples) {
        float minDist = d_distance[n * numClusters + 0];
        int minIdx = 0;
        float tmp;
        for (int i = 1; i < numClusters; ++i) {
            tmp = __ldg(&d_distance[n * numClusters + i]);  // __ldg() 是一个只读内存访问函数，适用于只读数据
            if (tmp < minDist) {
                minDist = tmp;
                minIdx = i;
            }
        }
        d_sampleClasses[n] = minIdx;  // 记录下对应的类编号
        d_minDist[n] = minDist;  // 记录下最小距离
    }
}

// 将 d_minDist 中所有样本的最小距离做一个加和（求总损失）
__global__ void reduceSum(
    float* d_minDist,   // [nsamples, ]，每个样本距离最近类的距离
    float* d_loss,      // [nsamples, ]，每个 block 的局部损失
    int nsamples        // 样本数量
) {
    int n = threadIdx.x + blockIdx.x * blockDim.x;  // 每个线程处理一个样本
    extern __shared__ float s_y[];  // 共享内存，存储每个 block 的局部损失
    float y = 0.0;
    const int stride = blockDim.x * gridDim.x;  // 用于 grid-stride-loop，确保所有数据都被访问
    for (; n < nsamples; n += stride)  // 多轮次处理数据，最大化利用线程
        y += d_minDist[n];
    s_y[threadIdx.x] = y;  // 每个线程将自己的局部损失存入共享内存
    __syncthreads();
    // 1.线程间规约
    for (int offset = blockDim.x >> 1; offset >= 32; offset >>= 1) {
        if (threadIdx.x < offset)
            s_y[threadIdx.x] += s_y[threadIdx.x + offset];
        __syncthreads();
    }
    // 2.warp内规约
    for (int offset = 16; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset)
            s_y[threadIdx.x] += s_y[threadIdx.x + offset];
        __syncwarp();
    }
    // 3.将每个 block 的局部损失写入全局内存
    if (threadIdx.x == 0)
        d_loss[blockIdx.x] = s_y[0];
}

// 将所有样本的特征值加到其对应的聚类中心上，并求平均，更新聚类中心的位置
__global__ void update(
    const float* d_data,    // [nsamples, numFeatures]，样本数据
    float* d_clusters,      // [numClusters, numFeatures]，类中心点坐标
    int* d_sampleClasses,   // [nsamples, ]，每个样本对应的类编号
    int* d_clusterSize,     // [numClusters, ]，每个类的样本数量
    const int nsamples,     // 样本数量
    const int numFeatures   // 特征数量
) {
    int n = threadIdx.x + numFeatures * blockIdx.x;
    int clusterId = d_sampleClasses[blockIdx.x];  // 每个样本对应的类编号
    int clustercnt = d_clusterSize[clusterId];  // 每个类的样本数量
    if (threadIdx.x < numFeatures) {
        atomicAdd(&(d_clusters[clusterId * numFeatures + threadIdx.x]),
            d_data[n] / clustercnt);
    }
}

// 统计每个聚类（cluster）中被分配了多少个样本
__global__ void countCluster(
    int* d_sampleClasses,   // [nsamples, ]，每个样本对应的类编号
    int* d_clusterSize,     // [numClusters, ]，每个类的样本数量
    int nsamples            // 样本数量
) {
    int n = threadIdx.x + blockDim.x * blockIdx.x;
    if (n < nsamples) {
        int clusterID = d_sampleClasses[n];  // 每个样本对应的类编号
        atomicAdd(&(d_clusterSize[clusterID]), 1);  // 原子操作，避免数据竞争
    }
}

// 用于在 CPU 上统计每个聚类（cluster）中被分配了多少个样本
void countClusterHost(
    int* h_sampleClasses,   // [nsamples, ]，每个样本对应的类编号
    int* h_clusterSize,     // [numClusters, ]，每个类的样本数量
    int nsamples,           // 样本数量
    int numClusters         // 类的数量
) {
    // 初始化 h_clusterSize 为 0
    std::fill(h_clusterSize, h_clusterSize + numClusters, 0);

    for (int n = 0; n < nsamples; ++n) {
        int clusterID = h_sampleClasses[n];  // 每个样本对应的类编号
        if (clusterID >= 0 && clusterID < numClusters) {
            h_clusterSize[clusterID]++;  // 更新计数
        }
    }
}

// 根据当前样本的分配情况，重新计算并更新每个聚类中心的位置（即均值）
void updateClusterWithCuda(
    const float* d_data,    // [nsamples, numFeatures]，样本数据
    float* d_clusters,      // [numClusters, numFeatures]，类中心点坐标
    int* d_sampleClasses,   // [nsamples, ]，每个样本对应的类编号
    int* d_clusterSize,     // [numClusters, ]，每个类的样本数量
    const int nsamples,     // 样本数量
    const int numClusters,  // 类的数量
    const int numFeatures   // 特征数量
) {
    init<float> << <1, 1024 >> > (d_clusters, 0.0, numClusters * numFeatures);  // 初始化类中心点坐标，启动一个线程块，大小为1024
    int blockSize = 1024;   // 线程块的大小
    int gridSize = (nsamples - 1) / blockSize + 1;  // 计算网格大小，确保覆盖所有样本

    // *******************在CPU上统计聚类***************************
    // int* h_sampleClasses = new int[nsamples];
    // cudaMemcpy(h_sampleClasses, d_sampleClasses, nsamples * sizeof(int), cudaMemcpyDeviceToHost); // 从 GPU 将样本分类信息拷贝到 CPU
    // int* h_clusterSize = new int[numClusters];
    // countClusterHost(h_sampleClasses, h_clusterSize, nsamples, numClusters);  // 在 CPU 上统计每个聚类的样本数量
    // cudaMemcpy(d_clusterSize, h_clusterSize, numClusters * sizeof(int), cudaMemcpyHostToDevice);  // 将聚类大小从 CPU 拷贝回 GPU

    // ********************在GPU上统计聚类**********************
    countCluster << <gridSize, blockSize >> > (d_sampleClasses, d_clusterSize, nsamples);  // 统计每个聚类中被分配了多少个样本
    // *****************************************************

    update << <nsamples, 128 >> > (d_data, d_clusters, d_sampleClasses,
        d_clusterSize, nsamples, numFeatures);  // 更新聚类中心的位置
}

// 更新聚类中心
// d_data: 样本数据
void KmeansGPU::updateClusters(const float* d_data) {
    int clustermem = sizeof(float) * m_numClusters * m_numFeatures; // 存储聚类中心的内存大小
    int sampleClassmem = sizeof(int) * m_nsamples;  // 存储样本类编号的内存大小
    int* h_clusterSize = new int[m_numClusters] {0};    // 创建一个host上的数组，用于存储每个聚类的样本数量
    CHECK(  // 将设备端的 d_clusterSize 数据（表示每个聚类的大小）复制到主机端的 h_clusterSize 数组中。
        cudaMemcpy(h_clusterSize, d_clusterSize, sizeof(int) * m_numClusters, cudaMemcpyDeviceToHost)
    );
    updateClusterWithCuda(d_data, d_clusters, d_sampleClasses, d_clusterSize, m_nsamples, m_numClusters, m_numFeatures);  // 更新聚类中心
    CHECK(  // 将更新后的 d_clusters（设备端聚类中心数据）复制回主机端的 m_clusters 数组中。
        cudaMemcpy(m_clusters, d_clusters, clustermem, cudaMemcpyDeviceToHost)
    );
    CHECK(  // 将每个样本的所属聚类（d_sampleClasses）从设备端复制到主机端的 m_sampleClasses 数组中。
        cudaMemcpy(m_sampleClasses, d_sampleClasses, sampleClassmem, cudaMemcpyDeviceToHost)
    );
    delete[] h_clusterSize;  // 释放主机端的 h_clusterSize 数组
}

// 完整的fit函数
void KmeansGPU::fit(const float* v_data) {
    // 1.准备变量
    float* d_data;                                                  // 样本数据的设备端指针
    int datamem = sizeof(float) * m_nsamples * m_numFeatures;       // 样本数据的内存大小
    int clustermem = sizeof(float) * m_numClusters * m_numFeatures; // 聚类中心的内存大小
    int sampleClassmem = sizeof(int) * m_nsamples;                  // 样本类编号的内存大小
    int distmem = sizeof(float) * m_nsamples * m_numClusters;       // 每个样本到每个类的距离的内存大小
    int* h_clusterSize = new int[m_numClusters] {0};                // 主机端的聚类大小数组
    float* h_loss = new float[m_nsamples] {0.0};                    // 主机端的损失数组

    // 2.分配设备端内存
    CHECK(cudaMalloc((void**)&d_data, datamem));
    CHECK(cudaMalloc((void**)&d_clusters, clustermem));
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
        this->getDistance(d_data);  // 计算每个样本到每个类的距离
        this->updateClusters(d_data);   // 更新聚类中心
        CHECK(cudaMemcpy(h_loss, d_loss, sampleClassmem, cudaMemcpyDeviceToHost));  // 将损失从设备端复制到主机端
        this->m_optTarget = h_loss[0];  // 获取损失值
        if (std::abs(lastLoss - this->m_optTarget) < this->m_epsilon)   // 判断是否收敛
            break;
        lastLoss = this->m_optTarget;   // 更新上次损失值
        std::cout << "Iters: " << i + 1 << "  current loss : " << m_optTarget << std::endl;
    }

    // 5.将结果从设备端复制到主机端
    CHECK(cudaMemcpy(m_clusters, d_clusters, clustermem, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(m_sampleClasses, d_sampleClasses, sampleClassmem, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(m_distances, d_distances, distmem, cudaMemcpyDeviceToHost));

    // 6.释放设备端内存
    CHECK(cudaFree(d_data));
    CHECK(cudaFree(d_clusters));
    CHECK(cudaFree(d_sampleClasses));
    CHECK(cudaFree(d_distances));
    CHECK(cudaFree(d_minDist));
    CHECK(cudaFree(d_loss));
    CHECK(cudaFree(d_clusterSize));
    // 7.释放主机端内存
    delete[] h_clusterSize;
    delete[] h_loss;
}

// 完整的 KMeans 距离计算及样本分类流程。
void calDistWithCuda(
    const float* d_data,    // [nsamples, numFeatures]，样本数据
    float* d_clusters,      // [numClusters, numFeatures]，类中心点坐标
    float* d_distance,      // [nsamples, numClusters]，每个样本到每个类的距离
    int* d_sampleClasses,   // [nsamples, ]，每个样本对应的类编号
    float* d_minDist,       // [nsamples, ]，每个样本到各个类的最小距离
    float* d_loss,          // [nsamples, ]，每个样本距离最近类的距离（损失函数）
    int* d_clusterSize,     // [numClusters, ]，每个类的样本数量
    const int numClusters,  // 类的数量
    const int nsamples,     // 样本数量
    const int numFeatures   // 特征数量
) {
    // 1.启动一个线程块，大小为128，初始化类中心点坐标为0
    init<int> << <1, 128 >> > (d_clusterSize, 0, numClusters);  // 初始化每个类的样本数量为0
    // 2.创建多个CUDA流
    int smem = sizeof(float) * 128;
    cudaStream_t streams[20];   // 流的数量，最多20个流
    for (int i = 0; i < numClusters; i++) {
        CHECK(cudaStreamCreate(&(streams[i])));  // 创建流
    }
    // 3.计算每个样本到每个类的距离
    for (int i = 0; i < numClusters; i++) {
        calDistKernel << <nsamples, 128, smem, streams[i] >> > (    // 对每个样本 d_data 计算它到类中心 i 的欧式距离，结果保存在 d_distance。
            d_data, d_clusters, d_distance, numClusters, i, nsamples, numFeatures);
    }
    // 4.销毁流
    for (int i = 0; i < numClusters; ++i) { // 销毁流
        CHECK(cudaStreamDestroy(streams[i]));
    }
    // 5.计算每个样本到各个类的最小距离，并记录下对应的类编号
    int blockSize = 256;
    int gridSize = (nsamples - 1) / blockSize + 1;
    reduceMin << <gridSize, blockSize, sizeof(int)* blockSize >> > (    // // 计算每个样本到各个类的最小距离，并记录下对应的类编号
        d_distance, d_sampleClasses, d_clusterSize, numClusters, nsamples,
        d_minDist);
    // 6.累计损失
    reduceSum << <256, 256, sizeof(float) * 256 >> > (d_minDist, d_loss, nsamples);  // 把d_minDist分多个block并行求和写入d_loss
    reduceSum << <1, 256, sizeof(float) * 256 >> > (d_loss, d_loss, 256);   // 把d_loss中多个block的部分和再规约为最终loss，写入d_loss[0]
}

// 对calDistWithCuda函数的封装，方便调用
void KmeansGPU::getDistance(const float* d_data) {
    calDistWithCuda(d_data, d_clusters, d_distances, d_sampleClasses, d_minDist,
        d_loss, d_clusterSize, m_numClusters, m_nsamples,
        m_numFeatures);
}