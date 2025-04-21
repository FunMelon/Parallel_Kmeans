#include "MACRO.h"
#include <cuda_runtime.h>
#include "helper_cuda.h"

// 模板：将数组x的前N个元素初始化为value
// x: 指向device上数组的指针
// value: 初始化值
// N: 数组大小
template <typename T>
__global__ void init(T* x, const T value, const int N) {
    const int n =
        threadIdx.x +
        blockIdx.x *
            blockDim
                .x;  // 将2D/3D的线程布局映射为一维数组索引n（线程在bolck中的编号
                     // + block在grid中的编号 * block中线程的数量）
    if (n < N)  // 当前线程下标合法时，执行赋值
        x[n] = value;
}

// 定义全局函数包装核函数（由于核函数不能作为成员函数）
// 计算每个样本到每个类的距离
// d_data: [nsamples, numFeatures]，样本数据
// d_clusters: [numClusters, numFeatures]，类中心点坐标
// d_distance: [nsamples, numClusters]，每个样本到每个类的距离
// numClusters: 类的数量
// clusterNo: 当前计算的类的编号
// nsamples: 样本数量
// numFeatures: 特征数量
__global__ void calDistKernel(
    const float* d_data,
    const float* d_clusters,  // [numClusters, numFeatures]
    float* d_distance,        // [nsamples, numClusters]
    const int numClusters,
    const int clusterNo,
    const int nsamples,
    const int numFeatures) {
    int n =
        threadIdx.x +
        numFeatures * blockIdx.x;  // 指向样本 blockIdx.x 的 threadIdx.x 维度
    int m =
        threadIdx.x +
        numFeatures * clusterNo;  // 指向聚类中心 clusterNo 的 threadIdx.x 维度
    extern __shared__ float
        s_c[];  // 声明一个共享内存数组 s_c，长度在 kernel 启动时动态指定。
    s_c[threadIdx.x] = 0.0;
    // 计算每个样本到当前类的距离
    if (n < numFeatures * nsamples && threadIdx.x < numFeatures) {
        s_c[threadIdx.x] = powf(d_data[n] - d_clusters[m], 2);
    }
    __syncthreads();
    // 开始进行规约
    for (int offset = blockDim.x >> 1; offset >= 32;
         offset >>= 1) {  // 标准归约操作，从两两合并开始，逐层减少线程
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
    // 将每个 block 的归约结果写入 d_distance
    if (threadIdx.x == 0)
        d_distance[blockIdx.x * numClusters + clusterNo] = sqrt(s_c[0]);
}

__global__ void reduceMin(float* d_distance,
                          int* d_sampleClasses,
                          int* d_clusterSize,
                          int numClusters,
                          int nsamples,
                          float* d_minDist) {
    int n = threadIdx.x + blockDim.x * blockIdx.x;
    if (n < nsamples) {
        float minDist = d_distance[n * numClusters + 0];
        int minIdx = 0;
        float tmp;
        for (int i = 1; i < numClusters; i++) {
            tmp = __ldg(&d_distance[n * numClusters + i]);
            if (tmp < minDist) {
                minDist = tmp;
                minIdx = i;
            }
        }
        d_sampleClasses[n] = minIdx;
        d_minDist[n] = minDist;
    }
}

__global__ void reduceSum(float* d_minDist, float* d_loss, int nsamples) {
    int n = threadIdx.x + blockIdx.x * blockDim.x;
    extern __shared__ float s_y[];
    float y = 0.0;
    const int stride = blockDim.x * gridDim.x;
    for (; n < nsamples; n += stride)
        y += d_minDist[n];
    s_y[threadIdx.x] = y;
    __syncthreads();
    for (int offset = blockDim.x >> 1; offset >= 32; offset >>= 1) {
        if (threadIdx.x < offset)
            s_y[threadIdx.x] += s_y[threadIdx.x + offset];
        __syncthreads();
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset)
            s_y[threadIdx.x] += s_y[threadIdx.x + offset];
        __syncwarp();
    }
    if (threadIdx.x == 0)
        d_loss[blockIdx.x] = s_y[0];
}

__global__ void countCluster(int* d_sampleClasses,
                             int* d_clusterSize,
                             int nsamples) {
    int n = threadIdx.x + blockDim.x * blockIdx.x;
    if (n < nsamples) {
        int clusterID = d_sampleClasses[n];
        atomicAdd(&(d_clusterSize[clusterID]), 1);
    }
}

__global__ void update(const float* d_data,
                       float* d_clusters,
                       int* d_sampleClasses,
                       int* d_clusterSize,
                       const int nsamples,
                       const int numFeatures) {
    int n = threadIdx.x + numFeatures * blockIdx.x;
    int clusterId = d_sampleClasses[blockIdx.x];
    int clustercnt = d_clusterSize[clusterId];
    if (threadIdx.x < numFeatures) {
        atomicAdd(&(d_clusters[clusterId * numFeatures + threadIdx.x]),
                  d_data[n] / clustercnt);
    }
}

void updateClusterWithCuda(const float* d_data,
                           float* d_clusters,
                           int* d_sampleClasses,
                           int* d_clusterSize,
                           const int nsamples,
                           const int numClusters,
                           const int numFeatures) {
    init<float><<<1, 1024>>>(d_clusters, 0.0, numClusters * numFeatures);
    int blockSize = 1024;
    int gridSize = (nsamples - 1) / blockSize + 1;
    countCluster<<<gridSize, blockSize>>>(d_sampleClasses, d_clusterSize,
                                          nsamples);
    update<<<nsamples, 128>>>(d_data, d_clusters, d_sampleClasses,
                              d_clusterSize, nsamples, numFeatures);
}

void calDistWithCuda(const float* d_data,
                     float* d_clusters,
                     float* d_distance,
                     int* d_sampleClasses,
                     float* d_minDist,
                     float* d_loss,
                     int* d_clusterSize,
                     const int numClusters,
                     const int nsamples,
                     const int numFeatures) {
    init<int><<<1, 128>>>(d_clusterSize, 0, numClusters);
    int smem = sizeof(float) * 128;
    cudaStream_t streams[20];
    for (int i = 0; i < numClusters; i++) {
        CHECK(cudaStreamCreate(&(streams[i])));
    }
    for (int i = 0; i < numClusters; i++) {
        calDistKernel<<<nsamples, 128, smem, streams[i]>>>(
            d_data, d_clusters, d_distance, numClusters, i, nsamples,
            numFeatures);
    }
    for (int i = 0; i < numClusters; ++i) {
        CHECK(cudaStreamDestroy(streams[i]));
    }
    int blockSize = 256;
    int gridSize = (nsamples - 1) / blockSize + 1;
    reduceMin<<<gridSize, blockSize, sizeof(int) * blockSize>>>(
        d_distance, d_sampleClasses, d_clusterSize, numClusters, nsamples,
        d_minDist);
    reduceSum<<<256, 256, sizeof(float) * 256>>>(d_minDist, d_loss, nsamples);
    reduceSum<<<1, 256, sizeof(float) * 256>>>(d_loss, d_loss, 256);
}