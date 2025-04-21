#include <iostream>
#include "kmeansGPU.h"
#include "cuda_func.cu"

// 构造函数
KmeansGPU::KmeansGPU(int numClusters,
                     int numFeatures,
                     float* clusters,
                     int nsamples)
    : Kmeans(numClusters, numFeatures, clusters, nsamples) {}

// 复杂构造函数
KmeansGPU::KmeansGPU(int numClusters,
                     int numFeatures,
                     float* clusters,
                     int nsamples,
                     int maxIters,
                     float eplison)
    : Kmeans(numClusters, numFeatures, clusters, nsamples, maxIters, eplison) {}


void KmeansGPU::fit(const float* v_data) {
    float* d_data;
    int datamem = sizeof(float) * m_nsamples * m_numFeatures;
    int clustermem = sizeof(float) * m_numClusters * m_numFeatures;
    int sampleClassmem = sizeof(int) * m_nsamples;
    int distmem = sizeof(float) * m_nsamples * m_numClusters;
    int* h_clusterSize = new int[m_numClusters]{0};
    float* h_loss = new float[m_nsamples]{0.0};

    CHECK(cudaMalloc((void**)&d_data, datamem));
    CHECK(cudaMalloc((void**)&d_clusters, clustermem));
    CHECK(cudaMalloc((void**)&d_sampleClasses, sampleClassmem));
    CHECK(cudaMalloc((void**)&d_distances, distmem));
    CHECK(cudaMalloc((void**)&d_minDist, sizeof(float) * m_nsamples));
    CHECK(cudaMalloc((void**)&d_loss, sizeof(float) * m_nsamples));
    CHECK(cudaMalloc((void**)&d_clusterSize, sizeof(int) * m_numClusters));

    CHECK(cudaMemcpy(d_data, v_data, datamem, cudaMemcpyHostToDevice));
    CHECK(
        cudaMemcpy(d_clusters, m_clusters, clustermem, cudaMemcpyHostToDevice));

    float lastLoss = 0;
    for (int i = 0; i < m_maxIters; ++i) {
        this->getDistance(d_data);
        this->updateClusters(d_data);
        CHECK(
            cudaMemcpy(h_loss, d_loss, sampleClassmem, cudaMemcpyDeviceToHost));
        this->m_optTarget = h_loss[0];
        if (std::abs(lastLoss - this->m_optTarget) < this->m_eplison)
            break;
        lastLoss = this->m_optTarget;
        std::cout << "Iters: " << i + 1 << "  current loss : " << m_optTarget
                  << std::endl;
    }

    CHECK(
        cudaMemcpy(m_clusters, d_clusters, clustermem, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(m_sampleClasses, d_sampleClasses, sampleClassmem,
                     cudaMemcpyDeviceToHost));
    CHECK(
        cudaMemcpy(m_distances, d_distances, distmem, cudaMemcpyDeviceToHost));

    CHECK(cudaFree(d_data));
    CHECK(cudaFree(d_clusters));
    CHECK(cudaFree(d_sampleClasses));
    CHECK(cudaFree(d_distances));
    CHECK(cudaFree(d_minDist));
    CHECK(cudaFree(d_loss));
    CHECK(cudaFree(d_clusterSize));
    delete[] h_clusterSize;
    delete[] h_loss;
}



void KmeansGPU::getDistance(const float* d_data) {
    calDistWithCuda(d_data, d_clusters, d_distances, d_sampleClasses, d_minDist,
                    d_loss, d_clusterSize, m_numClusters, m_nsamples,
                    m_numFeatures);
}