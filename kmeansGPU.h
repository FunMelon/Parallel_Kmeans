#ifndef KMEANSGPU_H
#define KMEANSGPU_H

#include "kmeans.h"

class KmeansGPU : public Kmeans {
public:
    KmeansGPU(int numClusters, int numFeatures, float* clusters, int nsamples);
    KmeansGPU(int numClusters,
        int numFeatures,
        float* clusters,
        int nsamples,
        int maxIters,
        float eplison);
    virtual void getDistance(const float* d_data);
    virtual void updateClusters(const float* d_data);
    virtual void fit(const float* v_data);
    virtual const char* getLabelFileName() const override {
        return "cluster_labels_parallel_cpp.csv";
    }
    
    float* d_clusters;     // [numClusters, numFeatures]，设备端的m_clusters
    int* d_sampleClasses;  // 设备端的m_sampleClasses
    float* d_distances;    // 设备端的m_distances

    float* d_minDist;    // [nsamples, ]，设备端的每个样本到各个类的最小距离
    float* d_loss;       // [nsamples, ]，存储 d_minDist 的规约结果
    int* d_clusterSize;  //[numClusters,
    //]，存储每个类的样本数量，方便求样本均值更新中心点坐标

private:
    KmeansGPU(const Kmeans& model);
    KmeansGPU& operator=(const Kmeans& model);
};

#endif