#include "kmeans.h"
#include <cmath>
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <vector>
#include <algorithm>
// 实现构造函数
// numClusters: 聚类数量
// numFeatures：特征维度
// clusters: 指向一个浮点数数组的指针，用于存储聚类中心的坐标
// nsamples: 样本的数量
Kmeans::Kmeans(int numClusters, int numFeatures, float* clusters, int nsamples)
    : m_numClusters(numClusters),
    m_numFeatures(numFeatures),
    m_clusters(new float[numClusters * numFeatures]),
    m_nsamples(nsamples),
    m_optTarget(1e7),
    m_maxIters(50),
    m_epsilon(0.001),
    m_distances(new float[nsamples * numClusters] {0.0}),
    m_sampleClasses(new int[nsamples] {0}) {
    for (int i = 0; i < numClusters * numFeatures; ++i) {
        this->m_clusters[i] = clusters[i];
    }
}

// 详细的构造函数
// ...
// maxIterszx: 最大迭代此时
// epsilon: 目标阈值
Kmeans::Kmeans(int numClusters,
    int numFeatures,
    float* clusters,
    int nsamples,
    int maxIters,
    float epsilon)
    : m_numClusters(numClusters),
    m_numFeatures(numFeatures),
    m_clusters(new float[numClusters * numFeatures]),
    m_nsamples(nsamples),
    m_optTarget(1e7),
    m_maxIters(maxIters),
    m_epsilon(epsilon),
    m_distances(new float[nsamples * numClusters] {0.0}),
    m_sampleClasses(new int[nsamples] {0}) {
    for (int i = 0; i < numClusters * numFeatures; ++i) {
        this->m_clusters[i] = clusters[i];
    }
}

// 析构函数
Kmeans::~Kmeans() {
    if (m_clusters)
        delete[] m_clusters;
    if (m_distances)
        delete[] m_distances;
    if (m_sampleClasses)
        delete[] m_sampleClasses;
}

// 计算距离函数
// 1.针对数据集的每个样本计算其到k个聚类中心的距离
// 2.找出距离最近的中心点，更新样本类别
// 3.将所有样本与其最近中心的距离求和做loss
void Kmeans::getDistance(const float* v_data) {
    /*
        v_data: [nsamples, numFeatures, ]
    */

    float loss = 0.0;
    for (int i = 0; i < m_nsamples; ++i) {
        float minDist = 1e8;
        int minIdx = -1;
        for (int j = 0; j < m_numClusters; ++j) {
            float sum = 0.0;
            for (int k = 0; k < m_numFeatures; ++k) {
                sum += (v_data[i * m_numFeatures + k] -
                    m_clusters[j * m_numFeatures + k]) *
                    (v_data[i * m_numFeatures + k] -
                        m_clusters[j * m_numFeatures + k]);
            }
            this->m_distances[i * m_numClusters + j] = sqrt(sum);
            if (sum <= minDist) {
                minDist = sum;
                minIdx = j;
            }
        }
        m_sampleClasses[i] = minIdx;
        loss += m_distances[i * m_numClusters + minIdx];
    }
    m_optTarget = loss;
}

// 根据聚类结果
// m_sampleClasses，计算每个类中的样本均值，作为类的新的中心，然后更新到
// m_clusters 上。
void Kmeans::updateClusters(const float* v_data) {
    for (int i = 0; i < m_numClusters * m_numFeatures; ++i)
        this->m_clusters[i] = 0.0;
    for (int i = 0; i < m_numClusters; ++i) {
        int cnt = 0;
        for (int j = 0; j < m_nsamples; ++j) {
            if (i != m_sampleClasses[j])
                continue;
            for (int k = 0; k < m_numFeatures; ++k) {
                this->m_clusters[i * m_numFeatures + k] +=
                    v_data[j * m_numFeatures + k];
            }
            cnt++;
        }
        for (int ii = 0; ii < m_numFeatures; ii++)
            this->m_clusters[i * m_numFeatures + ii] /= cnt;
    }
}

// 保存标签
void Kmeans::saveLabels() {
    const char* filename = getLabelFileName();  // 调用虚函数获取文件名
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        std::cerr << "cannot open " << filename << std::endl;
        return;
    }
    for (int i = 0; i < m_nsamples; ++i) {
        ofs << m_sampleClasses[i] << std::endl;
    }
    ofs.close();
    std::cout << "save labels to " << filename << std::endl;
}

// 计算精度
float Kmeans::accuracy(const int* label) {
    // 聚类结果 -> m_sampleClasses
    // 真实标签 -> label

    std::vector<std::unordered_map<int, int>> clusterLabelCount(m_numClusters);

    // 1. 统计每个聚类编号中，真实标签出现频次
    for (int i = 0; i < m_nsamples; ++i) {
        int clusterId = m_sampleClasses[i];
        int trueLabel = label[i];
        clusterLabelCount[clusterId][trueLabel]++;
    }

    // 2. 找出每个聚类中最常见的真实标签，计算正确预测数量
    int correct = 0;
    for (int i = 0; i < m_nsamples; ++i) {
        int clusterId = m_sampleClasses[i];
        int trueLabel = label[i];

        // 获取该聚类中最常出现的真实标签
        const auto& labelMap = clusterLabelCount[clusterId];
        int mostCommonLabel = -1;
        int maxCount = 0;
        for (const auto& pair : labelMap) {
            if (pair.second > maxCount) {
                mostCommonLabel = pair.first;
                maxCount = pair.second;
            }
        }

        if (trueLabel == mostCommonLabel)
            correct++;
    }

    return static_cast<float>(correct) / m_nsamples;
}

// 聚类算法启动函数
void Kmeans::fit(const float* v_data) {
    float lastLoss = this->m_optTarget;
    for (int i = 0; i < m_maxIters; ++i) {
        this->getDistance(v_data);
        this->updateClusters(v_data);
        if (std::abs(lastLoss - this->m_optTarget) < this->m_epsilon)
            break;
        lastLoss = this->m_optTarget;
        std::cout << "Iters: " << i + 1 << "  current loss : " << m_optTarget
            << std::endl;
    }
}