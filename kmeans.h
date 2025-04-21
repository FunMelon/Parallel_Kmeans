#ifndef KMEANS_H
#define KMEANS_H
// Kmeans类
class Kmeans {
    public:
        Kmeans(int numClusters, int numFeatures, float* clusters, int nsamples);
        Kmeans(int numClusters,
               int numFeatures,
               float* clusters,
               int nsamples,
               int maxIters,
               float eplison);
        virtual ~Kmeans();
        virtual void getDistance(const float* v_data);
        virtual void updateClusters(const float* v_data);
        virtual void fit(const float* v_data);
        virtual void saveLabels(const char* filename);
        
        int m_numClusters;  // 类别数量，即k
        int m_numFeatures;  // 特征数量
        float* m_clusters;  // [numClusters, numFeatures]，存储当前各个类的中心点坐标
        int m_nsamples;     // 样本数量
        float m_optTarget;  // 优化目标值，即loss
        int m_maxIters;     // 最大迭代次数
        float m_eplison;    // 目标阈值，两次loss相差超过该值停止迭代
        float* m_distances; // [nsamples, numClusters]，用于存储每个样本到每个类的中心两两之间的距离
        int* m_sampleClasses; // [nsamples, ]，记录每个样本的类比编号

    private:
        Kmeans(const Kmeans& model);
        Kmeans& operator=(const Kmeans& model);
};

#endif  // KMEANS_H