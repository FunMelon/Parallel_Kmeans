#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include "kmeans.h"
#include "kmeansGPU.h"
#include "error.cuh"

// 读取合成数据集，并将所有数据放到一个一维数组中
// data: 指向浮点数数组的指针，用于存储读取的特征数据
// label: 指向整型数组的指针，用于存储读取的标签数据
// n_features: 每个样本的特征数量
// n: 引用参数，用于存储读取的样本数量
void readCoordinate(float* data, int* label, const int n_features, int& n) {
    std::ifstream ifs;
    ifs.open("./synthetic_dataset.csv", std::ios::in);
    if (ifs.fail()) {
        std::cout << "No such file or directory: synthetic_dataset.csv"
            << std::endl;
        exit(1);
    }
    std::string line;
    while (std::getline(ifs, line)) {
        std::stringstream sstream(line);
        if (line.empty())
            continue;
        int m = 0;  // 跟踪当前行中已解析的字段数量
        std::string s_fea;
        while (std::getline(sstream, s_fea, ',')) {
            if (m < n_features)
                data[n * n_features + m] =
                std::stod(s_fea);  // 将前 n_features
            // 个字段（特征值）转换为浮点数，并存储到
            // data 数组中。
            else
                label[n] =
                std::stoi(s_fea);  // 将第 n_features
            // 个字段（标签）转换为整数，并存储到
            // label 数组中。
            m++;
        }
        n++;
    }
    ifs.close();
}

// timing 函数用于测量 KMeans 聚类算法的运行时间
void timing(
    float* data,            // 指向浮点数数组的指针，用于存储读取的特征数据
    int* label,             // 指向整型数组的指针，用于存储读取的标签数据
    float* clusters,        // 存储初始聚类中心的数组
    const int numClusters,  // 聚类中心数量
    const int n_features,   // 特征数量
    const int n_samples,    // 样本数量
    const int method        // 方法选择，0表示CPU，1表示GPU
) {

    Kmeans* model;

    switch (method) {
    case 0: // CPU
        model = new Kmeans(numClusters, n_features, clusters, n_samples, 50, 0.1);
        break;
    case 1: // GPU
        model = new KmeansGPU(numClusters, n_features, clusters, n_samples, 50, 0.1);
        break;
    default:
        std::cout << "method not supported!" << std::endl;
        break;
    }

    std::cout << "*********starting fitting*********" << std::endl;

    cudaEvent_t start, stop;    // CUDA事件，用于测量时间
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventRecord(start));
    cudaEventQuery(start);
    model->fit(data);

    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    float elapsedTime;
    CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));

    printf("Time = %g ms.\n", elapsedTime);

    std::cout << "********* final clusters**********" << std::endl;
    std::cout << "*********    accuracy  **********" << std::endl;
    std::cout << "model accuracy : " << model->accuracy(label) << std::endl;

    std::cout << "*********    result saving   **********" << std::endl;
    model->saveLabels();
    std::cout << "*********    result saving done   **********" << std::endl;
    delete model;
}

int main(int argc, char* argv[]) {
    int N = 0;  // 样本数量
    int n_features = 100;   // 特征数量
    const int bufferSize = 10000 * n_features;  // 缓冲区大小
    float* data = new float[bufferSize];    // 指向浮点数数组的指针，用于存储读取的特征数据
    int* label = new int[bufferSize];       // 指向整型数组的指针，用于存储读取的标签数据
    readCoordinate(data, label, n_features, N); // 读取数据集
    std::cout << "num of samples : " << N << std::endl;
    int cidx[] = { 1, 3, 6, 8 };    // 选择的初始聚类中心的索引
    int numClusters = 4;    // 聚类中心数量
    float clusters[400] = { 0 };    // 存储初始聚类中心的数组
    for (int i = 0; i < numClusters; ++i) { // 将初始聚类中心的坐标从数据集中复制到 clusters 数组中
        for (int j = 0; j < n_features; ++j) {
            clusters[i * n_features + j] = data[cidx[i] * n_features + j];
        }
    }
    std::cout << "********* init clusters **********" << std::endl;
    std::cout << "Using CPU:" << std::endl;
    timing(data, label, clusters, numClusters, n_features, N, 0);
    std::cout << "Using CUDA:" << std::endl;
    timing(data, label, clusters, numClusters, n_features, N, 1);

    delete[] data;
    delete[] label;

    return 0;
}