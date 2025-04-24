#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <ctime>
#include "kmeans.h"
#include "kmeansGPU.h"
#include "error.cuh"

// 计算样本数目和特征数目的函数
std::pair<int, int> compute_n_samples_and_features(const std::string& filename) {
    std::ifstream ifs(filename, std::ios::in);
    if (ifs.fail()) {
        std::cerr << "No such file or directory: " << filename << std::endl;
        exit(1);
    }

    std::string line;
    std::pair<int, int> result(0, 0); // 第一个元素是样本数目，第二个元素是特征数目

    // 首先读取第一行来确定特征数目
    if (std::getline(ifs, line)) {
        std::stringstream sstream(line);
        std::string s_fea;
        int fieldCount = 0;

        while (std::getline(sstream, s_fea, ',')) {
            fieldCount++;
        }

        if (fieldCount < 1) {
            std::cerr << "Invalid file format: no features found in the first line." << std::endl;
            ifs.close();
            exit(1);
        }

        result.second = fieldCount - 1; // 特征数目
    } else {
        std::cerr << "The file is empty or has no valid data." << std::endl;
        ifs.close();
        return result;
    }

    // 然后读取所有行来确定样本数目
    ifs.seekg(0, std::ios::beg); // 重置文件指针到文件开头
    int sampleCount = 0;
    while (std::getline(ifs, line)) {
        if (!line.empty()) {
            sampleCount++;
        }
    }

    result.first = sampleCount; // 样本数目

    ifs.close();
    return result;
}

// 读取合成数据集，并将所有数据放到一个一维数组中
// 数据集必须是已经对齐的，以","为分隔符，最后一列为标签的全浮点数数据 
void readCoordinate(
    const std::string& filename,    // 数据集文件名
    float* data,                    // 指向浮点数数组的指针，用于存储读取的特征数据
    int* label,                     // 指向整型数组的指针，用于存储读取的标签数据
    const int n_features            // 特征数量
) {
    std::ifstream ifs(filename, std::ios::in);
    if (ifs.fail()) {
        std::cout << "No such file or directory: " << filename << std::endl;
        exit(1);
    }

    std::string line;
    int n = 0;
    while (std::getline(ifs, line)) {
        if (line.empty()) {
            continue;
        }

        std::stringstream sstream(line);
        int m = 0;
        std::string s_fea;
        int fieldCount = 0;

        while (std::getline(sstream, s_fea, ',')) {
            fieldCount++;
            if (m < n_features) {
                data[n * n_features + m] = std::stod(s_fea);
                m++;
            } else {
                label[n] = std::stoi(s_fea);
            }
        }

        if (fieldCount != n_features + 1) {
            std::cout << "Invalid line format in file: " << filename << std::endl;
            std::cout << "Expected " << n_features + 1 << " fields, but found " << fieldCount << std::endl;
            continue;
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
    const int m_maxIters,   // 最大迭代次数
    const float m_epsilon,     // 目标阈值，两次loss相差超过该值停止迭代
    const int method       // 方法选择，0表示CPU，1表示GPU
) {

    Kmeans* model;

    switch (method) {
    case 0: // CPU
        model = new Kmeans(numClusters, n_features, clusters, n_samples, m_maxIters, m_epsilon);
        break;
    case 1: // GPU
        model = new KmeansGPU(numClusters, n_features, clusters, n_samples, m_maxIters, m_epsilon);
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
// 随机初始化聚类中心
void randomInit(
    float* data,        // 数据集
    int numClusters,    // 聚类中心数量
    int* centers,       // 存储初始聚类中心的数组
    int n_samples       // 样本数量
) {
    // 设置随机种子
    std::srand(std::time(0));

    // 确保 numClusters 不超过数据点数量
    if (numClusters > n_samples) {
        std::cerr << "Error: Number of clusters exceeds the number of data points." << std::endl;
        return;
    }

    // 创建一个随机索引的列表，用于选择初始聚类中心
    int* indices = new int[n_samples];
    for (int i = 0; i < n_samples; ++i) {
        indices[i] = i;
    }
    // 对索引列表进行随机洗牌
    for (int i = n_samples - 1; i > 0; --i) {
        int j = std::rand() % (i + 1);
        std::swap(indices[i], indices[j]);
    }

    // 选择前 numClusters 个随机索引作为初始聚类中心
    for (int i = 0; i < numClusters; ++i) {
        centers[i] = data[indices[i]];
    }

    delete[] indices; // 释放动态分配的内存
}

int main(int argc, char* argv[]) {

    // 可指定的外部参数
    std::string filename = "./synthetic_dataset.csv"; // 数据集文件名
    int numClusters = 4;    // 聚类中心数量
    int m_maxIters = 50;    // 最大迭代次数
    float m_epsilon = 0.01f; // 目标阈值，两次loss相差超过该值停止迭代

    // 解析命令行参数
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--filename") {
            filename = argv[++i];
        } else if (arg == "--numClusters") {
            numClusters = std::stoi(argv[++i]);
        } else if (arg == "--maxIters") {
            m_maxIters = std::stoi(argv[++i]);
        } else if (arg == "--epsilon") {
            m_epsilon = std::stof(argv[++i]);
        }
    }

    // 根据外部参数动态变化的变量
    auto [n_samples, n_features] = compute_n_samples_and_features(filename);  // 计算样本数目和特征数目
    std::cout << "samples: " << n_samples << std::endl;
    std::cout << "features: " << n_features << std::endl;
    const int bufferSize = n_samples * n_features;  // 缓冲区大小
    float *clusters = new float[numClusters * n_features]; // 动态分配内存以存储聚类中心
    int *cidx = new int[numClusters]; // 动态分配内存以存储初始聚类中心的索引
    randomInit(clusters, numClusters, cidx, n_samples); // 随机初始化聚类中心

    for (int i = 0; i < numClusters; ++i) {  // 选择的初始聚类中心的索引
        cidx[i] = i;  // 默认使用样本索引 0, 1, 2, ..., numClusters-1
    }

    float* data = new float[bufferSize];    // 指向浮点数数组的指针，用于存储读取的特征数据
    int* label = new int[bufferSize];       // 指向整型数组的指针，用于存储读取的标签数据
    readCoordinate(filename, data, label, n_features); // 读取数据集
    for (int i = 0; i < numClusters; ++i) { // 将初始聚类中心的坐标从数据集中复制到 clusters 数组中
        for (int j = 0; j < n_features; ++j) {
            clusters[i * n_features + j] = data[cidx[i] * n_features + j];
        }
    }
    std::cout << "********* init clusters **********" << std::endl;
    std::cout << "Using CPU:" << std::endl;
    timing(data, label, clusters, numClusters, n_features, n_samples, m_maxIters, m_epsilon, 0); // 使用 CPU 进行 KMeans 聚类
    std::cout << "Using CUDA:" << std::endl;
    timing(data, label, clusters, numClusters, n_features, n_samples, m_maxIters, m_epsilon, 1); // 使用 GPU 进行 KMeans 聚类

    delete[] data;
    delete[] label;
    delete[] clusters;
    delete[] cidx;

    return 0;
}