#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include "kmeans.h"

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

int main() {
    const int n_features = 100;  // 特征数量
    int n = 0;                   // 样本数量，将由 readCoordinate 函数填充
    float* data = new float[1000 * n_features]{0.0};  // 存储特征数据
    int* label = new int[1000]{0};                    // 存储标签数据
    std::cout << "reading dataset..." << std::endl;
    readCoordinate(data, label, n_features, n);  // 读取数据集
    std::cout << "reading dataset done!" << std::endl;
    const int numClusters = 4;  // 聚类数量
    Kmeans kmeans(numClusters, n_features, data, n);
    kmeans.fit(data);
}