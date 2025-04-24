# 北京理工并行编程原理与实践作业 —— kmeans算法的并行化
在python.ipynb中的第一段代码块是生成合成数据集的代码。数据集要求每个样本1行，前 n - 1 列元素为维度特征，第n列元素为类别标签，类之间的分隔符为``,``。使用其他数据集需要在外部指定参数
推荐使用make语句对C++代码进行编译，可能需要在makefile中：
- 根据显卡指定nvcc编译计算能力选项。
- 保证可以链接cuda库和cuda sample
```shell
make
```
## 本实验实现功能
- 数据合成代码
- kmeans实现代码
  - 无并行的C++代码
  - 使用sklearn第三方库的python代码
  - 并行的C++代码
- 结果可视化代码
## 运行代码
```shell
 ./main.out --filename ./synthetic_datast.csv --numClusters 4 --maxIters 50 --epsilon 0.01 
```
## 使用nvprof工具进行性能分析
```shell
nvprof ./main.out
```
## 关于cuda库
### 在编译中引入cuda库
参考的版本是cuda 12.6，需要在makefile中需要为nvcc编译器引入cuda和cuda sample的地址(helper_cuda.h)。
路径如下，仅供参考
```shell
NVCCFLAGS = -arch=sm_75 -std=c++17 -I/home/funmelon/cuda-samples/Common -I/usr/local/cuda/include
```
### vscode代码检查无法识别cuda库
需要在c_cpp_properities.json中需要增加对cuda和cuda sample(helper_cuda.h)的include。
路径如下，仅供参考。之后vscode的C++的代码检查就不会报错。
```
"/usr/local/cuda/include",
"/home/funmelon/cuda-samples/Common"
```
## 参考代码来自
[知乎用户“后来”](https://www.zhihu.com/people/cai-wan-xian-14)