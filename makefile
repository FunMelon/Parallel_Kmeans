# 定义编译器
CXX = g++
NVCC = nvcc

# 定义编译选项
CXXFLAGS = -Wall -std=c++11
NVCCFLAGS = -arch=sm_52 -std=c++11 -I/home/funmelon/cuda-samples/Common -I/usr/local/cuda/include

# 定义链接选项
LDFLAGS = -L/usr/local/cuda/lib64 -lcudart

# 定义目标文件
TARGET = main.out

# 定义源文件和目标文件
SRC = main.cu kmeans.cpp kmeansGPU.cu
OBJ = main.o kmeans.o kmeansGPU.o

# 定义默认目标
all: clean $(TARGET)

# 使用 nvcc 进行链接，确保处理 CUDA 库
$(TARGET): $(OBJ)
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o $@ $^

# 定义对象文件的构建规则
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# 定义清理规则
clean:
	rm -f $(OBJ) $(TARGET) cluster_labels_cpp.csv cluster_labels_parallel_cpp.csv
