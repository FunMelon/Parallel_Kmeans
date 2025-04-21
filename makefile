# Makefile

# 定义编译器
NVCC = nvcc

# 定义编译选项
CXXFLAGS = -Wall -std=c++11
NVCCFLAGS = -arch=sm_52 -std=c++11 -I/home/funmelon/cuda-samples/Common -I/usr/local/cuda/include

# 定义链接选项
LDFLAGS = -L/usr/local/cuda/lib64 -lcudart

# 定义目标文件
TARGET = main.out

# 定义源文件和目标文件
SRC = main.cpp kmeans.cpp cuda_func.cu kmeansGPU.cpp
OBJ = main.o kmeans.o kmeansGPU.o cuda_func.o

# 定义默认目标
all: clean $(TARGET)

# 定义目标文件的构建规则
$(TARGET): $(OBJ)
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o $@ $^

# 定义对象文件的构建规则
%.o: %.cpp
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# 清理规则
clean:
	rm -f $(OBJ) $(TARGET)
