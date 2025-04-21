# Makefile
# 定义编译器
CXX = g++

# 定义编译选项
CXXFLAGS = -Wall -std=c++11

# 定义目标文件
TARGET = main.out

# 定义源文件和目标文件
SRC = main.cpp kmeans.cpp
OBJ = $(SRC:.cpp=.o)

# 定义默认目标
all: clean $(TARGET)

# 定义目标文件的构建规则
$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

# 定义对象文件的构建规则
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# 定义清理规则
clean:
	rm -f $(OBJ) $(TARGET)