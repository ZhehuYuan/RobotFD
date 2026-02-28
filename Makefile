CXX = g++
NVCC = nvcc

CXXFLAGS = -O3 -std=c++17 -Wall -Iinclude -I/home/yuan.645/local/cuda/include
NVCCFLAGS = -O3 -Iinclude -arch=sm_70 -I/home/yuan.645/local/cuda/include -lineinfo -Xptxas -O3

SRC_DIR = src
OBJ_DIR = obj

CPP_SRCS = $(wildcard $(SRC_DIR)/*.cpp)
CUDA_SRCS = $(wildcard $(SRC_DIR)/*.cu)

CPP_OBJS = $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(CPP_SRCS))
CUDA_OBJS = $(patsubst $(SRC_DIR)/%.cu, $(OBJ_DIR)/%.o, $(CUDA_SRCS))

ALL_OBJS = $(CPP_OBJS) $(CUDA_OBJS)

TARGET = fd

all: $(TARGET)

$(TARGET): $(ALL_OBJS)
	$(CXX) -o $@ $^ -L/home/yuan.645/local/cuda/lib64 -lcudart \

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

clean:
	rm -rf $(OBJ_DIR) $(TARGET)

