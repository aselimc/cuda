# CUDA compiler and flags
NVCC = nvcc
CFLAGS = -O3 -std=c++17
CUDA_FLAGS = -arch=sm_60 # Adjust for your GPU architecture

# Directories
SRC_DIR = examples
BUILD_DIR = build

# Find all source files
CU_SOURCES = $(wildcard $(SRC_DIR)/**/*.cu $(SRC_DIR)/*.cu)
CPP_SOURCES = $(wildcard $(SRC_DIR)/**/*.cpp $(SRC_DIR)/*.cpp)
C_SOURCES = $(wildcard $(SRC_DIR)/**/*.c $(SRC_DIR)/*.c)

# Create corresponding target paths
CU_TARGETS = $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%,$(CU_SOURCES))
CPP_TARGETS = $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%,$(CPP_SOURCES))
C_TARGETS = $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%,$(C_SOURCES))

# Combined targets
TARGETS = $(CU_TARGETS) $(CPP_TARGETS) $(C_TARGETS)

# Default rule
all: $(TARGETS)

# Rule to build CUDA files
$(BUILD_DIR)/%: $(SRC_DIR)/%.cu
	@mkdir -p $(@D)
	$(NVCC) $(CFLAGS) $(CUDA_FLAGS) $< -o $@

# Rule to build C++ files
$(BUILD_DIR)/%: $(SRC_DIR)/%.cpp
	@mkdir -p $(@D)
	$(NVCC) $(CFLAGS) $< -o $@

# Rule to build C files
$(BUILD_DIR)/%: $(SRC_DIR)/%.c
	@mkdir -p $(@D)
	$(NVCC) $(CFLAGS) $< -o $@

# Clean rule
clean:
	rm -rf $(BUILD_DIR)

# Make sure build directory exists
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

.PHONY: all clean