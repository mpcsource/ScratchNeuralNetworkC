# Compiler to gcc
CC = gcc 
NVCC = nvcc
# Find include/, shows warnings, debug info
CFLAGS = -Iinclude -g -O2
# Finds all .c in src/
SRC = $(wildcard src/*.cu) 
# Path to test source file
TEST = tests/test.c
# Build directory
BUILD_DIR = build
# Executable directory
TARGET = $(BUILD_DIR)/test 

# Builds and executes
all: $(BUILD_DIR) $(TARGET) 
	$(TARGET)

# Creates build/
$(BUILD_DIR): 
	mkdir -p $(BUILD_DIR)

# Compiles source and test files and links to executable
$(TARGET): $(SRC) $(TEST) 
	$(NVCC) $(CFLAGS) $(SRC) $(TEST) -o $(TARGET) -lcublas

# Deletes build/
clean: 
	rm -rf $(BUILD_DIR)
	