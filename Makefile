# Compiler to gcc
CC = gcc 
# Find include/, shows warnings, debug info
CFLAGS = -Iinclude -Wall -Wextra -g -O2
# Finds all .c in src/
SRC = $(wildcard src/*.c) 
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
	$(CC) $(CFLAGS) $(SRC) $(TEST) -o $(TARGET)

# Deletes build/
clean: 
	rm -rf $(BUILD_DIR)
	