# --- Configuration ---
# The CUDA compiler
NVCC = nvcc

# Compiler flags:
# -O3       : Maximum optimization for speed
# -rdc=true : Allows __device__ functions to be linked across multiple .cu files
# -arch     : Target GPU architecture (sm_70 = Volta, sm_75 = Turing, sm_80 = Ampere, sm_89 = Ada/RTX 40 series)
# -I.       : Look for headers (like helper_math.h) in the current directory
NVCC_FLAGS = -O3 -rdc=true -arch=sm_80 -I../../include

# --- Files ---
# List all of your .cu source files here
SRCS = test.cu

# This automatically creates a list of object files (.o) from your source files (.cu)
OBJS = $(SRCS:.cu=.o)

# The name of your final executable program
TARGET = test

# --- Rules ---
# The default rule that runs when you just type 'make'
all: $(TARGET)

# Rule to link all the object files into the final executable
$(TARGET): $(OBJS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

# Rule to compile individual .cu files into .o object files
%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Rule to clean up the compiled files
clean:
	rm -f $(OBJS) $(TARGET)