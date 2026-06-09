# --- Configuration ---
# The CUDA compiler
NVCC = nvcc

# Compiler flags:
# -O3       : Maximum optimization for speed
# -rdc=true : Allows __device__ functions to be linked across multiple .cu files
# -arch     : Target GPU architecture (sm_70 = Volta, sm_75 = Turing, sm_80 = Ampere, sm_89 = Ada/RTX 40 series)
# -I.       : Look for headers (like helper_math.h) in the current directory
NVCC_FLAGS = -O3 -rdc=true --extended-lambda -arch=sm_80 -Iinclude

# --- Files ---
# CUDA source files
CU_SRCS = src/main.cu
# C++ source files
CPP_SRCS = src/utils.cpp src/Domain.cpp

CU_OBJS = $(CU_SRCS:.cu=.o)
CPP_OBJS = $(CPP_SRCS:.cpp=.o)
OBJS = $(CU_OBJS) $(CPP_OBJS)

# To save changes also in include files
HEADERS = $(wildcard include/*.h include/*.cuh)

# The name of your final executable program
TARGET = sim 

# --- Rules ---
# The default rule that runs when you just type 'make'
all: $(TARGET)

# Rule to link all the object files into the final executable
$(TARGET): $(OBJS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

# Rule to compile individual .cu files into .o object files
%.o: %.cu $(HEADERS) 
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

%.o: %.cpp $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ 


# Rule to clean up the compiled files
clean:
	rm -f $(OBJS) $(TARGET)
