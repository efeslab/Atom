# Variables
NVCC = nvcc
ARCH = sm_86
STD = c++17
SRC = test_Reorder.cu
OUT = test_Reorder

# Main target
all: $(OUT)

# Build rule
$(OUT): $(SRC)
	$(NVCC) -arch=$(ARCH) -std=$(STD) $< -o $@

# Clean rule
clean:
	rm -f $(OUT)
