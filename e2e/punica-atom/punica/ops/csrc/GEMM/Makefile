# Variables
NVCC = nvcc
ARCH = sm_86
STD = c++17
SRC = DenseLayerGEMM_i4_fp8.cu
OUT = DenseLayerGEMM_i4_fp8

# Main target
all: $(OUT)

# Build rule
$(OUT): $(SRC)
	$(NVCC) -arch=$(ARCH) -std=$(STD) $< -o $@

# Clean rule
clean:
	rm -f $(OUT)
