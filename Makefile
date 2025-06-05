NVCC = nvcc

CUDA_FLAGS = -c -O2 -arch=sm_75 -allow-unsupported-compiler

CUDA_SYSTEM_INCLUDE_DIR = /opt/cuda/include/

PROJECT_CUDA_HEADER_DIR = src/cuda

CUDA_SOURCE_DIR = src/cuda
CUDA_SOURCE_FILE = kernel.cu

CUDA_C_API_HEADER = $(PROJECT_CUDA_HEADER_DIR)/kernel.h

CUDA_OBJECT_FILE = kernel.o

ZIG = zig

fmt:
	$(ZIG) fmt .

$(CUDA_OBJECT_FILE): $(CUDA_SOURCE_DIR)/$(CUDA_SOURCE_FILE) $(CUDA_C_API_HEADER)
	$(NVCC) $(CUDA_FLAGS) \
		-I$(CUDA_SYSTEM_INCLUDE_DIR) \
		-I$(PROJECT_CUDA_HEADER_DIR) \
		$(CUDA_SOURCE_DIR)/$(CUDA_SOURCE_FILE) -o $(CUDA_OBJECT_FILE)


build: $(CUDA_OBJECT_FILE)
	clear && $(ZIG) build -Dtarget=native -Dcpu=native

build-fast: $(CUDA_OBJECT_FILE)
	clear && $(ZIG) build -Doptimize=ReleaseFast -Dtarget=native -Dcpu=native

build-safe: $(CUDA_OBJECT_FILE)
	clear && $(ZIG) build -Doptimize=ReleaseSafe -Dtarget=native -Dcpu=native

build-small: $(CUDA_OBJECT_FILE)
	clear && $(ZIG) build -Doptimize=ReleaseSmall -Dtarget=native -Dcpu=native

run:
	./zig-out/bin/MicroRush

run-metrics:
	./zig-out/bin/MicroRush --metrics

clean:
	rm -rf zig-out
	rm -rf .zig-cache
	rm -f $(CUDA_OBJECT_FILE)

.PHONY: fmt build run clean

