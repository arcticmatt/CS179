CC = /usr/bin/g++

LD_FLAGS = -lrt

CUDA_PATH       ?= /usr/local/cuda
CUDA_INC_PATH   ?= $(CUDA_PATH)/include
CUDA_BIN_PATH   ?= $(CUDA_PATH)/bin
CUDA_LIB_PATH   ?= $(CUDA_PATH)/lib

# CUDA code generation flags
GENCODE_FLAGS   := -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35

# Common binaries
NVCC            ?= $(CUDA_BIN_PATH)/nvcc

# OS-specific build flags
ifeq ($(shell uname),Darwin)
        LDFLAGS       := -Xlinker -rpath $(CUDA_LIB_PATH) -L$(CUDA_LIB_PATH) -lcudart -lcufft
        CCFLAGS           := -arch $(OS_ARCH)
else
        ifeq ($(OS_SIZE),32)
                LDFLAGS   := -L$(CUDA_LIB_PATH) -lcufft -lcudart
                CCFLAGS   := -m32
        else
                CUDA_LIB_PATH := $(CUDA_LIB_PATH)64
                LDFLAGS       := -L$(CUDA_LIB_PATH) -lcufft -lcudart
                CCFLAGS       := -m64
        endif
endif


# OS-architecture specific flags
ifeq ($(OS_SIZE),32)
      NVCCFLAGS := -m32 -lcufft
else
      NVCCFLAGS := -m64 -lcufft
endif

TARGETS = xray_ct_recon

all: $(TARGETS)

xray_ct_recon: ta_utilities.cpp xray_ct_recon.o 
	$(CC) $^ -o $@ -O3 $(LDFLAGS) -Wall -I$(CUDA_INC_PATH)

xray_ct_recon.o: xray_ct_recon.cu
	$(NVCC) $(NVCCFLAGS) -O3 $(EXTRA_NVCCFLAGS) $(GENCODE_FLAGS) -I$(CUDA_INC_PATH) -o $@ -c $<


clean:
	rm -f *.o $(TARGETS)

again: clean $(TARGETS)
