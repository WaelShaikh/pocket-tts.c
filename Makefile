# Supertonic 2 TTS (WIP)
# Makefile

CC = gcc
CFLAGS_BASE = -Wall -Wextra -O3 -march=native -ffast-math
LDFLAGS = -lm
BLAS_LIBS ?= -lopenblas
CUDA_LIBS ?= -lcudart -lcublas -lnvrtc -lcuda

# ONNX Runtime settings
ORT_VER = 1.16.3
ORT_DIR = onnxruntime-linux-x64-$(ORT_VER)
ORT_INCLUDE = -I$(ORT_DIR)/include
ORT_LIB_DIR = $(ORT_DIR)/lib
ORT_LIBS = -L$(ORT_LIB_DIR) -lonnxruntime -Wl,-rpath,'$$ORIGIN/$(ORT_LIB_DIR)'

SRCS = st_audio.c st_safetensors.c st_spm.c st_kernels.c
OBJS = $(SRCS:.c=.o)
CUDA_OBJS = $(OBJS) st_cuda.o
MAIN = main.c
TARGET = supertonic
LIB = libsupertonic.a

.PHONY: all clean help cpu cpu-opt onnx lib info blas cuda

all: help

help:
	@echo "Supertonic 2 TTS (WIP) - Build Targets"
	@echo ""
	@echo "  make cpu      - Pure C, no dependencies"
	@echo "  make cpu-opt  - Pure C + OpenMP"
	@echo "  make onnx     - Link against ONNX Runtime (requires download_ort.sh)"
	@echo "  make blas     - OpenBLAS accelerated"
	@echo "  make cuda     - NVIDIA CUDA + cuBLAS accelerated"
	@echo ""
	@echo "Other targets:"
	@echo "  make clean    - Remove build artifacts"
	@echo "  make info     - Show build configuration"
	@echo "  make lib      - Build static library"
	@echo ""

# =============================================================================
# Backend: cpu (pure C, no deps)
# =============================================================================
cpu: CFLAGS = $(CFLAGS_BASE) -DCPU_BUILD
cpu: clean $(TARGET)
	@echo ""
	@echo "Built with CPU backend (pure C)"

# =============================================================================
# Backend: cpu-opt (pure C + OpenMP)
# =============================================================================
cpu-opt: CFLAGS = $(CFLAGS_BASE) -DCPU_BUILD -fopenmp
cpu-opt: LDFLAGS += -fopenmp
cpu-opt: clean $(TARGET)
	@echo ""
	@echo "Built with CPU optimized backend (OpenMP)"

# =============================================================================
# Backend: ONNX Runtime
# =============================================================================
onnx: CFLAGS = $(CFLAGS_BASE) -DST_USE_ONNX $(ORT_INCLUDE)
onnx: LDFLAGS += $(ORT_LIBS)
onnx: clean $(TARGET)
	@echo ""
	@echo "Built with ONNX Runtime backend"

# =============================================================================
# Backend: BLAS (OpenBLAS)
# =============================================================================
blas: CFLAGS = $(CFLAGS_BASE) -DST_USE_BLAS
blas: LDFLAGS = -lm $(BLAS_LIBS)
blas: clean $(TARGET)
	@echo ""
	@echo "Built with BLAS backend (OpenBLAS)"

# =============================================================================
# Backend: CUDA (cuBLAS)
# =============================================================================
cuda: CFLAGS = $(CFLAGS_BASE) -DST_USE_CUDA
cuda: LDFLAGS = -lm $(CUDA_LIBS)
cuda: clean $(CUDA_OBJS) main.o
	$(CC) $(CFLAGS) -o $(TARGET) $(CUDA_OBJS) main.o $(LDFLAGS)
	@echo ""
	@echo "Built with CUDA backend (cuBLAS)"

# =============================================================================
# Build rules
# =============================================================================
$(TARGET): $(OBJS) main.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

st_cuda.o: st_cuda.c st_cuda.h

lib: $(LIB)

$(LIB): $(OBJS)
	ar rcs $@ $^

%.o: %.c st_safetensors.h st_audio.h st_spm.h st_kernels.h st_cuda.h
	$(CC) $(CFLAGS) -c -o $@ $<

main.o: main.c st_kernels.h st_safetensors.h st_audio.h st_spm.h
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJS) st_cuda.o main.o $(TARGET) $(LIB)

info:
	@echo "Compiler: $(CC)"
	@echo "CFLAGS:   $(CFLAGS_BASE)"

# =============================================================================
# Dependencies
# =============================================================================
st_audio.o: st_audio.c st_audio.h
st_safetensors.o: st_safetensors.c st_safetensors.h
st_spm.o: st_spm.c st_spm.h
st_kernels.o: st_kernels.c st_kernels.h
