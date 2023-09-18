# Variables
NVCC = nvcc
CUDA_FLAGS = -arch=sm_37 # Tesla K80 GPU architecture
TARGET = mandelbrot
SRCDIR = src
BUILDDIR = build
SOURCES = $(wildcard $(SRCDIR)/*.cu)

# Build rule
$(BUILDDIR)/$(TARGET): $(SOURCES)
	@mkdir -p $(BUILDDIR)
	$(NVCC) $(CUDA_FLAGS) $(SOURCES) -o $(BUILDDIR)/$(TARGET)

# Phony targets
.PHONY: clean run

# Clean target
clean:
	rm -rf $(BUILDDIR)

# Run target
run: $(BUILDDIR)/$(TARGET)
	./$(BUILDDIR)/$(TARGET)
