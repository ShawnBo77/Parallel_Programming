KERNEL ?= kernel
COMPUTE_CAPABILITY ?= 61

APP_NAME = mandelbrot
OBJDIR = objs
COMMONDIR = common

NVCC = nvcc
CXX = g++
# Flags shared among C++ and CUDA.
CPPFLAGS = -I$(COMMONDIR) -I$(OBJDIR) -O3 -g -fPIC
CXXFLAGS = -O3 -std=c++17 -Wall
# The shorthand `-arch=sm` is equivalent to `-arch=compute -code=sm`.
# Real architecture with `-arch` is allowed only when there's no value for `-code`.
CUDAFLAGS = -rdc=true -arch=sm_$(COMPUTE_CAPABILITY) -Wno-deprecated-gpu-targets

LDLIBS = -lm
CUDALDFLAGS = -rdc=true -arch=sm_$(COMPUTE_CAPABILITY)

PPM_CXX = $(COMMONDIR)/ppm.cpp
PPM_OBJ = $(addprefix $(OBJDIR)/, $(subst $(COMMONDIR)/,, $(PPM_CXX:.cpp=.o)))


default: $(APP_NAME)

dirs:
	mkdir -p $(OBJDIR)/
.PHONY: dirs

clean:
	$(RM) -r $(OBJDIR) *.ppm *~ $(APP_NAME)
.PHONY: clean

OBJS = $(OBJDIR)/main.o $(OBJDIR)/$(KERNEL).o $(OBJDIR)/mandelbrot_serial.o $(OBJDIR)/mandelbrot_thread.o $(PPM_OBJ)

$(APP_NAME): dirs $(OBJS)
	$(NVCC) $(CUDALDFLAGS) -o $@ $(OBJS) mandelbrot_thread_ref.a $(LDLIBS)

$(OBJDIR)/%.o: %.cpp
	$(CXX) $< $(CPPFLAGS) $(CXXFLAGS) -c -o $@

$(OBJDIR)/%.o: $(COMMONDIR)/%.cpp
	$(CXX) $< $(CPPFLAGS) $(CXXFLAGS) -c -o $@

$(OBJDIR)/main.o: $(COMMONDIR)/cycle_timer.h kernel.h

$(OBJDIR)/$(KERNEL).o : $(KERNEL).cu kernel.h
	$(NVCC) $(CUDAFLAGS) $(foreach flag,$(CPPFLAGS),-Xcompiler $(flag)) -c -o $@ $<
