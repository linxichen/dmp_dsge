# Paths for Linux CUDA
ICUDA    = /usr/local/cuda-7.0/include
LCUDA    = /usr/local/cuda-7.0/lib64
LCUDA65  = /usr/local/cuda-6.5/lib64
ICUDA_MAC = /Developer/NVIDIA/CUDA-7.0/include
LCUDA_MAC = /Developer/NVIDIA/CUDA-7.0/lib
ICPP_MAC = /usr/local/include
LCPP_MAC = /usr/local/lib
ILAPACK = /usr/include/lapacke

SDIR     = .
IDIR     = .
LDIR     = .

# Compiler for CUDA
NVCC      = nvcc

# CUDA compiling options
NVCCFLAGS =  --std=c++11 -m64 -arch sm_30 #-use_fast_math

# Compiler for C code
CXX       = g++ 

# Standard optimization flags to C++ compiler
CXXFLAGS  = -I$(ICUDA) -I$(ICUDA_MAC) -I$(ICPP_MAC) -I$(ILAPACK)

# Add CUDA libraries to C++ compiler linking process
LDFLAGS  += -lcublas -lcurand -lcudart -L$(LCUDA) -L$(LCUDA_MAC) -L$(LCUDA65) -L$(LCPP_MAC)

# List Executables and Objects
EXEC = vfi_hosios

all : $(EXEC)

# Link objects from CUDA and C++ codes
vfi_hosios : vfi_hosios.o
	$(CXX) --std=c++11 -o $@ $? $(LDFLAGS)
adrian : adrian.o
	$(CXX) --std=c++11 -o $@ $? $(LDFLAGS)

# Compile CUDA code
vfi_hosios.o : vfi_hosios.cu 
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) -c $<  
adrian.o : adrian.cu 
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) -c $<  

clean :
	rm -f *.o
	rm -f core core.*

veryclean :
	rm -f *.o
	rm -f core core.*
	rm -f $(EXEC)

runvfi : vfi_hosios
	./vfi_hosios

runadrian: adrian
	./adrian noisy
