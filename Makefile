NVCC_DIR := $(shell which nvcc 2> /dev/null)
ifneq (,$(findstring nvcc,$(NVCC_DIR)))

APE_INSTALL_DIR = /afs/cern.ch/atlas/offline/external/APE/1.1.1/x86_64-slc6-gcc48-opt
TBB_INSTALL_DIR = /cvmfs/atlas.cern.ch/repo/sw/software/x86_64-slc6-gcc48-opt/20.1.4/AtlasCore/20.1.4/InstallArea/x86_64-slc6-gcc48-opt

CUDA_INSTALL_DIR= $(dir $(NVCC_DIR))#/usr/local/cuda-6.5/

CC   =/cvmfs/atlas.cern.ch/repo/sw/atlas-gcc/481/x86_64/slc6/gcc48/bin/c++
#CC   =g++
CUDA = $(NVCC_DIR)

USE_CUDA =true
else #Lxplus
# APE TBB g++ and CUDA settings for lxplus:
APE_INSTALL_DIR = /afs/cern.ch/atlas/offline/external/APE/1.1.1/x86_64-slc6-gcc48-opt
TBB_INSTALL_DIR = /afs/cern.ch/atlas/offline/external/tbb/tbb40_233oss

CC   =g++

USE_CUDA =false

endif

#search for local APE
ifneq (,$(findstring APE,$(TestArea)/InstallArea/x86_64-slc6-gcc48-opt/include/APE))
APE_INCLUDE_DIR=$(TestArea)/InstallArea/x86_64-slc6-gcc48-opt/include
APE_LIB_DIR=$(TestArea)/InstallArea/x86_64-slc6-gcc48-opt/lib
else
APE_INCLUDE_DIR=$(APE_INSTALL_DIR)/include
APE_LIB_DIR=$(APE_INSTALL_DIR)/lib
endif

#USE_CUDA =false
#CUFLAGS = -DTHRUST_DEBUG -O2 -g -DNDEBUG -std=c++11 -gencode arch=compute_37,code=sm_37 -Xcompiler "-O2 -g -DNDEBUG -Wall -fPIC -std=c++11"

CFLAGS = -O2 -g -DNDEBUG -Wall -fPIC -std=c++11
CFLAGS += -I $(APE_INSTALL_DIR)/include
CFLAGS += -I $(TBB_INSTALL_DIR)/include  
CCFLAGS = -O2 -g -DNDEBUG -ccbin c++ -std=c++11 -gencode arch=compute_37,code=sm_37 -Xcompiler "-O2 -g -DNDEBUG -Wall -fPIC -std=c++11"
CCFLAGS += -I $(APE_INSTALL_DIR)/include
CCFLAGS += -I $(TBB_INSTALL_DIR)/include
CCFLAGS += -I $(CUDA_INSTALL_DIR)/../samples/common/inc
CUFLAGS = -O2 -g -DNDEBUG -std=c++11 -gencode arch=compute_37,code=sm_37 --default-stream per-thread -Xcompiler "-O2 -g -DNDEBUG -Wall -fPIC -std=c++11" #-Xptxas="-dlcm=ca" - -ptx - --ptxas-options=-v
CUFLAGS += -I $(CUDA_INSTALL_DIR)/../samples/common/inc
LDFLAGS = -L $(APE_INSTALL_DIR)/lib
LDFLAGS += -L $(CUDA_INSTALL_DIR)/../lib64/
LDFLAGS += -L $(CUDA_INSTALL_DIR)/../lib64/stubs
LDFLAGS += -lAPEContainer
ifeq ($(USE_CUDA),true)
LDFLAGS += -lcuda -lcudart
endif

# package folders
SOURCES_DIR   = src/
BINARIES_DIR  = bin/
LIBRARIES_DIR = lib/
SHARE_DIR     = share/

ifeq ($(USE_CUDA),true)
CULIBRARY:= libmuonTriggerModuleGPU.so
CULINK := linkMuonDevice.o
else
CLIBRARY:= libmuonTriggerModuleCPU.so
endif

CSOURCES := HTConfigWork.cxx muonSimpleWork.cxx
CMODULE_SOURCE := muonTriggerModuleCPU.cxx
CCSOURCES := muonTriggerModuleGPU.cxx muonSimpleWorkGPU.cxx
#CCSOURCES := muonTriggerModuleGPU.cxx HTConfigWorkGPU.cxx muonSimpleWorkGPU.cxx
#CUSOURCES := NewVotingKernelGPU.cu muonDeviceWrapperGPU.cu
CUSOURCES := HoughKernelsGPU.cu muonDeviceWrapperGPU.cu
COBJECTS := $(CSOURCES:.cxx=.o)
CCOBJECTS := $(CCSOURCES:GPU.cxx=Host.o)
CUOBJECTS := $(CUSOURCES:GPU.cu=Dev.o)
COBJECTS := $(addprefix $(BINARIES_DIR),$(COBJECTS))
CCOBJECTS := $(addprefix $(BINARIES_DIR),$(CCOBJECTS))
CUOBJECTS := $(addprefix $(BINARIES_DIR),$(CUOBJECTS))
CLIBRARY := $(addprefix $(LIBRARIES_DIR),$(CLIBRARY))
CULIBRARY := $(addprefix $(LIBRARIES_DIR),$(CULIBRARY))
CULINK    := $(addprefix $(BINARIES_DIR),$(CULINK))
CMODULE_OBJECT := $(CMODULE_SOURCE:.cxx=.o)

all: $(CULIBRARY) $(CLIBRARY) install

##################### CUDA GPU code ###################################   
$(CULIBRARY): $(CUOBJECTS) $(CCOBJECTS) $(CULINK)
	mkdir -p $(LIBRARIES_DIR)
	$(CC) $(LDFLAGS) -shared -Wl,-soname,$@ -o $@ $(CUOBJECTS)  $(CCOBJECTS) $(CULINK) -lcudadevrt

$(BINARIES_DIR)%Host.o: $(SOURCES_DIR)%GPU.cxx
	mkdir -p $(BINARIES_DIR)
	$(CUDA) $(CCFLAGS) -c $< -o $@

$(BINARIES_DIR)%Dev.o: $(SOURCES_DIR)%GPU.cu
	mkdir -p $(BINARIES_DIR)
	$(CUDA) $(CUFLAGS) -dc $< -o $@

$(CULINK): $(CUOBJECTS)
	$(CUDA) $(CUFLAGS) -dlink $(CUOBJECTS) -o $@


##################### OLD CPU code ###################################
$(CLIBRARY): $(LIBRARIES_DIR)lib%ModuleCPU.so: $(COBJECTS) $(BINARIES_DIR)%ModuleCPU.o 
	mkdir -p $(LIBRARIES_DIR)
	$(CC) $(LDFLAGS) -shared -Wl,-soname,$@ -o $@ $^

#
##compiles every .o agains only($<) the respective .cxx
## alternative: $(OBJECTS): %.o: %.cxx
$(BINARIES_DIR)%.o: $(SOURCES_DIR)%.cxx
	mkdir -p $(BINARIES_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

DEBUG:
	@echo $(NVCC_DIR)
ifneq (,$(findstring nvcc,$(NVCC_DIR)))
	@echo $(NVCC_DIR)
	@echo $(CUDA_INSTALL_DIR)
	@echo $(APE_INSTALL_DIR)
	@echo $(LDFLAGS)
else
	@echo "Not found"
endif

install: $(CULIBRARY) $(CLIBRARY)
	cp  $(LIBRARIES_DIR)*.so $(APE_LIB_DIR)
ifndef CMTCONFIG
	cp $(SHARE_DIR)*.yaml   $(TestArea)/InstallArea/share/
else
	mkdir -p $(TestArea)/InstallArea/$(CMTCONFIG)/share/
	cp $(SHARE_DIR)*.yaml   $(TestArea)/InstallArea/$(CMTCONFIG)/share/
endif

clean:
	rm -f $(BINARIES_DIR)*.o $(LIBRARIES_DIR)*.so