cmake_minimum_required (VERSION 2.8)
project (TrigMuonModules)
set(TRIGMUONMODULES_MAJOR_VERSION 1)
set(TRIGMUONMODULES_MINOR_VERSION 0)
set(TRIGMUONMODULES_PATCH_VERSION 0)
set(TRIGMUONMODULES_VERSION ${TRIGMUONMODULES_MAJOR_VERSION}.${TRIGMUONMODULES_MINOR_VERSION}.${TRIGMUONMODULES_PATCH_VERSION} )
# macro (dump_var arg )
#   message("${arg}=${${arg}}")
# endmacro (dump_var)
if( "${CMAKE_BUILD_TYPE}" STREQUAL "" )
  set(CMAKE_BUILD_TYPE "RelWithDebInfo")
endif()
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  set(COMP "llvm")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  # using GCC
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -ftree-vectorize -ftree-vectorizer-verbose=1 ")
  set(COMP "gcc")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
  # using Intel C++
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -vec-report=1 ")
  set(COMP "icc")
endif()

if ( "$ENV{CMTCONFIG}" STREQUAL "")
  if ( "${CMAKE_BUILD_TYPE}" STREQUAL "Debug" )
    set(REL "dbg")
  else()
    set(REL "opt")
  endif()
  execute_process(COMMAND ${CMAKE_C_COMPILER} -dumpversion OUTPUT_VARIABLE COMP_VERSION)
  string(REGEX MATCHALL "[0-9]+" COMP_VERSION_COMPONENTS ${COMP_VERSION})
  list(GET COMP_VERSION_COMPONENTS 0 COMP_MAJOR)
  list(GET COMP_VERSION_COMPONENTS 1 COMP_MINOR)
  set(CMTCONFIG "${CMAKE_SYSTEM_PROCESSOR}-slc6-${COMP}${COMP_MAJOR}${COMP_MINOR}-${REL}")
else("$ENV{CMTCONFIG}" STREQUAL "")
  set(CMTCONFIG $ENV{CMTCONFIG})
endif("$ENV{CMTCONFIG}" STREQUAL "")
set (CMAKE_VERBOSE_MAKEFILE ON)
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  if( NOT  "$ENV{TestArea}" STREQUAL "")
    set(CMAKE_INSTALL_PREFIX 
      "$ENV{TestArea}/InstallArea/${CMTCONFIG}")
  endif()
  find_package(APE REQUIRED PATHS $ENV{ATLAS_EXTERNAL}/APE/1.0.2/${CMTCONFIG}/)
else()
  if(NOT DEFINED ENV{CMTCONFIG})
    set (CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}/TRIGMUONMODULES/${TRIGMUONMODULES_VERSION}/")
  else()
    set (CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}/${CMTCONFIG}")    
  endif()
  find_package(APE REQUIRED)
endif()

find_package(CUDA 6.0 REQUIRED)
dump_var_ape("COMP_VERSION")
dump_var_ape("CMTCONFIG")
dump_var_ape("CMAKE_INSTALL_PREFIX")
dump_var_ape("CMAKE_INSTALL_PREFIX")
dump_var_ape("CMAKE_BINARY_DIR")
dump_var_ape("APE_INCLUDE_DIR")
include_directories(${TBB_INCLUDES_DIR} ${APE_INCLUDE_DIR})
add_definitions(-std=c++11)
set(CUDA_VERBOSE_BUILD ON)
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc)
set(nvccExtraFlags -Xcompiler=-fPIC -O2 -g; -Xlinker=-fPIC -g; -lineinfo ;-std=c++11)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_20,code=sm_20;-gencode arch=compute_20,code=sm_21;-gencode arch=compute_30,code=sm_30;-gencode arch=compute_35,code=sm_35;-shared;${nvccExtraFlags})

CUDA_ADD_LIBRARY(TrigMuonModule SHARED 
  src/muonTriggerModuleGPU.cu 
  src/muonSimpleWorkGPU.cu 
#  src/simpleIndexing.cu
  src/HTConfigWorkGPU.cu)

dump_var_ape(kernels)
dump_var_ape(CUDA_NVCC_FLAGS)
add_library(TrigMuonModuleCPU SHARED  
  src/muonTriggerModuleCPU.cxx 
  src/muonSimpleWork.cxx 
  src/HTConfigWork.cxx)
set_target_properties(TrigMuonModuleCPU PROPERTIES COMPILE_FLAGS " -I${CUDA_INCLUDE_DIRS} ")
target_link_libraries(TrigMuonModuleCPU ${APE_LIBS} )
install(TARGETS TrigMuonModuleCPU TrigMuonModule
  LIBRARY DESTINATION "lib"
  ARCHIVE DESTINATION "lib"
  RUNTIME DESTINATION "bin"
)
