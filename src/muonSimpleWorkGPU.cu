#include "../include/muonSimpleWorkGPU.cuh"
#include "../../../DataStructures/TrigMuonDataStructs/TrigMuonDataStructs/TrigMuonAccelerationEDM.h"
#include "tbb/tick_count.h"
#include <cstring>
#include <cmath>
#include <iostream>
#include <unistd.h>
#include "APE/BufferAccessor.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>       // helper functions for CUDA timing and initialization
#include <helper_functions.h>  // helper functions for timing, string parsing

#include "../include/HTHelpers.cuh"
#include "../include/VotingKernel.cuh"
#include "../include/MaxFinderKernel.cuh"

#define BENCHMARK_SIZE  10
texture<float, 2, cudaReadModeElementType> TexSrc;

APE::muonSimpleWorkGPU::muonSimpleWorkGPU(const HoughTransformContextGPU& ctx, std::shared_ptr<APE::BufferContainer> data) :
  m_context(0),
  m_input(data) {

  m_context = new HoughTransformContextGPU(ctx);
  m_buffer = std::make_shared<APE::BufferContainer>(sizeof(SIMPLE_OUTPUT_DATA));//output data
  m_buffer->setAlgorithm(m_input->getAlgorithm());
  m_buffer->setModule(m_input->getModule());
  APE::BufferAccessor::setToken(*m_buffer,m_input->getToken());
  tbb::tick_count tstart=tbb::tick_count::now();  
  //std::cout<<"In work: Received Algorithm="<<m_input->getAlgorithm()
  // 	   <<" token="<<m_input->getToken()
  // 	   <<" module="<<m_input->getModule()
  // 	   <<" payloadSize="<<m_input->getPayloadSize()
  // 	   <<" TransferSize="<<m_input->getTransferSize()
  // 	   <<" userBuffer="<<m_input->getBuffer()
  // 	   <<std::endl;


  m_stats.reserve(10); 

  m_stats.push_back((tbb::tick_count::now()-tstart).seconds()*1000.);
}

APE::muonSimpleWorkGPU::~muonSimpleWorkGPU(){
  if(m_context) delete m_context;
}

std::shared_ptr<APE::BufferContainer>  APE::muonSimpleWorkGPU::getOutput(){
  return m_buffer;
}

void APE::muonSimpleWorkGPU::run(){
  tbb::tick_count tstart=tbb::tick_count::now();

  std::cout<<"running the job..." << std::endl;

  const APE::HoughTransformDeviceContext& devC = m_context->m_devC;
  int id = m_context->m_devC.m_deviceId; 
  std::cout << "Device Id: " << id << std::endl;

  checkCudaErrors(cudaSetDevice(id));

//  memcpy(devC.h_HTConfig, &m_context->m_HTConfig, sizeof(HT_ALGO_CONFIGURATION));
  memcpy(devC.h_MdtData, &m_context->m_MdtData, sizeof(INPUT_MDT_DATA));

  std::cout<<"copying data to GPU ..."<<std::endl; 
//  checkCudaErrors(cudaMemcpyAsync(devC.d_HTConfig, devC.h_HTConfig, sizeof(HT_ALGO_CONFIGURATION), cudaMemcpyHostToDevice, 0));
  checkCudaErrors(cudaMemcpyAsync(devC.d_MdtData, devC.h_MdtData, sizeof(INPUT_MDT_DATA), cudaMemcpyHostToDevice, 0));
  checkCudaErrors( cudaDeviceSynchronize() );

  std::cout << "Checking settings: maximum_residu_mm = " << ((HT_ALGO_CONFIGURATION*)devC.h_HTConfig)->m_maximum_residu_mm << std::endl;

  tbb::tick_count::interval_t duration=tbb::tick_count::now()-tstart;
  m_stats.push_back(duration.seconds()*1000.0);
  tstart=tbb::tick_count::now();

  // Add output preparation here
  //  prepareOutput(); 

  //  INPUT_MDT_DATA *pMdtArray=static_cast<INPUT_MDT_DATA*>(m_input->getBuffer());
  int nWords=m_context->m_MdtData.m_nDataWords;//pMdtArray->m_nDataWords;
  std::cout<<"In Work: run nWords = "<<nWords<<std::endl;

  //Begin the real algorithm
  //
  makeHTPatterns();

  SIMPLE_OUTPUT_DATA *pOutput = static_cast<SIMPLE_OUTPUT_DATA*>(m_buffer->getBuffer());
  pOutput->m_nDataWords = nWords;
  std::cout<<" processed "<<nWords<<" data words"<<std::endl;

  duration=tbb::tick_count::now()-tstart;
  m_stats.push_back(duration.seconds()*1000.0);
}

const std::vector<double>& APE::muonSimpleWorkGPU::getStats(){return m_stats;}

void APE::muonSimpleWorkGPU::makeHTPatterns() {

  size_t size=m_context->m_MdtData.m_nDataWords;
  std::cout << "makeHTPatterns() - size(words): " << size << std::endl;

  std::shared_ptr<APE::BufferContainer> resBuff=std::make_shared<APE::BufferContainer>(sizeof(APE::APEHeaders) + (sizeof(int) * (Nsec*Ntheta*Nphi*Nrho)));
  int* resMat=(int*)resBuff->getBuffer();

  std::shared_ptr<APE::BufferContainer> numBuff=std::make_shared<APE::BufferContainer>(sizeof(APE::APEHeaders) + (sizeof(int)));
  int* num=(int*)numBuff->getBuffer();

  cudaDeviceProp deviceProp;
  checkCudaErrors(cudaGetDeviceProperties(&deviceProp, 0));
  unsigned int maxThreadsPerBlock = deviceProp.maxThreadsPerBlock;
  std::cout << "thX=" << deviceProp.maxThreadsDim[0] << " thY=" << deviceProp.maxThreadsDim[1] 
  << " thZ=" << deviceProp.maxThreadsDim[2] << " grX=" << deviceProp.maxGridSize[0] 
  << " grY=" << deviceProp.maxGridSize[1] << " grZ=" << deviceProp.maxGridSize[2] 
  << " mThPerBlock=" << maxThreadsPerBlock << std::endl;

  std::cout << "Warp " << deviceProp.warpSize << " GM=" << deviceProp.totalGlobalMem << " SM=" << deviceProp.sharedMemPerBlock << std::endl;

  int *accMat;
  checkCudaErrors(cudaMalloc((void **)&accMat, sizeof(int) * Nsec*Ntheta*Nphi*Nrho)); //move this to the context
  checkCudaErrors(cudaMemset(accMat, 0, sizeof(int) * Nsec*Ntheta*Nphi*Nrho));

  //setup execution parameters
  dim3 threads1(Nrho);
  dim3 grid1(size);

  //create and start CUDA timer
  StopWatchInterface *timerCUDA = 0;
  sdkCreateTimer(&timerCUDA);
  sdkResetTimer(&timerCUDA);

  // finalize CUDA timer
  sdkStartTimer(&timerCUDA);
  voteHoughSpace<<< grid1, threads1 >>>((INPUT_MDT_DATA*) m_context->m_devC.d_MdtData, accMat);
  checkCudaErrors(cudaDeviceSynchronize());
  sdkStopTimer(&timerCUDA);
  float TimerCUDASpan = sdkGetAverageTimerValue(&timerCUDA);
  getLastCudaError("Kernel execution failed");

  std::cout << "1st kernel exe: " << TimerCUDASpan << std::endl;

  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaMemcpy((void*)resMat, accMat, sizeof(int) * Nsec*Ntheta*Nphi*Nrho, cudaMemcpyDeviceToHost));

  int cs = 0;
  for(int j=0; j<(Nsec*Ntheta*Nphi*Nrho);j++) cs+= resMat[j];
  std::cout << "Kernel 1 - matrix sum: " << cs << std::endl;

  int *tPar;
  checkCudaErrors(cudaMalloc((void**)&tPar, sizeof(int) * (Nsec*Ntheta*Nphi*Nrho)));
  checkCudaErrors(cudaMemset(tPar, -1, sizeof(int) * (Nsec*Ntheta*Nphi*Nrho)));

  int *nMRel;
  checkCudaErrors(cudaMalloc((void **)&nMRel, sizeof(int)));
  checkCudaErrors(cudaMemset(nMRel, 0, sizeof(int)));

  dim3 threads2(Nrho, maxThreadsPerBlock/Nrho);
  dim3 grid2(Nsec, Ntheta, (Nrho*Nphi)/maxThreadsPerBlock);

  sdkResetTimer(&timerCUDA);
  sdkStartTimer(&timerCUDA);
  findRelativeMax<<< grid2, threads2 >>>(accMat, tPar, nMRel);

  checkCudaErrors(cudaDeviceSynchronize());
  sdkStopTimer(&timerCUDA);
  std::cout << "2nd kernel exe: " << sdkGetAverageTimerValue(&timerCUDA) << std::endl;
  TimerCUDASpan += sdkGetAverageTimerValue(&timerCUDA);
  getLastCudaError("Kernel execution failed");

  checkCudaErrors(cudaMemcpy((void*)num, nMRel, sizeof(int), cudaMemcpyDeviceToHost));
  std::cout << "Kernel 2 - max rel found: " << *num << std::endl;

    checkCudaErrors(cudaFree(accMat));
    checkCudaErrors(cudaFree(tPar));
    checkCudaErrors(cudaFree(nMRel));

    std::cout<<"Total kernels time "<<TimerCUDASpan<<std::endl;

    //return time taken by the operation
     m_stats.push_back(TimerCUDASpan);
}

