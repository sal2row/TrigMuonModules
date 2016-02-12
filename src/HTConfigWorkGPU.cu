#include "../include/HTConfigWorkGPU.cuh"
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

APE::HTConfigWorkGPU::HTConfigWorkGPU(const HoughTransformContextGPU& ctx, std::shared_ptr<APE::BufferContainer> data) :
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

APE::HTConfigWorkGPU::~HTConfigWorkGPU(){
  if(m_context) delete m_context;
}

std::shared_ptr<APE::BufferContainer>  APE::HTConfigWorkGPU::getOutput(){
  return m_buffer;
}

void APE::HTConfigWorkGPU::run(){
  tbb::tick_count tstart=tbb::tick_count::now();

  std::cout<<"running HTConfigWork ..." << std::endl;

  const APE::HoughTransformDeviceContext& devC = m_context->m_devC;
  int id = m_context->m_devC.m_deviceId; 
  std::cout << "Device Id: " << id << std::endl;

  checkCudaErrors(cudaSetDevice(id));

  memcpy(devC.h_HTConfig, &m_context->m_HTConfig, sizeof(HT_ALGO_CONFIGURATION));

  std::cout<<"copying configuration to GPU ..."<<std::endl; 
  checkCudaErrors(cudaMemcpyAsync(devC.d_HTConfig, devC.h_HTConfig, sizeof(HT_ALGO_CONFIGURATION), cudaMemcpyHostToDevice, 0));
  checkCudaErrors( cudaDeviceSynchronize() );

  tbb::tick_count::interval_t duration=tbb::tick_count::now()-tstart;
  m_stats.push_back(duration.seconds()*1000.0);
  tstart=tbb::tick_count::now();

  SIMPLE_OUTPUT_DATA *pOutput = static_cast<SIMPLE_OUTPUT_DATA*>(m_buffer->getBuffer());
  pOutput->m_nDataWords = 777;
  std::cout << "Check code: " << pOutput->m_nDataWords <<std::endl;

  duration=tbb::tick_count::now()-tstart;
  m_stats.push_back(duration.seconds()*1000.0);
}

const std::vector<double>& APE::HTConfigWorkGPU::getStats(){return m_stats;}

/*
void APE::HTConfigWorkGPU::saveConfiguration(HT_ALGO_CONFIGURATION* config){

  std::cout << "Storing algorithm configurations..." << std::endl;
}
*/