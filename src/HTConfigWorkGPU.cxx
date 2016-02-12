#include "../include/HTConfigWorkGPU.hpp"
#include "../../../DataStructures/TrigMuonDataStructs/TrigMuonDataStructs/TrigMuonAccelerationEDM.h"
#include "../include/muonDeviceWrapper.cuh"
#include "tbb/tick_count.h"
#include <cstring>
#include <cmath>
#include <iostream>
#include <unistd.h>
#include "APE/BufferAccessor.hpp"

APE::HTConfigWorkGPU::HTConfigWorkGPU(const HoughTransformContextGPU& ctx, std::shared_ptr<APE::BufferContainer> data, tbb::concurrent_queue<APE::Work*>& compQ, tbb::concurrent_queue<HoughTransformDeviceContext*>& ctxQ) :
  m_context(0),
  m_input(data),
  m_outputQueue(compQ),
  m_contextQueue(ctxQ)
{

  m_context = new HoughTransformContextGPU(ctx);
  //m_buffer = std::make_shared<APE::BufferContainer>(sizeof(SIMPLE_OUTPUT_DATA));
  m_buffer->setAlgorithm(m_input->getAlgorithm());
  m_buffer->setModule(m_input->getModule());
  APE::BufferAccessor::setToken(*m_buffer,m_input->getToken());
  tbb::tick_count tstart=tbb::tick_count::now();  

  m_stats.reserve(10); 

  m_stats.push_back((tbb::tick_count::now()-tstart).seconds()*1000.);
}

APE::HTConfigWorkGPU::~HTConfigWorkGPU(){
  if(m_context) delete m_context;
}

std::shared_ptr<APE::BufferContainer>  APE::HTConfigWorkGPU::getOutput(){
  return m_buffer;
}

bool APE::HTConfigWorkGPU::run(){

  tbb::tick_count tstart=tbb::tick_count::now();

  std::cout<<"Running HTConfigWork ..." << std::endl;

  const APE::HoughTransformDeviceContext& devC = m_context->m_devC;

  memcpy((void*)devC.h_HTConfig,(void*) &m_context->m_HTConfig, sizeof(HT_ALGO_CONFIGURATION));

  std::cout << "copying configuration to GPU ..." << std::endl;
  //checkCudaErrors(cudaMemcpyAsync(devC.d_HTConfig, devC.h_HTConfig, sizeof(HT_ALGO_CONFIGURATION), cudaMemcpyHostToDevice, 0));
  //checkCudaErrors( cudaDeviceSynchronize() );

  TrigMuonModuleKernels::wrpHoughCtx(devC);

  tbb::tick_count::interval_t duration=tbb::tick_count::now()-tstart;
  m_stats.push_back(duration.seconds()*1000.0);
  tstart=tbb::tick_count::now();


  duration=tbb::tick_count::now()-tstart;
  m_stats.push_back(duration.seconds()*1000.0);
  m_outputQueue.push(this);
  m_contextQueue.push(&(m_context->m_devC));

  return true;

}

const std::vector<double>& APE::HTConfigWorkGPU::getStats(){return m_stats;}
