#include "../include/muonSimpleWorkGPU.hpp"
#include "../../../DataStructures/TrigMuonDataStructs/TrigMuonDataStructs/TrigMuonAccelerationEDM.h"
#include "../include/muonDeviceWrapperGPU.cuh"
#include "tbb/tick_count.h"
#include <cstring>
#include <cmath>
#include <iostream>
#include <unistd.h>
#include "APE/BufferAccessor.hpp"

APE::muonSimpleWorkGPU::muonSimpleWorkGPU(const HoughTransformContextGPU& ctx, std::shared_ptr<APE::BufferContainer> data, tbb::concurrent_queue<APE::Work*>& compQ, tbb::concurrent_queue<HoughTransformDeviceContext*>& ctxQ) :
  m_context(0),
  m_input(data),
  m_outputQueue(compQ),
  m_contextQueue(ctxQ)
{

  m_context = new HoughTransformContextGPU(ctx);
  m_buffer = std::make_shared<APE::BufferContainer>(sizeof(MUON_HOUGH_RED_PATTERN));
  m_buffer->setAlgorithm(m_input->getAlgorithm());
  m_buffer->setModule(m_input->getModule());
  APE::BufferAccessor::setToken(*m_buffer,m_input->getToken());
  tbb::tick_count tstart=tbb::tick_count::now();  

  m_stats.reserve(10); 

  m_stats.push_back((tbb::tick_count::now()-tstart).seconds()*1000.);
}

APE::muonSimpleWorkGPU::~muonSimpleWorkGPU(){
  if(m_context) delete m_context;
}

std::shared_ptr<APE::BufferContainer>  APE::muonSimpleWorkGPU::getOutput(){
  return m_buffer;
}

bool APE::muonSimpleWorkGPU::run(){

  tbb::tick_count tstart=tbb::tick_count::now();

  std::cout<<"Inside worker: running HT job..." << std::endl;

  const APE::HoughTransformDeviceContext& devC = m_context->m_devC;

  memcpy(devC.h_HTData, &m_context->m_HTData, sizeof(INPUT_HT_DATA));
  memcpy(devC.h_HTConfig, &m_context->m_HTConfig, sizeof(HT_ALGO_CONFIGURATION));

  TrigMuonModuleKernels::wrpHoughCtx(devC);

  tbb::tick_count::interval_t duration=tbb::tick_count::now()-tstart;
  m_stats.push_back(duration.seconds()*1000.0);
  tstart=tbb::tick_count::now();

  MUON_HOUGH_RED_PATTERN *pOutput = static_cast<MUON_HOUGH_RED_PATTERN*>(m_buffer->getBuffer());
  pOutput->m_nPatterns = 0;
  pOutput->m_nTotHits = 0;

  std::vector<float> timeVec = TrigMuonModuleKernels::wrpHoughAlgo(devC, pOutput);

  for(std::vector<float>::const_iterator t = timeVec.begin(); t != timeVec.end(); t++)
    m_stats.push_back(*t);  

  duration=tbb::tick_count::now()-tstart;
  m_stats.push_back(duration.seconds()*1000.0);
  m_outputQueue.push(this);
  m_contextQueue.push(&(m_context->m_devC));

  return true;
}

const std::vector<double>& APE::muonSimpleWorkGPU::getStats(){return m_stats;}
