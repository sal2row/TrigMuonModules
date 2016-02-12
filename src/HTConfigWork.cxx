#include "../include/HTConfigWork.hpp"
#include "../../../DataStructures/TrigMuonDataStructs/TrigMuonDataStructs/TrigMuonAccelerationEDM.h"
#include "tbb/tick_count.h"
#include <cstring>
#include <cmath>
#include <iostream>
#include <unistd.h>
#include "APE/BufferAccessor.hpp"
APE::HTConfigWork::HTConfigWork(const HoughTransformContext& ctx, std::shared_ptr<APE::BufferContainer> data) : 
  m_context(0),
  m_input(data) {

  m_context = new HoughTransformContext(ctx);
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

APE::HTConfigWork::~HTConfigWork(){
  if(m_context) delete m_context;
}

std::shared_ptr<APE::BufferContainer>  APE::HTConfigWork::getOutput(){
  return m_buffer;
}

bool APE::HTConfigWork::run(){
  tbb::tick_count tstart=tbb::tick_count::now();

  std::cout<<"running HTConfigWork ..." << std::endl;
  std::cout<<"dummy, already stored!"<<std::endl;
  tbb::tick_count::interval_t duration=tbb::tick_count::now()-tstart;
  m_stats.push_back(duration.seconds()*1000.0);
  tstart=tbb::tick_count::now();

  SIMPLE_OUTPUT_DATA *pOutput = static_cast<SIMPLE_OUTPUT_DATA*>(m_buffer->getBuffer());
  pOutput->m_nDataWords = 777;
  std::cout << "Check code: " << pOutput->m_nDataWords <<std::endl;

  duration=tbb::tick_count::now()-tstart;
  m_stats.push_back(duration.seconds()*1000.0);

  return true;

}

const std::vector<double>& APE::HTConfigWork::getStats(){return m_stats;}

void APE::HTConfigWork::saveConfiguration(HT_ALGO_CONFIGURATION* config){

  std::cout << "Storing algorithm configurations..." << std::endl;
  // int nWords=mdtHitArray->m_nDataWords;
  // for(int i=0;i<nWords;i++) {
  //   MuonHoughHit mdtHit = mdtHitArray->m_hit[i];
  //   if(false) std::cout << mdtHit.m_phi << " ";
  // }
}
