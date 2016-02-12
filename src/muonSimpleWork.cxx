#include "../include/muonSimpleWork.hpp"
#include "HoughTransform.cxx"
#include "../../../DataStructures/TrigMuonDataStructs/TrigMuonDataStructs/TrigMuonAccelerationEDM.h"
#include "tbb/tick_count.h"
#include <cstring>
#include <cmath>
#include <iostream>
#include <unistd.h>
#include "APE/BufferAccessor.hpp"
APE::muonSimpleWork::muonSimpleWork(const HoughTransformContext& ctx, std::shared_ptr<APE::BufferContainer> data) : 
  m_context(0),
  m_input(data) {

  m_context = new HoughTransformContext(ctx);
  m_buffer = std::make_shared<APE::BufferContainer>(sizeof(OUTPUT_DATA_CONTAINER));//output data
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
  /*
  if(outBuff){
    m_buffer=outBuff;
  }else{
    m_buffer=std::make_shared<APE::BufferContainer>(sizeof(SHM_RAW_PIX_OUTPUT_DATA));
    m_buffer->setHeaderInfoFrom(*data);
  }
  */
  /*  
  SHM_RAW_INPUT_DATA *pShmArray=static_cast<SHM_RAW_INPUT_DATA*>(data->getBuffer());
  for(int i = 0; i<pShmArray->m_nROBs; i++) {
    int val = pShmArray->m_rodInfo[i].m_rodId;
    for(int j=pShmArray->m_rodInfo[i].m_begin;j<pShmArray->m_rodInfo[i+1].m_begin;j++) {
      rodIDArray->m_rodIds[j]=val;
    }
  }
  m_nWords=pShmArray->m_nDataWords;
  */
  m_stats.push_back((tbb::tick_count::now()-tstart).seconds()*1000.);
}

APE::muonSimpleWork::~muonSimpleWork(){
  if(m_context) delete m_context;
}

std::shared_ptr<APE::BufferContainer>  APE::muonSimpleWork::getOutput(){
  return m_buffer;
}

bool APE::muonSimpleWork::run(){
  tbb::tick_count tstart=tbb::tick_count::now();

  std::cout<<"running the job..." << std::endl;

  tbb::tick_count::interval_t duration=tbb::tick_count::now()-tstart;
  m_stats.push_back(duration.seconds()*1000.0);
  tstart=tbb::tick_count::now();

  // usleep(20); //MB-DEVEL
  
  //  prepareOutput();

  INPUT_MDT_DATA *pMdtArray=static_cast<INPUT_MDT_DATA*>(m_input->getBuffer());
  int nWords=pMdtArray->m_nDataWords;
  std::cout<<"In Work: run nWords = "<<nWords<<std::endl;

  OUTPUT_DATA_CONTAINER *pOutput = static_cast<OUTPUT_DATA_CONTAINER*>(m_buffer->getBuffer());
  pOutput->m_number_of_ids = m_context->m_HTConfig.m_number_of_ids;
  pOutput->m_maximum_level = m_context->m_HTConfig.m_maximum_level;
  pOutput->m_number_of_maxima = m_context->m_HTConfig.m_number_of_maxima;

  // makeHTPatterns(pMdtArray);
  HoughTransform::makeHTPatterns(m_context, pMdtArray, pOutput);

  for(int j=0; j< pOutput->m_N_maxima_found; j++) {
    std::cout << "Test in muonSimpleWork.cxx: id_number, etheta: " << pOutput->patterns[j].m_id_number << " " << pOutput->patterns[j].m_etheta << std::endl;
  }
  // std::cout << std::endl;

  //pOutput->m_nDataWords = nWords;
  //std::cout<<" processed "<<nWords<<" data words"<<std::endl;

  duration=tbb::tick_count::now()-tstart;
  m_stats.push_back(duration.seconds()*1000.0);

  return true;
}

const std::vector<double>& APE::muonSimpleWork::getStats(){return m_stats;}

/*
void APE::muonSimpleWork::makeHTPatterns(INPUT_MDT_DATA* mdtHitArray){

  int nWords=mdtHitArray->m_nDataWords;
  for(int i=0;i<nWords;i++) {
    MuonHoughHitFlat mdtHit = mdtHitArray->m_hit[i];
    if(false) std::cout << mdtHit.m_phi << " ";
  }
  std::cout << "Checking settings: maximum_residu_mm = " << m_context->m_HTConfig.m_maximum_residu_mm << std::endl;
}
*/
