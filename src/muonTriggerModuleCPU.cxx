#include "../include/muonTriggerModuleCPU.hpp"
#include "../include/muonSimpleWork.hpp"
#include "../include/HTConfigWork.hpp"

extern "C" APE::Module* getModule(){
  return new APE::muonTriggerModuleCPU();
}


extern "C" void deleteModule(APE::Module* c){
  APE::muonTriggerModuleCPU* dc=reinterpret_cast<APE::muonTriggerModuleCPU*>(c);
  delete dc;
}

APE::muonTriggerModuleCPU::muonTriggerModuleCPU() : m_HTAlgoConfig(0), m_MdtData(0) {

  m_toDoList=new tbb::concurrent_queue<APE::Work*>();
  m_completedList=new tbb::concurrent_queue<APE::Work*>();

}

APE::muonTriggerModuleCPU::~muonTriggerModuleCPU(){
  delete m_HTAlgoConfig;
  delete m_MdtData;
  delete m_toDoList;
  delete m_completedList;

}

bool APE::muonTriggerModuleCPU::addWork(std::shared_ptr<APE::BufferContainer> data){
  uint32_t workType=(data->getAlgorithm());
  //std::cout<<" In addWork : Received Algorithm="<<data->getAlgorithm()
  //	    <<" token="<<data->getToken()
  //	    <<" module="<<data->getModule()
  //	    <<" payloadSize="<<data->getPayloadSize()
  //	    <<" TransferSize="<<data->getTransferSize()
  //	    <<" userBuffer="<<data->getBuffer()
  //	    <<std::endl;
  
  /*  if(workType==11001){
    std::cout<<"Algorithm to execute: pixel BS decoding"<<std::endl;
    mySimpleWork* w = new mySimpleWork(data);
    w->run();
    m_completedList->push(w);
    }*/

  bool aw = false;
  if(workType==MuonJobControlCode::CFG_EXPORT){
    /*
    if(m_HTAlgoConfig) delete m_HTAlgoConfig;
    m_HTAlgoConfig = new HT_ALGO_CONFIGURATION;
    HT_ALGO_CONFIGURATION *pConfig=static_cast<HT_ALGO_CONFIGURATION*>(data->getBuffer());
    memcpy(m_HTAlgoConfig, pConfig, sizeof(HT_ALGO_CONFIGURATION));
    std::cout << "MB DEBUG: check " << m_HTAlgoConfig->m_number_of_ids << " - " << pConfig->m_number_of_ids << std::endl;
    std::cout << "MB DEBUG: check " << m_HTAlgoConfig->m_maximum_residu_mm << " - " << pConfig->m_maximum_residu_mm << std::endl;
    HTConfigWork* w = new HTConfigWork(HoughTransformContext(*m_MdtData, *m_HTAlgoConfig), data);
    w->run();
    m_completedList->push(w);
    */
    std::cout<<"Now executing: HT configuration setting... "<<std::endl;

    pid_t pId = data->getSrcID();
    HT_ALGO_CONFIGURATION*& p = checkMap<HT_ALGO_CONFIGURATION> (m_htCfgStorageMap, pId);
    auto pConfig=static_cast<HT_ALGO_CONFIGURATION*>(data->getBuffer());
    memcpy(p, pConfig, sizeof(HT_ALGO_CONFIGURATION));

    std::cout << "Copy successful, returning" << std::endl;


  } else if(workType==MuonJobControlCode::MS_HOUGH){
    std::cout<<"Algorithm to execute: MS data processing (one day...)"<<std::endl;
    if(m_MdtData) delete m_MdtData;
    m_MdtData = new INPUT_MDT_DATA;
    INPUT_MDT_DATA *pMdtData=static_cast<INPUT_MDT_DATA*>(data->getBuffer());
    memcpy(m_MdtData, pMdtData, sizeof(INPUT_MDT_DATA));
    // int token = data->getToken();
    // int deviceIdx = token % m_maxDevice;
    // std::cout<<"allocating the job to device "<<deviceIdx<<std::endl;

    //    muonSimpleWorkGPU* w = new muonSimpleWorkGPU(HoughTransformContextGPU(*m_MdtData, *m_HTAlgoConfig, *m_devC[deviceIdx]), data);//, m_dMdtData, m_dHTAlgoConfig); //Context pointer?!?
    muonSimpleWork* w = new muonSimpleWork(HoughTransformContext(*m_MdtData, *m_HTAlgoConfig), data);
    w->run();
    m_completedList->push(w);
  } else
    std::cout << "addWork - unknown worktype:" << workType << std::endl;

  return aw;
}

APE::Work * APE::muonTriggerModuleCPU::getResult(){
  APE::Work* w=0;
  m_completedList->try_pop(w);
  if(w){
    return w;
  }
  return 0;
}

int APE::muonTriggerModuleCPU::getModuleId(){return 3;}

void APE::muonTriggerModuleCPU::printStats(std::ostream &out){

}

const std::vector<int> APE::muonTriggerModuleCPU::getProvidedAlgs(){
  std::vector<int> v{77006,77007};
  return v;
}
