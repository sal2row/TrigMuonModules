#include "../include/muonTriggerModule.hpp"
#include "../include/muonSimpleWork.hpp"

extern "C" APE::Module* getModule(){
  return new APE::muonTriggerModule();
}


extern "C" void deleteModule(APE::Module* c){
  APE::muonTriggerModule* dc=reinterpret_cast<APE::muonTriggerModule*>(c);
  delete dc;
}

APE::muonTriggerModule::muonTriggerModule(){

  m_toDoList=new tbb::concurrent_queue<APE::Work*>();
  m_completedList=new tbb::concurrent_queue<APE::Work*>();

}

APE::muonTriggerModule::~muonTriggerModule(){
  delete m_toDoList;
  delete m_completedList;

}

void APE::muonTriggerModule::addWork(std::shared_ptr<APE::BufferContainer> data){
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
  if(workType==77007){
    std::cout<<"Algorithm to execute: MS data processing (one day...)"<<std::endl;
    muonSimpleWork* w = new muonSimpleWork(data);
    w->run();
    m_completedList->push(w);
  } else {
    std::cout << "addWork - unknown worktype:" << workType << std::endl;
  }
}

APE::Work * APE::muonTriggerModule::getResult(){
  APE::Work* w=0;
  m_completedList->try_pop(w);
  if(w){
    return w;
  }
  return 0;
}

void APE::muonTriggerModule::printStats(std::ostream &out){

}

const std::vector<int> APE::muonTriggerModule::getProvidedAlgs(){
  std::vector<int> v{77007};
  return v;
}
