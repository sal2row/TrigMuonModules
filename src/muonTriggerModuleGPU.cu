#include "../include/muonTriggerModuleGPU.cuh"
#include "../include/muonSimpleWorkGPU.cuh"
#include "../include/HTConfigWorkGPU.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>       // helper functions for CUDA timing and initialization
#include <helper_functions.h>  // helper functions for timing, string parsing

extern "C" APE::Module* getModule(){
  return new APE::muonTriggerModuleGPU();
}

extern "C" void deleteModule(APE::Module* c){
  APE::muonTriggerModuleGPU* dc=reinterpret_cast<APE::muonTriggerModuleGPU*>(c);
  delete dc;
}

APE::muonTriggerModuleGPU::muonTriggerModuleGPU() : m_HTAlgoConfig(0), m_MdtData(0), m_HTAlgoConfig_new(0) {

  m_toDoList=new tbb::concurrent_queue<APE::Work*>();
  m_completedList=new tbb::concurrent_queue<APE::Work*>();

  //GPU initialization
  m_dimBlock = 400;
  m_dimGrid = 200000/m_dimBlock;

  checkCudaErrors(cudaGetDeviceCount(&m_maxDevice));

  for(int deviceId = 0; deviceId<m_maxDevice; deviceId++) {
  	  std::cout<<"creating hough transform context for device "<<deviceId<<std::endl;
	  HoughTransformDeviceContext* p = createHoughTransformContext(deviceId);
	  m_devC.push_back(p);
  }

}

APE::muonTriggerModuleGPU::~muonTriggerModuleGPU(){
  	delete m_HTAlgoConfig;
  	delete m_HTAlgoConfig_new;
	delete m_MdtData;
  	delete m_toDoList;
  	delete m_completedList;

  for(std::vector<HoughTransformDeviceContext*>::iterator it = m_devC.begin();it!=m_devC.end();++it) {
     deleteHoughTransformContext(*it);  
  }
  m_devC.clear();
}

APE::HoughTransformDeviceContext* APE::muonTriggerModuleGPU::createHoughTransformContext(int id) {
  checkCudaErrors(cudaSetDevice(id));
  APE::HoughTransformDeviceContext* p = new APE::HoughTransformDeviceContext;
  p->m_deviceId = id;

  //Allocate memory
  checkCudaErrors(cudaMalloc((void **)&p->d_HTConfig, sizeof(HT_ALGO_CONFIGURATION)));
  checkCudaErrors(cudaMalloc((void **)&p->d_HTConfig_new, sizeof(Config)));
  checkCudaErrors(cudaMalloc((void **)&p->d_MdtData, sizeof(INPUT_MDT_DATA)));
		
  checkCudaErrors(cudaMallocHost((void **)&p->h_HTConfig, sizeof(HT_ALGO_CONFIGURATION)));
  checkCudaErrors(cudaMallocHost((void **)&p->h_HTConfig_new, sizeof(Config)));
  checkCudaErrors(cudaMallocHost((void **)&p->h_MdtData, sizeof(INPUT_MDT_DATA)));

  return p;
}

void APE::muonTriggerModuleGPU::deleteHoughTransformContext(HoughTransformDeviceContext* p) {
  int id = p->m_deviceId;
  checkCudaErrors(cudaSetDevice(id));
	 
  checkCudaErrors(cudaFree(p->d_HTConfig));
  checkCudaErrors(cudaFree(p->d_HTConfig_new));
  checkCudaErrors(cudaFree(p->d_MdtData));
	   
  checkCudaErrors(cudaFreeHost(p->h_HTConfig));
  checkCudaErrors(cudaFreeHost(p->h_HTConfig_new));
  checkCudaErrors(cudaFreeHost(p->h_MdtData));
  
  //cudaDeviceReset();
   
  delete p;
}

void APE::muonTriggerModuleGPU::addWork(std::shared_ptr<APE::BufferContainer> data){
  uint32_t workType=(data->getAlgorithm());
  //std::cout<<" In addWork : Received Algorithm="<<data->getAlgorithm()
  //	    <<" token="<<data->getToken()
  //	    <<" module="<<data->getModule()
  //	    <<" payloadSize="<<data->getPayloadSize()
  //	    <<" TransferSize="<<data->getTransferSize()
  //	    <<" userBuffer="<<data->getBuffer()
  //	    <<std::endl;
  
  if(workType==MuonJobControlCode::CFG_EXPORT){
    std::cout<<"Algorithm to execute: HT configuration setting... "<<std::endl;
    if(m_HTAlgoConfig) delete m_HTAlgoConfig;
    m_HTAlgoConfig = new HT_ALGO_CONFIGURATION;
    HT_ALGO_CONFIGURATION *pConfig=static_cast<HT_ALGO_CONFIGURATION*>(data->getBuffer());
    memcpy(m_HTAlgoConfig, pConfig, sizeof(HT_ALGO_CONFIGURATION));
    int token = data->getToken();
    int deviceIdx = token % m_maxDevice;
    std::cout<<"allocating the job to device "<<deviceIdx<<std::endl;
    HTConfigWorkGPU* w = new HTConfigWorkGPU(HoughTransformContextGPU(*m_MdtData, *m_HTAlgoConfig, *m_devC[deviceIdx]), data);
    w->run();
    m_completedList->push(w);
  } else if(workType==MuonJobControlCode::MS_HOUGH){
    std::cout<<"Algorithm to execute: MS data processing (one day...)"<<std::endl;
    if(m_MdtData) delete m_MdtData;
    m_MdtData = new INPUT_MDT_DATA;
    INPUT_MDT_DATA *pMdtData=static_cast<INPUT_MDT_DATA*>(data->getBuffer());
    memcpy(m_MdtData, pMdtData, sizeof(INPUT_MDT_DATA));
    int token = data->getToken();
    int deviceIdx = token % m_maxDevice;
    std::cout<<"allocating the job to device "<<deviceIdx<<std::endl;
    muonSimpleWorkGPU* w = new muonSimpleWorkGPU(HoughTransformContextGPU(*m_MdtData, *m_HTAlgoConfig, *m_devC[deviceIdx]), data);
    w->run();
    m_completedList->push(w);
  } else {
    std::cout << "addWork - unknown worktype:" << workType << std::endl;
  }
}

APE::Work * APE::muonTriggerModuleGPU::getResult(){
  APE::Work* w=0;
  m_completedList->try_pop(w);
  if(w){
    return w;
  }
  return 0;
}

int APE::muonTriggerModuleGPU::getModuleId(){return 3;}

void APE::muonTriggerModuleGPU::printStats(std::ostream &out){

}

const std::vector<int> APE::muonTriggerModuleGPU::getProvidedAlgs(){
  std::vector<int> v{MuonJobControlCode::CFG_EXPORT,MuonJobControlCode::MS_HOUGH};
  return v;
}
