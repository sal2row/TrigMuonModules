#include "../include/muonTriggerModuleGPU.hpp"
#include "../include/muonSimpleWorkGPU.hpp"

#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>       // helper functions for CUDA timing and initialization
#include <helper_functions.h>  // helper functions for timing, string parsing

#include <fstream>
extern "C" APE::Module* getModule(){
  return new APE::muonTriggerModuleGPU();
}

/*extern "C" int getFactoryId() {
  return TrigMuonModuleID_CUDA;
}*/

extern "C" void deleteModule(APE::Module* c){
  APE::muonTriggerModuleGPU* dc=reinterpret_cast<APE::muonTriggerModuleGPU*>(c);
  delete dc;
}

bool APE::muonTriggerModuleGPU::configure(std::shared_ptr<APEConfig::ConfigTree>& config) {

  if(config == 0){ return false; }

  //reading and validating configuration
  std::vector<int> allowedGPUs, nDeviceContexts;
  
  for(auto p : config->getSubtree("AllowedGPUs")->findParameters("param"))
    allowedGPUs.push_back(p->getValue<int>());
  for(auto p : config->getSubtree("NumTokens")->findParameters("param"))
    nDeviceContexts.push_back(p->getValue<int>());
  
  if(allowedGPUs.empty() || nDeviceContexts.empty()) return false;
  if(allowedGPUs.size() != nDeviceContexts.size()) return false;
  
  devManager = new TrigMuonModuleKernels::DeviceManager();
  m_maxDevice = devManager->getNDevice();

  unsigned int dcIndex=0;
  
  for(std::vector<int>::iterator devIt = allowedGPUs.begin(); devIt!= allowedGPUs.end();++devIt, dcIndex++) {
    
    int deviceId = (*devIt);
   
    if(deviceId<0 || deviceId>=m_maxDevice) continue;
    
    int nDC = nDeviceContexts[dcIndex];
    
    for(int dc=0;dc<nDC;dc++) {
      std::cout<<"creating Hough Transform context for device "<<deviceId<<std::endl;
      HoughTransformDeviceContext* p = devManager->createHTContext(deviceId);
      m_htDcQueue.push(p);
    }
  }

  return true;
  
}

APE::muonTriggerModuleGPU::muonTriggerModuleGPU() : m_maxNumberOfContexts(0), m_ctxPerDev(3), m_maxDevice(0)

{

  m_toDoList=new tbb::concurrent_queue<APE::Work*>();
  m_completedList=new tbb::concurrent_queue<APE::Work*>();

}

APE::muonTriggerModuleGPU::~muonTriggerModuleGPU(){

  clearMap<HT_ALGO_CONFIGURATION>(m_htCfgStorageMap);
  clearMap<INPUT_HT_DATA>(m_htDataStorageMap);
  //clearMap<INPUT_MDT_DATA>(m_htDataStorageMap);

  delete m_toDoList;
  delete m_completedList;
  
  HoughTransformDeviceContext* dc = 0;
  while(m_htDcQueue.try_pop(dc)) devManager->deleteHTContext(dc);
  delete devManager;

}

bool APE::muonTriggerModuleGPU::addWork(std::shared_ptr<APE::BufferContainer> data){
  uint32_t workType=(data->getAlgorithm());

  bool aw = true;  
  if(workType==MuonJobControlCode::CFG_EXPORT){
    std::cout<<"Now executing: HT configuration setting... "<<std::endl;

    pid_t dummyPId = 0;
    //pid_t dummyPId = data->getSrcID();
    HT_ALGO_CONFIGURATION*& p = checkMap<HT_ALGO_CONFIGURATION> (m_htCfgStorageMap, dummyPId);
    auto pConfig=static_cast<HT_ALGO_CONFIGURATION*>(data->getBuffer());
    memcpy(p, pConfig, sizeof(HT_ALGO_CONFIGURATION));

    /*std::ofstream outfile ("cfg.txt",std::ofstream::binary);
    outfile.write((char*)data->getBuffer(),sizeof(HT_ALGO_CONFIGURATION));
    outfile.close();*/

    std::cout << "Copy successful, returning" << std::endl;
    aw = false;
  }
  else if(workType==MuonJobControlCode::MS_HOUGH){
    std::cout<<"Now executing: MS data processing... "<<std::endl;

    pid_t pId = data->getSrcID();
    pid_t dummyPId = 0;

    INPUT_HT_DATA*& p = checkMap<INPUT_HT_DATA>(m_htDataStorageMap, pId);
    auto pArray = static_cast<INPUT_HT_DATA*>(data->getBuffer());
    memcpy(p, pArray, sizeof(INPUT_HT_DATA));

    const HT_ALGO_CONFIGURATION* p_htCfg = retrieveData<HT_ALGO_CONFIGURATION>(m_htCfgStorageMap, dummyPId);
    const INPUT_HT_DATA* p_htData = retrieveData<INPUT_HT_DATA>(m_htDataStorageMap, pId);

    if(p_htCfg == NULL) return false;
    if(p_htData == NULL) return false;

    /*if(p_htData->m_nDataWords == 87){    
      std::ofstream outfile ("data.txt",std::ofstream::binary);
      outfile.write((char*)data->getBuffer(),sizeof(INPUT_HT_DATA));
      outfile.close();
    }*/

    HoughTransformDeviceContext * ctx = NULL;
    while(!m_htDcQueue.try_pop(ctx)) {
      std::cout<<"Waiting for free device context..."<<std::endl;
    };    
    
    std::cout<<"Allocating the Hough job to device "<<ctx->m_deviceId<<std::endl;

    auto w = new muonSimpleWorkGPU(HoughTransformContextGPU(*p_htData, *p_htCfg, *ctx),
				   data, *m_completedList, m_htDcQueue);

    std::cout<<"Hough transform job created"<<std::endl;
    m_toDoList->push(w);
  }
  else
    std::cout << "addWork - unknown worktype:" << workType << std::endl;

  return aw;

}

APE::Work * APE::muonTriggerModuleGPU::getResult(){
  APE::Work* w=0;
  m_completedList->try_pop(w);
  if(w){
    return w;
  }
  return 0;
}

int APE::muonTriggerModuleGPU::getModuleId(){return TrigMuonModuleID_CUDA;}

void APE::muonTriggerModuleGPU::printStats(std::ostream &out){

}

const std::vector<int> APE::muonTriggerModuleGPU::getProvidedAlgs(){
  std::vector<int> v{MuonJobControlCode::CFG_EXPORT,MuonJobControlCode::MS_HOUGH};
  return v;
}
