#ifndef __APE_MUON_TRIGGER_MODULE_GPU_HPP
#define __APE_MUON_TRIGGER_MODULE_GPU_HPP

#include <map>
#include "../../../DataStructures/TrigMuonDataStructs/TrigMuonDataStructs/TrigMuonAccelerationEDM.h"
#include "../../../DataStructures/TrigMuonDataStructs/TrigMuonDataStructs/TrigMuonAccelerationCodes.h"
#include "../include/muonDataContexts.hpp"
#include "APE/Module.hpp"
#include "APE/BufferContainer.hpp"
#include "../include/muonDeviceWrapperGPU.cuh"

namespace APE {
  class muonTriggerModuleGPU:public APE::Module{
  public:
    muonTriggerModuleGPU();
    ~muonTriggerModuleGPU();
    bool addWork(std::shared_ptr<APE::BufferContainer> data);
    Work* getResult();
    const std::vector<int> getProvidedAlgs();
    virtual void printStats(std::ostream &out);
    virtual int getModuleId();

    TrigMuonModuleKernels::DeviceManager* devManager;

    virtual bool configure(std::shared_ptr<APEConfig::ConfigTree>&);

  protected:

    template<typename T, class K = unsigned int, class Cont=std::map<K, T*> > void clearMap(Cont& m) {
      for(auto it = m.begin(); it!=m.end(); ++it) delete (*it).second;
      m.clear();
    }

    template<typename T, class K = unsigned int, class Cont=std::map<K, T*> > T*& checkMap(Cont& m, K key) {
      auto it = m.find(key);
      if(it==m.end()) {
        T* t = new T;
        m.insert(std::pair<K, T*>(key,t));
      }
      it = m.find(key);
      return (*it).second;
    }

    template<typename T, class K = unsigned int, class Cont=std::map<K, T*> > const T* retrieveData(Cont& m, K key) const {
      auto it = m.find(key);
      if(it==m.end()) return NULL;
      return (*it).second;
    }

  private:

    std::map<unsigned int, HT_ALGO_CONFIGURATION*> m_htCfgStorageMap;
    std::map<unsigned int, INPUT_HT_DATA*> m_htDataStorageMap;
    //std::map<unsigned int, INPUT_MDT_DATA*> m_htDataStorageMap;

    int m_maxNumberOfContexts;
    int m_ctxPerDev;
    int m_maxDevice;
    bool m_usePinnedMemory;
    bool m_writeCombinedMemory;
    bool m_linkOutputToShm;

    tbb::concurrent_queue<HoughTransformDeviceContext*> m_htDcQueue;

    // context allocation/deallocation
    //HoughTransformDeviceContext* createHoughTransformContext(int);
    //void deleteHoughTransformContext(HoughTransformDeviceContext*);

    //GPU configurations
    int m_dimBlock;
    int m_dimGrid;

  };
}

#endif

