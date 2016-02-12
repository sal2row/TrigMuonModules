#ifndef __APE_MUON_TRIGGER_MODULE_GPU_CUH
#define __APE_MUON_TRIGGER_MODULE_GPU_CUH

#include "../../../DataStructures/TrigMuonDataStructs/TrigMuonDataStructs/TrigMuonAccelerationEDM.h"
#include "../../../DataStructures/TrigMuonDataStructs/TrigMuonDataStructs/TrigMuonAccelerationCodes.h"
#include "../include/muonDataContexts.hpp"
#include "APE/Module.hpp"
#include "APE/BufferContainer.hpp"

namespace APE {
  class muonTriggerModuleGPU:public APE::Module{
  public:
    muonTriggerModuleGPU();
    ~muonTriggerModuleGPU();
    void addWork(std::shared_ptr<APE::BufferContainer> data);
    Work* getResult();
    const std::vector<int> getProvidedAlgs();
    virtual void printStats(std::ostream &out);
    virtual int getModuleId();
  private:
    HT_ALGO_CONFIGURATION* m_HTAlgoConfig;
    HT_CONFIG* m_HTAlgoConfig_new;
    INPUT_MDT_DATA *m_MdtData;

    int m_maxDevice;
    bool m_usePinnedMemory;
    bool m_writeCombinedMemory;
    bool m_linkOutputToShm;

    std::vector<HoughTransformDeviceContext*> m_devC;

    // context allocation/deallocation
    HoughTransformDeviceContext* createHoughTransformContext(int);
    void deleteHoughTransformContext(HoughTransformDeviceContext*);

    //GPU configurations
    int m_dimBlock;
    int m_dimGrid;

  };
}

#endif
