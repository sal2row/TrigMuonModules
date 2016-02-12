#ifndef __APE_MUON_TRIGGER_MODULE_CPU_HPP
#define __APE_MUON_TRIGGER_MODULE_CPU_HPP

#include "../../../DataStructures/TrigMuonDataStructs/TrigMuonDataStructs/TrigMuonAccelerationEDM.h"
#include "../../../DataStructures/TrigMuonDataStructs/TrigMuonDataStructs/TrigMuonAccelerationCodes.h"
#include "APE/Module.hpp"
#include "APE/BufferContainer.hpp"

namespace APE {
  class muonTriggerModuleCPU:public APE::Module{
  public:
    muonTriggerModuleCPU();
    ~muonTriggerModuleCPU();
    bool addWork(std::shared_ptr<APE::BufferContainer> data);
    Work* getResult();
    const std::vector<int> getProvidedAlgs();
    virtual void printStats(std::ostream &out);
    virtual int getModuleId();
  private:
    HT_ALGO_CONFIGURATION* m_HTAlgoConfig;
    INPUT_MDT_DATA *m_MdtData;

  };
}

#endif
