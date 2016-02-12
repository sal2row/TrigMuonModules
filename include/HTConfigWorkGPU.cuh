#ifndef __APE_HT_CONFIG_WORK_CUH
#define __APE_HT_CONFIG_WORK_CUH
#include "APE/Work.hpp"
#include "../../../DataStructures/TrigMuonDataStructs/TrigMuonDataStructs/TrigMuonAccelerationEDM.h"
#include "../include/muonDataContexts.hpp"
#include <vector>

namespace APE{
  class HTConfigWorkGPU:public APE::Work{
  public:
    HTConfigWorkGPU(const HoughTransformContextGPU&, std::shared_ptr<APE::BufferContainer> input);
    ~HTConfigWorkGPU();
    std::shared_ptr<APE::BufferContainer> getOutput();
    void run();
    const std::vector<double>& getStats();
  private:
    HoughTransformContextGPU* m_context;

    std::shared_ptr<APE::BufferContainer> m_input;
    std::vector<double> m_stats;
    //
    //void saveConfiguration(HT_ALGO_CONFIGURATION* config);

  };
}

#endif
