#ifndef __APE_HT_CONFIG_WORK_HPP
#define __APE_HT_CONFIG_WORK_HPP
#include "APE/Work.hpp"
#include "../../../DataStructures/TrigMuonDataStructs/TrigMuonDataStructs/TrigMuonAccelerationEDM.h"
#include "../include/muonDataContexts.hpp"
#include <vector>

namespace APE{
  class HTConfigWork:public APE::Work{
  public:
    HTConfigWork(const HoughTransformContext&, std::shared_ptr<APE::BufferContainer> input);
    ~HTConfigWork();
    std::shared_ptr<APE::BufferContainer> getOutput();
    bool run();
    const std::vector<double>& getStats();
  private:
    HoughTransformContext* m_context;

    std::shared_ptr<APE::BufferContainer> m_input;
    std::vector<double> m_stats;
    //
    void saveConfiguration(HT_ALGO_CONFIGURATION* config);

  };
}

#endif
