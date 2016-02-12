#ifndef __APE_HT_CONFIG_WORK_CUH
#define __APE_HT_CONFIG_WORK_CUH
#include "APE/Work.hpp"
#include "../include/muonDataContexts.hpp"
#include <vector>
#include "tbb/concurrent_queue.h"

#include "../../../DataStructures/TrigMuonDataStructs/TrigMuonDataStructs/TrigMuonAccelerationEDM.h"

namespace APE{

  class HTConfigWorkGPU:public APE::Work{
  public:
    HTConfigWorkGPU(const HoughTransformContextGPU&,
		    std::shared_ptr<APE::BufferContainer> input,
		      tbb::concurrent_queue<APE::Work*>& compQ,
		      tbb::concurrent_queue<HoughTransformDeviceContext*>& ctxQ);
    ~HTConfigWorkGPU();
    std::shared_ptr<APE::BufferContainer> getOutput();
    bool run();
    const std::vector<double>& getStats();

  private:
    HoughTransformContextGPU* m_context;

    std::shared_ptr<APE::BufferContainer> m_input;
    std::vector<double> m_stats;
    tbb::concurrent_queue<APE::Work*>& m_outputQueue;
    tbb::concurrent_queue<HoughTransformDeviceContext*>& m_contextQueue;

  };
}

#endif
