#ifndef __APE_MUON_SIMPLE_WORK_GPU_HPP
#define __APE_MUON_SIMPLE_WORK_GPU_HPP
#include "APE/Work.hpp"
#include "../include/muonDataContexts.hpp"
#include <vector>
#include "tbb/concurrent_queue.h"

namespace APE{

  class muonSimpleWorkGPU:public APE::Work{
  public:
    muonSimpleWorkGPU(const HoughTransformContextGPU& ctx,
		      std::shared_ptr<APE::BufferContainer> data,
		      tbb::concurrent_queue<APE::Work*>& compQ,
		      tbb::concurrent_queue<HoughTransformDeviceContext*>& ctxQ);
    ~muonSimpleWorkGPU();
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
