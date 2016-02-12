#ifndef __APE_MUON_SIMPLE_WORK_GPU_CUH
#define __APE_MUON_SIMPLE_WORK_GPU_CUH
#include "APE/Work.hpp"
#include "../include/muonDataContexts.hpp"
#include <vector>

namespace APE{

  class muonSimpleWorkGPU:public APE::Work{
  public:
    muonSimpleWorkGPU(const HoughTransformContextGPU&, std::shared_ptr<APE::BufferContainer> input);
    ~muonSimpleWorkGPU();
    std::shared_ptr<APE::BufferContainer> getOutput();
    void run();
    const std::vector<double>& getStats();


  private:
    HoughTransformContextGPU* m_context;

    std::shared_ptr<APE::BufferContainer> m_input;
    std::vector<double> m_stats;

    void makeHTPatterns();


  };
}

#endif
