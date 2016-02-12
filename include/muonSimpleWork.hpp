#ifndef __APE_MUON_SIMPLE_WORK_HPP
#define __APE_MUON_SIMPLE_WORK_HPP
#include "APE/Work.hpp"
#include "../include/muonDataContexts.hpp"
#include <vector>

namespace APE{

  class muonSimpleWork:public APE::Work{
  public:
    muonSimpleWork(const HoughTransformContext&, std::shared_ptr<APE::BufferContainer> input);
    ~muonSimpleWork();
    std::shared_ptr<APE::BufferContainer> getOutput();
    bool run();
    const std::vector<double>& getStats();
  private:
    HoughTransformContext* m_context;

    std::shared_ptr<APE::BufferContainer> m_input;
    std::vector<double> m_stats;
    //
    //    void makeHTPatterns(INPUT_MDT_DATA* mdtHitArray);

  };
}

#endif
