#ifndef __APE_MUON_DATA_CONTEXTS_HPP
#define __APE_MUON_DATA_CONTEXTS_HPP
#include "../../../DataStructures/TrigMuonDataStructs/TrigMuonDataStructs/TrigMuonAccelerationEDM.h"
#include <vector>

namespace APE{

  class HoughTransformContext {
  public:
    //HoughTransformContext (const INPUT_MDT_DATA& mdtData, const HT_ALGO_CONFIGURATION& config) :
    HoughTransformContext (const INPUT_HT_DATA& mdtData, const HT_ALGO_CONFIGURATION& config) :
      m_HTData(mdtData), m_HTConfig(config)
    {};

    HoughTransformContext (const HoughTransformContext& c) : m_HTData(c.m_HTData), m_HTConfig(c.m_HTConfig)
    {};

    // const INPUT_MDT_DATA& m_HTData;
    const INPUT_HT_DATA& m_HTData;
    const HT_ALGO_CONFIGURATION& m_HTConfig;

  };

  struct HoughTransformDeviceContext {
  public:
    HoughTransformDeviceContext() { HoughTransformDeviceContext(-1); };
    HoughTransformDeviceContext(const int& dev) : m_deviceId(dev), h_HTData(0), d_HTData(0) {};
    int m_deviceId;
    unsigned char *h_HTData;
    unsigned char *d_HTData;
    unsigned char *h_HTConfig;
    unsigned char *d_HTConfig;
    unsigned char *h_HTConfig_new;
    unsigned char *d_HTConfig_new;

  };

  class HoughTransformContextGPU : public HoughTransformContext {
  public:
    //HoughTransformContextGPU (const INPUT_MDT_DATA& mdtData, const HT_ALGO_CONFIGURATION& config, HoughTransformDeviceContext& devC) : HoughTransformContext(mdtData, config), m_devC(devC)
    HoughTransformContextGPU (const INPUT_HT_DATA& mdtData, const HT_ALGO_CONFIGURATION& config, HoughTransformDeviceContext& devC) : HoughTransformContext(mdtData, config), m_devC(devC)
    {};
    HoughTransformContextGPU (const HoughTransformContextGPU& c) : HoughTransformContext(c.m_HTData, c.m_HTConfig), m_devC(c.m_devC)
    {};
    HoughTransformDeviceContext& m_devC;
  };

}

#endif
