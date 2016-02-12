#ifndef CUDAFUNCS_H
#define CUDAFUNCS_H

#include <vector>

//#include "../include/HTHelpers.cuh"
#include "../include/muonDataContexts.hpp"

namespace TrigMuonModuleKernels{

  class DeviceManager {

  public:
    
    DeviceManager();
    ~DeviceManager(){};

    inline const int& getNDevice(){ return nDevices; };

    APE::HoughTransformDeviceContext* createHTContext(const int&);
    void deleteHTContext(APE::HoughTransformDeviceContext*);

    int nDevices;

  };

  float wrpHoughCtx(const APE::HoughTransformDeviceContext& dC);
  std::vector<float> wrpHoughAlgo(const APE::HoughTransformDeviceContext& dC, MUON_HOUGH_RED_PATTERN *pOutput);

}

#endif