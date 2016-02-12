#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>       // helper functions for CUDA timing and initialization
#include <helper_functions.h>  // helper functions for timing, string parsing

#include "../include/muonDeviceWrapperGPU.cuh"
#include "../include/HoughKernels.cuh"

#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iomanip>

namespace TrigMuonModuleKernels{

  DeviceManager::DeviceManager() : nDevices(0)
  {
    checkCudaErrors(cudaGetDeviceCount(&nDevices));
  }

  APE::HoughTransformDeviceContext*
  DeviceManager::createHTContext(const int& devId){

    checkCudaErrors(cudaSetDevice(devId));
    APE::HoughTransformDeviceContext* p = new APE::HoughTransformDeviceContext;
    p->m_deviceId = devId;
    
    checkCudaErrors(cudaMalloc((void **)&p->d_HTConfig, sizeof(HT_ALGO_CONFIGURATION)));
    checkCudaErrors(cudaMalloc((void **)&p->d_HTData, sizeof(INPUT_HT_DATA)));
    checkCudaErrors(cudaMallocHost((void **)&p->h_HTConfig, sizeof(HT_ALGO_CONFIGURATION)));
    checkCudaErrors(cudaMallocHost((void **)&p->h_HTData, sizeof(INPUT_HT_DATA)));

    return p;

  }


  void DeviceManager::deleteHTContext(APE::HoughTransformDeviceContext* p){

    checkCudaErrors(cudaSetDevice(p->m_deviceId));
    
    checkCudaErrors(cudaFree(p->d_HTConfig));
    checkCudaErrors(cudaFree(p->d_HTData));
    checkCudaErrors(cudaFreeHost(p->h_HTConfig));
    checkCudaErrors(cudaFreeHost(p->h_HTData));

    delete p;

  }

  float wrpHoughCtx(const APE::HoughTransformDeviceContext& devC){
    
    checkCudaErrors(cudaSetDevice(devC.m_deviceId));
    //checkCudaErrors(cudaMemcpy(devC.d_HTConfig, devC.h_HTConfig, sizeof(HT_ALGO_CONFIGURATION), cudaMemcpyHostToDevice));

    double * curvGM;
    checkCudaErrors(cudaMalloc((void**)&curvGM, sizeof(double) * curvBins));

    copyCurvVals<<<1, curvBins*0.5>>>(curvGM);
    checkCudaErrors(cudaStreamSynchronize(0));

    copyCfgData((HT_ALGO_CONFIGURATION*) devC.h_HTConfig, curvGM);
    checkCudaErrors(cudaStreamSynchronize(0));

    return 0.;

  };

  std::vector<float> wrpHoughAlgo(const APE::HoughTransformDeviceContext& devC, MUON_HOUGH_RED_PATTERN *pOutput){

    std::vector<float> timeVec;

    struct timeval tStart, tMid1, tMid2, tEnd;
    float totalCUDATime = 0.;

    gettimeofday (&tStart, NULL);

    HT_ALGO_CONFIGURATION * hConf = reinterpret_cast<HT_ALGO_CONFIGURATION*>(devC.h_HTConfig);
    INPUT_HT_DATA* hData = reinterpret_cast<INPUT_HT_DATA*>(devC.h_HTData);
    int Nsec[2] = {hConf->steps.sectors.xyz, hConf->steps.sectors.rz};
    int NA[2] = {(int)(2*hConf->steps.ip.xy/hConf->steps.stepsize.xy), (int)hConf->steps.nbins_curved};//+2
    //S:12,16 A:16,160 B:1440,720
    int voteXY = hData->m_nVoteXY;
    int pattXY = hData->m_nPattXY;
    int voteCC = hData->m_nVoteRZ;
    int pattCC = hData->m_nPattRZ;

    gettimeofday (&tMid1, NULL);
    std::cout << "TOTAL Host Preliminary Stuff " << (((tMid1.tv_sec - tStart.tv_sec)*1000000L +tMid1.tv_usec) - tStart.tv_usec) * 0.001 << " ms" <<  std::endl;

    // stomp a foot on the device    
    checkCudaErrors(cudaSetDevice(devC.m_deviceId));

    cudaDeviceProp prop;
    checkCudaErrors(cudaGetDeviceProperties(&prop, devC.m_deviceId));
    int gpuProps[2] = {prop.warpSize, prop.maxThreadsPerBlock};

    gettimeofday (&tMid2, NULL);
    std::cout << "TOTAL DEVICE Preliminary Stuff " << (((tMid2.tv_sec - tMid1.tv_sec)*1000000L +tMid2.tv_usec) - tMid1.tv_usec) * 0.001 << " ms" <<  std::endl;

    // create and start CUDA timer
    StopWatchInterface *timerCUDA = 0;
    sdkCreateTimer(&timerCUDA);
    sdkResetTimer(&timerCUDA);
    sdkStartTimer(&timerCUDA);

    // Copy from Host to Device
    checkCudaErrors(cudaMemcpyAsync(devC.d_HTData, devC.h_HTData, sizeof(INPUT_HT_DATA), cudaMemcpyHostToDevice, 0));

    // allocate output mem area
    MUON_HOUGH_RED_PATTERN * pOut;
    checkCudaErrors(cudaMalloc((void **)&pOut, sizeof(MUON_HOUGH_RED_PATTERN)));
    //checkCudaErrors(cudaMemcpyAsync(pOut, pOutput, sizeof(MUON_HOUGH_RED_PATTERN), cudaMemcpyHostToDevice, 0));

    int * votXYHit, * assXYHit, * votCCHit, * assCCHit;
    checkCudaErrors(cudaMalloc((void **)&votXYHit, sizeof(int) * voteXY));
    checkCudaErrors(cudaMalloc((void **)&assXYHit, sizeof(int) * pattXY));
    checkCudaErrors(cudaMalloc((void **)&votCCHit, sizeof(int) * voteCC));
    checkCudaErrors(cudaMalloc((void **)&assCCHit, sizeof(int) * pattCC));
    /*
    int * b_xy_maxes, * v_xy_maxes;
    checkCudaErrors(cudaMalloc(&b_xy_maxes, sizeof(int) * Nsec[0] * NA[0]));
    checkCudaErrors(cudaMalloc(&v_xy_maxes, sizeof(int) * Nsec[0] * NA[0]));*/
    int * b_cc_maxes, * v_cc_maxes;
    checkCudaErrors(cudaMalloc(&b_cc_maxes, sizeof(int) * Nsec[1] * NA[1]));
    checkCudaErrors(cudaMalloc(&v_cc_maxes, sizeof(int) * Nsec[1] * NA[1]));

    int * devProps;
    checkCudaErrors(cudaMalloc((void**)&devProps, 2*sizeof(int)));
    checkCudaErrors(cudaMemcpyAsync(devProps, gpuProps, 2*sizeof(int), cudaMemcpyHostToDevice, 0));
    /*
    int * s_xy_max, * b_xy_max, * v_xy_max;
    checkCudaErrors(cudaMalloc(&s_xy_max, sizeof(int) * Nsec[0]));
    checkCudaErrors(cudaMalloc(&b_xy_max, sizeof(int) * Nsec[0]));
    checkCudaErrors(cudaMalloc(&v_xy_max, sizeof(int) * Nsec[0]));*/
    int * s_cc_max, * b_cc_max, * v_cc_max;
    checkCudaErrors(cudaMalloc(&s_cc_max, sizeof(int) * Nsec[1]));
    checkCudaErrors(cudaMalloc(&b_cc_max, sizeof(int) * Nsec[1]));
    checkCudaErrors(cudaMalloc(&v_cc_max, sizeof(int) * Nsec[1]));

    int * controls;
    checkCudaErrors(cudaMalloc((void**)&controls, sizeof(int) * 4));

    sdkStopTimer(&timerCUDA);
    float TimerCUDASpan = sdkGetAverageTimerValue(&timerCUDA);

    timeVec.push_back(TimerCUDASpan);
    totalCUDATime += TimerCUDASpan;

    std::cout << "Input allocation time: " << TimerCUDASpan << " ms" << std::endl;

    sdkResetTimer(&timerCUDA);
    sdkStartTimer(&timerCUDA);

    int * monitor;
    checkCudaErrors(cudaMalloc(&monitor, sizeof(int)*NA[1]));

    //houghAlgo<<< 1, 1 >>>(devProps, controls, votXYHit, assXYHit, votCCHit, assCCHit, (INPUT_HT_DATA*) devC.d_HTData, b_xy_maxes, v_xy_maxes, b_cc_maxes, v_cc_maxes, s_xy_max, b_xy_max, v_xy_max, s_cc_max, b_cc_max, v_cc_max, pOut);
    houghAlgo<<< 1, 1 >>>(devProps, controls, votXYHit, assXYHit, votCCHit, assCCHit, (INPUT_HT_DATA*) devC.d_HTData, b_cc_maxes, v_cc_maxes, s_cc_max, b_cc_max, v_cc_max, pOut, monitor);
    checkCudaErrors(cudaStreamSynchronize(0));
    getLastCudaError("Wrapper Kernel execution failed");

    //int * h_xy=(int*)malloc(sizeof(int)*Nsec[0]*NA[0]);
    int * h_cc=(int*)malloc(sizeof(int)*Nsec[1]*NA[1]);
    //checkCudaErrors(cudaMemcpyAsync((void*)h_xy, v_xy_maxes, sizeof(int)*Nsec[0]*NA[0], cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy((void*)h_cc, v_cc_maxes, sizeof(int)*Nsec[1]*NA[1], cudaMemcpyDeviceToHost));
    /*
    std::cout << "XY\n";
    for(int i=0;i<Nsec[0]*NA[0];i++)
      std::cout << i << ":" << h_xy[i] << " ";*/
    std::cout << "\nCYLINDER\n";
    for(int i=0;i<Nsec[1]*NA[1];i++)
      std::cout << (int)((i+0.)/NA[1]) << ":" << h_cc[i] << " ";

    sdkStopTimer(&timerCUDA);
    TimerCUDASpan = sdkGetAverageTimerValue(&timerCUDA);


    int * h_mon=(int*)malloc(sizeof(int)*NA[1]);
    checkCudaErrors(cudaMemcpy((void*)h_mon, monitor, sizeof(int)*NA[1], cudaMemcpyDeviceToHost));
    std::cout << "\n";
    for(int y=0;y<NA[1];y++)
      std::cout << y << ":" << h_mon[y] << " ";//+0.)/100000. << " ";
    std::cout << "\n";

    timeVec.push_back(TimerCUDASpan);
    totalCUDATime += TimerCUDASpan;
    std::cout <<"Wrapper kernel execution " << TimerCUDASpan << " ms";// << std::endl;
 
    sdkResetTimer(&timerCUDA);
    sdkStartTimer(&timerCUDA);
    checkCudaErrors(cudaMemcpyAsync((void*)pOutput, pOut, sizeof(MUON_HOUGH_RED_PATTERN), cudaMemcpyDeviceToHost));
    sdkStopTimer(&timerCUDA);
    TimerCUDASpan = sdkGetAverageTimerValue(&timerCUDA);

    timeVec.push_back(TimerCUDASpan);
    totalCUDATime += TimerCUDASpan;
    std::cout <<"Output copy to host " << TimerCUDASpan << " ms" << std::endl;

    checkCudaErrors(cudaFree(devProps));
    checkCudaErrors(cudaFree(controls));/*
    checkCudaErrors(cudaFree(b_xy_maxes));
    checkCudaErrors(cudaFree(v_xy_maxes));*/
    checkCudaErrors(cudaFree(b_cc_maxes));
    checkCudaErrors(cudaFree(v_cc_maxes));/*
    checkCudaErrors(cudaFree(s_xy_max));
    checkCudaErrors(cudaFree(b_xy_max));
    checkCudaErrors(cudaFree(v_xy_max));*/
    checkCudaErrors(cudaFree(s_cc_max));
    checkCudaErrors(cudaFree(b_cc_max));
    checkCudaErrors(cudaFree(v_cc_max));
    checkCudaErrors(cudaFree(votXYHit));
    checkCudaErrors(cudaFree(assXYHit));
    checkCudaErrors(cudaFree(votCCHit));
    checkCudaErrors(cudaFree(assCCHit));
    checkCudaErrors(cudaFree(pOut));

    checkCudaErrors(cudaStreamSynchronize(0));
 
    gettimeofday (&tEnd, NULL);
    float totalRUNTime = (((tEnd.tv_sec - tStart.tv_sec)*1000000L +tEnd.tv_usec) - tStart.tv_usec) * 0.001;
    std::cout << "TOTAL RUNNING TIME " << totalRUNTime << " ms\nTOTAL CUDA TIME " << totalCUDATime << " ms" << std::endl;
    return timeVec;

  };

}