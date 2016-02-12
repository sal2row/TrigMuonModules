#pragma once

#include "../../../DataStructures/TrigMuonDataStructs/TrigMuonDataStructs/TrigMuonAccelerationEDM.h"

#define curvBins 160

__global__ void copyCurvVals(double * curvGM);

__host__ void copyCfgData(HT_ALGO_CONFIGURATION * cfgData, double * curvGM);

__global__ void siftHit(INPUT_HT_DATA *mdtData, int *voteXYData, int *assXYData, int *voteCCData, int *assCCData);

__global__ void houghCCSpace(int * v_data, INPUT_HT_DATA *mdtData, int * b_maxes, int * v_maxes, int * monitor);

__global__ void voteHoughSpace(int * v_data, INPUT_HT_DATA *mdtData, int * b_maxes, int * v_maxes);

__global__ void voteCCHoughSpace(int * v_data, INPUT_HT_DATA *mdtData, int * b_maxes, int * v_maxes);

__global__ void sorter(int * b_maxes, int * v_maxes, int * s_max, int * b_max, int * v_max, int * len);

__global__ void computeOutput(int * a_data, INPUT_HT_DATA *mdtData, int * b_max, int * s_max, int * v_max, MUON_HOUGH_RED_PATTERN *pOut, int level, int d_max, int * nxt);

__global__ void houghAlgo(int * devProps, int * controls, int * vot_xy_hits, int * ass_xy_hits, int * vot_cc_hits, int * ass_cc_hits, INPUT_HT_DATA *mdtData, int * b_cc_maxes, int * v_cc_maxes, int * s_cc_max, int * b_cc_max, int * v_cc_max, MUON_HOUGH_RED_PATTERN *pOut, int * monitor);
//__global__ void houghAlgo(int * devProps, int * controls, int * vot_xy_hits, int * ass_xy_hits, int * vot_cc_hits, int * ass_cc_hits, INPUT_HT_DATA *mdtData, int * b_xy_maxes, int * v_xy_maxes, int * b_cc_maxes, int * v_cc_maxes, int * s_xy_max, int * b_xy_max, int * v_xy_max, int * s_cc_max, int * b_cc_max, int * v_cc_max, MUON_HOUGH_RED_PATTERN *pOut);