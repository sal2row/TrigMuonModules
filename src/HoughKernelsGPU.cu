#include "../include/HoughKernels.cuh"
#include "../include/Common.h"
#include "../../../DataStructures/TrigMuonDataStructs/TrigMuonDataStructs/TrigMuonAccelerationEDM.h"
#include <cuda.h>
#include <cuda_runtime.h>

// includes, project
#include <helper_cuda.h>
#include <helper_functions.h> // helper utility functions

#include <math.h>
#include <algorithm>
#include <unistd.h>

#include <thrust/sort.h>

using namespace std;

////////////////////////////////////// HELPER KERNELS //////////////////////////////////////

__device__ int get3DIndex(int s, int a, int b, int nA, int nB){
  return (s * (nA*nB))+((a * nB) + b);
}

__device__ int get2DIndex(int s, int b, int N){
  return s * N + b;
}

__device__ int coordToBin(double x, double min, double binw) {return (int) floor((x-min)/binw);}

__device__ int sector(double theta, int nof_sectors)
{
  //double theta = atan2(radius,z);
  //int sectorhit = (int) rint(theta * nof_sectors * M_1_PI);
  int sectorhit = (int) floor(theta * nof_sectors * M_1_PI);
  sectorhit -= (sectorhit == nof_sectors) ? 1 : 0;
  return sectorhit;
}

__device__ double calcAngle(double x, double y, double r)
{
  double heigth_squared = x*x + y*y - r*r;
  double phi = (heigth_squared>=0) ? atan2(y,x)+atan2(r,sqrt(heigth_squared)) : atan2(y,x);
  phi += (phi < 0) ? 2.*M_PI : 0;
  phi -= (phi > 2.*M_PI) ? 2.*M_PI : 0;
  return phi;
}


__constant__ HT_ALGO_CONFIGURATION cfgPar;

__constant__ double invCurv[curvBins];

__global__ void copyCurvVals(double * curvGM){

  __shared__ double curvs[curvBins];

  double x0 = threadIdx.x < 9 ? 19.5 - threadIdx.x * 0.5 : -threadIdx.x+24;
  curvs[threadIdx.x] = (x0-19.75)/(-3. * 3500. * 20.25);
  curvs[threadIdx.x+blockDim.x] = threadIdx.x < 11 ? 1. : (threadIdx.x < 21 ? 1.5 - 0.05*threadIdx.x : 0.5);

  curvGM[threadIdx.x] = curvs[threadIdx.x];
  __syncthreads();//intended to avoid coalescing conflicts which might slow the business down
  curvGM[threadIdx.x+blockDim.x] = curvs[threadIdx.x+blockDim.x];

}

__host__ void copyCfgData(HT_ALGO_CONFIGURATION * cfgData, double * curvGM){

  cudaMemcpyToSymbol(cfgPar, cfgData, sizeof(HT_ALGO_CONFIGURATION));
  cudaMemcpyToSymbol(invCurv, curvGM, sizeof(double) * curvBins, 0, cudaMemcpyDeviceToDevice);
  cudaFree(curvGM);

}


////////////////////////////////////// REAL KERNELS //////////////////////////////////////

__global__ void siftHit(INPUT_HT_DATA *mdtData, int *voteXYData, int *assXYData, int *voteCCData, int *assCCData)
{

  __shared__ int v_xy;
  __shared__ int a_xy;
  __shared__ int v_cc;
  __shared__ int a_cc;
  __shared__ int size;
  if(threadIdx.x == 0){
    v_xy = 0;
    a_xy = 0;
    v_cc = 0;
    a_cc = 0;
    size = mdtData->m_nDataWords;
  }

  __syncthreads();

  bool useCSCvote = cfgPar.det.use_csc_in_hough;
  bool useCSCpatt = cfgPar.det.use_csc_in_pattern;

#pragma unroll
for(int i = threadIdx.x; i < size; i += blockDim.x){

  bool measPhi = mdtData->m_meas_phi[i];
  int detTech = mdtData->m_det_tech[i];
  double prob = mdtData->m_prob[i];

  bool voteBase = ( !mdtData->m_ass[i] && ( useCSCvote || (!useCSCvote && detTech == 1) ) &&
                   ( !cfgPar.algo.weightcutmdt || detTech != 0 || prob >= cfgPar.steps.weight_mdt ) &&
		   ( !cfgPar.algo.weightcut || prob >= cfgPar.algo.weight ));
  bool pattBase = ( (useCSCpatt == 1) || (useCSCpatt == 0 && detTech != 1) );

  bool voteXY = voteBase && measPhi;
  bool pattXY = pattBase && measPhi;

  bool voteCC = voteBase && !measPhi;
  bool pattCC = pattBase && !measPhi;

  int votXYIdx = voteXY ? atomicAdd(&v_xy, 1) : -1;
  int assXYIdx = pattXY ? atomicAdd(&a_xy, 1) : -1;
  int votCCIdx = voteCC ? atomicAdd(&v_cc, 1) : -1;
  int assCCIdx = pattCC ? atomicAdd(&a_cc, 1) : -1;

  if(voteXY)
    voteXYData[votXYIdx] = i;
  if(pattXY)
    assXYData[assXYIdx] = i;

  if(voteCC)
    voteCCData[votCCIdx] = i;
  if(pattCC)
    assCCData[assCCIdx] = i;

  }

  __syncthreads();

  if(threadIdx.x == 0){
    mdtData->m_nVoteXY = v_xy;
    mdtData->m_nPattXY = a_xy;
    mdtData->m_nVoteRZ = v_cc;
    mdtData->m_nPattRZ = a_cc;
  }

}

__global__ void houghCCSpace(int * v_data, INPUT_HT_DATA *mdtData, int * b_maxes, int * v_maxes, int * monitor){

  int NB = (int)((cfgPar.steps.angle.rz/cfgPar.steps.stepsize_angle.rz));
  extern __shared__ int bins[];
  int * thetaBins = &bins[0];
  int * maxBins = &bins[NB];
  int * maxVals = &bins[NB+blockDim.x];
  int * size = &bins[NB+2*blockDim.x];
  int * maxBin = &bins[1+NB+2*blockDim.x];

#pragma unroll
  for(int j=threadIdx.x; j<NB; j += blockDim.x){
    thetaBins[j] = 0;
    if(j<blockDim.x){
      maxBins[j] = -1;
      maxVals[j] = -1;
    }
  }

  if(threadIdx.x == 0){
    size[0] = mdtData->m_nVoteRZ;
    maxBin[0] = 0;
  }
  __syncthreads();

  int sz_wght = 20.*sqrt(size[0]*0.000142857);
  int halfAGrid = (int)(0.5*gridDim.x);
  double Amax = cfgPar.steps.ip.rz;
  double Bwdt = cfgPar.steps.stepsize_angle.rz;

  int h_scl = 10000;

#pragma unroll
  for(int i = threadIdx.x; i < size[0] ; i += blockDim.x){

    double hitth = mdtData->m_theta[v_data[i]];
    //int phiSec = 2*mdtData->m_phiSec[v_data[i]];
    int phiSec = 2*mdtData->m_phiSec[v_data[i]];
    bool isBar = mdtData->m_barrel[v_data[i]];
    double magRat = mdtData->m_mgRatio[v_data[i]];
    double wgh = mdtData->m_weight[v_data[i]];
    int signedA = blockIdx.x%halfAGrid;
    int sideSign = blockIdx.x - signedA == 0 ? -1 : 1;
    double ratio = magRat*invCurv[signedA];
    double ratSides[2] = {blockIdx.x > 0 ? magRat*invCurv[signedA-1] : magRat*invCurv[halfAGrid-1],
    	   	          blockIdx.x < gridDim.x-1 ? magRat*invCurv[signedA+1] : magRat*invCurv[0]};
    int sideSecs[2] = {blockIdx.y == 0 ? (int)gridDim.y-1 : (int)blockIdx.y-1, blockIdx.y == gridDim.y-1 ? 0 : (int)blockIdx.y+1};
    bool secCond = (phiSec == blockIdx.y || phiSec == sideSecs[0] || phiSec == sideSecs[1]);
    bool csqn = isBar && fabs(ratio) <= 0.5 && secCond;
    bool csqnSides[2] = {isBar && fabs(ratSides[0]) <= 0.5 && phiSec == blockIdx.y,
    	 	      	 isBar && fabs(ratSides[1]) <= 0.5 && phiSec == blockIdx.y};
    double rsin = sin(ratio);
    double rsinSides[2] = {sin(ratSides[0]), sin(ratSides[1])};
    double theta = hitth + (rsin*sideSign);//csqn ? hitth + (rsin*sideSign) : -1;
    double thetaSides[2] = {csqnSides[0] ? hitth + (rsinSides[0]*sideSign) : -1,//sideSign qui va differenziato...
    	   		    csqnSides[1] ? hitth + (rsinSides[1]*sideSign) : -1};
    csqn &= theta > 0. && theta < M_PI;
    csqnSides[0] &= (thetaSides[0] > 0. && thetaSides[0] < M_PI);
    csqnSides[1] &= (thetaSides[1] > 0. && thetaSides[1] < M_PI);
    double crv_wg = csqn ? (0.5+0.5*sin(theta)) * wgh / (1.+sz_wght*rsin) : 0.;
    double crv_wgSides[2] = {csqnSides[0] ? (0.5+0.5*sin(thetaSides[0])) * wgh / (1.+sz_wght*rsinSides[0]) : 0.,
    	   		     csqnSides[1] ? (0.5+0.5*sin(thetaSides[1])) * wgh / (1.+sz_wght*rsinSides[1]) : 0.};
    int tBin = (int)(theta*57.29578/Bwdt);
    int tBinSides[2] = {(int)(thetaSides[0]*57.29578/Bwdt), (int)(thetaSides[0]*57.29578/Bwdt)};
    if(blockIdx.y == 4)
      atomicMax(maxBin, phiSec == blockIdx.y ? 10*phiSec+1000+100000000+(v_data[i]*1000000)+(int)isBar : 10*phiSec+10000+100000000+(v_data[i]*1000000+(int)isBar));//(int)(theta*57.29578*1000));
    if(phiSec == blockIdx.y){
      //atomicAdd(thetaBins + tBin, h_scl*crv_wg);
      /*atomicAdd(thetaBins + tBinSides[0], 0.2*h_scl*crv_wgSides[0]);
      atomicAdd(thetaBins + tBinSides[1], 0.2*h_scl*crv_wgSides[1]);
      atomicAdd(thetaBins + tBinSides[2], 0.2*h_scl*crv_wgSides[2]);
      atomicAdd(thetaBins + tBinSides[3], 0.2*h_scl*crv_wgSides[3]);*/
    }/*
    if(phiSec == sideSecs[0] && phiSec < gridDim.y && gridDim.x > 2){
      if(aBin[0] == blockIdx.x)
        atomicAdd(thetaBins + tBin[0], 0.8*h_scl*crv_wg[0]);
      if(aBin[1] == blockIdx.x)
        atomicAdd(thetaBins + tBin[1], 0.8*h_scl*crv_wg[1]);
    }
    if(phiSec == sideSecs[0] && phiSec >= 0 && gridDim.x > 2){
      if(aBin[0] == blockIdx.x)
        atomicAdd(thetaBins + tBin[0], 0.8*h_scl*crv_wg[0]);
      if(aBin[1] == blockIdx.x)
        atomicAdd(thetaBins + tBin[1], 0.8*h_scl*crv_wg[1]);
    }*/
  }

  __syncthreads();
  if(blockIdx.y == 4)
    monitor[blockIdx.x] = maxBin[0];
  /*
  for(int k = threadIdx.x; k<NB; k += blockDim.x){
    bool cond = maxVals[threadIdx.x] > thetaBins[k];
    maxBins[threadIdx.x] = cond ? maxBins[threadIdx.x] : (maxVals[threadIdx.x]  == thetaBins[k] ? (k < maxBins[threadIdx.x] ? k : maxBins[threadIdx.x]) : k);
    maxVals[threadIdx.x] = cond ? maxVals[threadIdx.x] : thetaBins[k];
  }

  __syncthreads();

  for(unsigned int t = (int)(blockDim.x*0.5); t > 0; t>>=1){

    if (threadIdx.x < t){
      bool cond = maxVals[threadIdx.x] > maxVals[threadIdx.x + t];
      maxBins[threadIdx.x] = cond ? maxBins[threadIdx.x] : (maxVals[threadIdx.x] == maxVals[threadIdx.x + t] ? (maxBins[threadIdx.x] <= maxBins[threadIdx.x + t] ? maxBins[threadIdx.x] : maxBins[threadIdx.x + t]) : maxBins[threadIdx.x + t]);
      maxVals[threadIdx.x] = cond ? maxVals[threadIdx.x] : maxVals[threadIdx.x + t];
    }
    __syncthreads();
  }

  if(threadIdx.x == 0){
    bool cond = maxVals[0] > (int)(cfgPar.steps.threshold.xyz*10000.);
    b_maxes[blockIdx.y * gridDim.x + blockIdx.x] = cond ? (int)(blockIdx.x + (maxBins[0]*gridDim.x)) : -1;
    v_maxes[blockIdx.y * gridDim.x + blockIdx.x] = maxVals[0];//cond ? maxVals[0] : (int)(cfgPar.steps.threshold.xyz*10000.);

  }*/




}

__global__ void voteHoughSpace(int * v_data, INPUT_HT_DATA *mdtData, int * b_maxes, int * v_maxes)
{

  int NB = cfgPar.steps.angle.xyz/cfgPar.steps.stepsize_angle.xyz;

  extern __shared__ int bins[];
  int * phiBins = &bins[0];
  int * maxBins = &bins[NB];
  int * maxVals = &bins[NB+blockDim.x];
  int * size = &bins[NB+2*blockDim.x];

#pragma unroll
  for(int j=threadIdx.x; j<NB; j += blockDim.x){
    phiBins[j] = 0;
    if(j<blockDim.x){
      maxBins[j] = -1;
      maxVals[j] = -1;
    }
  }

  if(threadIdx.x == 0)
    size[0] = mdtData->m_nVoteXY;

  __syncthreads();

  double Amin = -cfgPar.steps.ip.xy;
  double Amax = cfgPar.steps.ip.xy;
  double Bmin = 0.;
  double Awdt = cfgPar.steps.stepsize.xy;
  double Bwdt = cfgPar.steps.stepsize_angle.xyz;
  /*int bins[512];
  int values[512];*/

  int h_scl = 5000;
  int th_scl = 3333;

#pragma unroll
  for(int i = threadIdx.x; i < size[0] ; i += blockDim.x){

    double radius = mdtData->m_radius[v_data[i]];
    double rEdges[2] = {(-radius < Amin) ? Amin+(0.5+blockIdx.x)*Awdt : -radius+0.00001+(blockIdx.x*Awdt),
    	   	        (radius > Amax) ? Amax : radius};
    double rSides[2] = {(-radius-Awdt < Amin) ? Amin+(blockIdx.x-0.5)*Awdt : -radius+0.00001+((blockIdx.x-1)*Awdt),
                        (-radius+Awdt < Amin) ? Amin+(blockIdx.x+1.5)*Awdt : -radius+0.00001+((blockIdx.x+1)*Awdt)};
    double hitx = mdtData->m_hitx[v_data[i]], hity = mdtData->m_hity[v_data[i]];
    double phi = calcAngle(hitx, hity, rEdges[0]);
    double phiSds[2] = {calcAngle(hitx, hity, rSides[0]), calcAngle(hitx, hity, rSides[1])};
    double dotprod = hity * sin(phi) + hitx * cos(phi);
    double dpSds[2] = {hity * sin(phiSds[0]) + hitx * cos(phiSds[0]), hity * sin(phiSds[1]) + hitx * cos(phiSds[1])};
    phi *= (180.*M_1_PI);
    phiSds[0] *= (180.*M_1_PI);
    phiSds[1] *= (180.*M_1_PI);

    // non c'e' discrepanza thra theta dell hit e atan2 di r,z quindi hSec OK
    int hSec = sector(mdtData->m_theta[v_data[i]], gridDim.y);
    int aBin = coordToBin(rEdges[0], Amin, Awdt);
    int aBSds[2] = {coordToBin(rSides[0], Amin, Awdt), coordToBin(rSides[1], Amin, Awdt)};
    int bBin = coordToBin(phi, Bmin, Bwdt);
    int bBSds[2] = {coordToBin(phiSds[0], Bmin, Bwdt), coordToBin(phiSds[1], Bmin, Bwdt)};
    bool csqn = (dotprod >= 0. && rEdges[0] <= rEdges[1] && aBin == blockIdx.x && (hSec >= blockIdx.y-1 && hSec <= blockIdx.y+1));
    bool csqn_m = (dotprod >= 0. && rSides[0] <= rEdges[1] && aBSds[0] == blockIdx.x-1 && hSec == blockIdx.y);
    bool csqn_p = (dotprod >= 0. && rSides[1] <= rEdges[1] && aBSds[1] == blockIdx.x+1 && hSec == blockIdx.y);

    if(!csqn && !csqn_m && !csqn_p)//zozzata
      continue;

    double wgh = (csqn || csqn_m || csqn_p) ? mdtData->m_weight[v_data[i]] : 0.;

    if(hSec == blockIdx.y){
      if(aBin == blockIdx.x && csqn){
        atomicAdd(phiBins + bBin, (int)(wgh*h_scl*2));
      }
      if(aBSds[1] == blockIdx.x+1 && csqn_p){
        int b_Bin = coordToBin(phiSds[1]-Bwdt+cfgPar.steps.stepsize_angle.xyz, Bmin, Bwdt);
        int lwlBin = phiSds[1] < Bwdt ? b_Bin : bBSds[1] - 1;
        atomicAdd(phiBins + lwlBin, (int)(wgh*h_scl));
	atomicAdd(phiBins + bBSds[1], (int)(wgh*h_scl));
      }
      if(aBSds[0] == blockIdx.x-1 && csqn_m){
        int b_Bin = coordToBin(phiSds[0]+Bwdt-cfgPar.steps.stepsize_angle.xyz, Bmin, Bwdt);
        int uprBin = phiSds[0] + Bwdt > cfgPar.steps.stepsize_angle.xyz ? b_Bin : bBSds[0] + 1;
        atomicAdd(phiBins + uprBin, (int)(wgh*h_scl));
	atomicAdd(phiBins + bBSds[0], (int)(wgh*h_scl));
      }
    }
    if(hSec == blockIdx.y+1 && hSec < gridDim.y && aBin == blockIdx.x && csqn)
      atomicAdd(phiBins + bBin, (int)(wgh*th_scl));
    if(hSec == blockIdx.y-1 && hSec >= 0 && aBin == blockIdx.x && csqn)
      atomicAdd(phiBins + bBin, (int)(wgh*th_scl));

  }

  __syncthreads();

  for(int k = threadIdx.x; k<NB; k += blockDim.x){
    bool cond = maxVals[threadIdx.x] > phiBins[k];
    maxBins[threadIdx.x] = cond ? maxBins[threadIdx.x] : (maxVals[threadIdx.x]  == phiBins[k] ? (k < maxBins[threadIdx.x] ? k : maxBins[threadIdx.x]) : k);
    maxVals[threadIdx.x] = cond ? maxVals[threadIdx.x] : phiBins[k];
  }

  __syncthreads();

  for(unsigned int t = (int)(blockDim.x*0.5); t > 0; t>>=1){

    if (threadIdx.x < t){
      bool cond = maxVals[threadIdx.x] > maxVals[threadIdx.x + t];
      maxBins[threadIdx.x] = cond ? maxBins[threadIdx.x] : (maxVals[threadIdx.x] == maxVals[threadIdx.x + t] ? (maxBins[threadIdx.x] <= maxBins[threadIdx.x + t] ? maxBins[threadIdx.x] : maxBins[threadIdx.x + t]) : maxBins[threadIdx.x + t]);
      maxVals[threadIdx.x] = cond ? maxVals[threadIdx.x] : maxVals[threadIdx.x + t];
    }
    __syncthreads();
  }

  if(threadIdx.x == 0){
    bool cond = maxVals[0] > (int)(cfgPar.steps.threshold.xyz*10000.);
    b_maxes[blockIdx.y * gridDim.x + blockIdx.x] = cond ? (int)(blockIdx.x + (maxBins[0]*gridDim.x)) : -1;
    v_maxes[blockIdx.y * gridDim.x + blockIdx.x] = cond ? maxVals[0] : (int)(cfgPar.steps.threshold.xyz*10000.);

  }


}



__global__ void sorter(int * b_maxes, int * v_maxes, int * s_max, int * b_max, int * v_max, int * len){

  extern __shared__ int sh_max[];

  int * vs_mxs = &sh_max[0];
  int * bs_mxs = &sh_max[blockDim.x*blockDim.y];
  int * vs_max = &sh_max[2*blockDim.x*blockDim.y];
  int * ss_max = &sh_max[blockDim.x*(2*blockDim.y+1)];
  int * bs_max = &sh_max[blockDim.x*(2*blockDim.y+2)];
  int * rf = &sh_max[blockDim.x*(2*blockDim.y+3)];

  int offset = threadIdx.x * blockDim.y;
  int tIdx = offset + threadIdx.y;
  vs_mxs[tIdx] = v_maxes[tIdx];
  bs_mxs[tIdx] = b_maxes[tIdx];

  __syncthreads();

  for(unsigned int t = blockDim.y*0.5; t > 0; t>>=1){
    if(threadIdx.y < t){
      bool cond = vs_mxs[tIdx] > vs_mxs[tIdx + t];
      bs_mxs[tIdx] = cond ? bs_mxs[tIdx] : (vs_mxs[tIdx] == vs_mxs[tIdx + t] ? (bs_mxs[tIdx] <= bs_mxs[tIdx + t] ? bs_mxs[tIdx] : bs_mxs[tIdx + t]) : bs_mxs[tIdx + t]);
      vs_mxs[tIdx] = cond ? vs_mxs[tIdx] : vs_mxs[tIdx + t];
    }
    __syncthreads();
  }

  if(threadIdx.y == 0){
    bool cond = vs_mxs[offset] > (int)(cfgPar.steps.threshold.xyz*10000.);
    s_max[threadIdx.x] = threadIdx.x;
    v_max[threadIdx.x] = cond ? vs_mxs[offset] : (int)(cfgPar.steps.threshold.xyz*10000.);
    b_max[threadIdx.x] = cond ? bs_mxs[offset] : -1;
  }

  if(threadIdx.x == 0 && threadIdx.y == 0){
    int sft = 1;
    rf[0] = blockDim.x-1;
#pragma unroll
    while(sft != sizeof(int)*2){
      rf[0] |= rf[0] >> sft;
      sft *= 2;
    }
    rf[0]++;
  }

  __syncthreads();

  if(threadIdx.y == 0){

  int i = 2;
  while(i <= rf[0]){

    if((threadIdx.x%i) == 0){
      int ind1 = threadIdx.x;
      int endind1 = ind1 + i*0.5;
      int ind2 = endind1;
      int endind2 = ind2 + i*0.5;
      endind2 = endind2 <= blockDim.x ? endind2 : blockDim.x;
      int targInd = threadIdx.x;
      int done = 0;
      
      while(!done){

	if ((ind1 == endind1) && (ind2 < endind2)){
          ss_max[targInd] = s_max[ind2];
          bs_max[targInd] = b_max[ind2];
          vs_max[targInd++] = v_max[ind2++];
	}
        else if ((ind2 == endind2) && (ind1 < endind1)){
          ss_max[targInd] = s_max[ind1];
          bs_max[targInd] = b_max[ind1];
          vs_max[targInd++] = v_max[ind1++];
	}
        else if (v_max[ind1] < v_max[ind2]){
          ss_max[targInd] = s_max[ind2];
          bs_max[targInd] = b_max[ind2];
          vs_max[targInd++] = v_max[ind2++];
	}
        else {
          ss_max[targInd] = s_max[ind1];
          bs_max[targInd] = b_max[ind1];
          vs_max[targInd++] = v_max[ind1++];
	}
        if ((ind1==endind1) && (ind2==endind2))
          done = 1;

      }
    }
    __syncthreads();
    v_max[threadIdx.x] = vs_max[threadIdx.x];
    s_max[threadIdx.x] = ss_max[threadIdx.x];
    b_max[threadIdx.x] = bs_max[threadIdx.x];
    __syncthreads();
    i *= 2;
  }

  ss_max[threadIdx.x] = s_max[threadIdx.x];
  bs_max[threadIdx.x] = b_max[threadIdx.x];
  vs_max[threadIdx.x] = v_max[threadIdx.x];

  }

  __syncthreads();

  if(threadIdx.x == 0 && threadIdx.y == 0){
    if(bs_max[0] < 0){
      len[0] = 0;
      return;
    }
    int lng = 1;

    for(int k=1; k<blockDim.x; k++){
      if( vs_max[k] <= (int)(cfgPar.steps.threshold.xyz*10000.) || lng == cfgPar.det.number_of_maxima )
        break;
      if( bs_max[k] < 0)
        continue;
      int df = ss_max[k] - s_max[lng-1];
      df *= df < 0 ? -1 : 1;
      if(df == 1 || df == blockDim.x-1)
	continue;
      lng++;
      if(k == lng-1)
	continue;
      s_max[lng-1] = ss_max[k];
      b_max[lng-1] = bs_max[k];
      v_max[lng-1] = vs_max[k];
    }
    len[0] = lng;
    __syncthreads();

  }

}


__global__ void computeOutput(int * a_data, INPUT_HT_DATA *mdtData, int * b_max, int * s_max, int * v_max, MUON_HOUGH_RED_PATTERN *pOut, int level, int max, int * nxt)
{

  __shared__ int lvl;
  __shared__ int npat;
  __shared__ int size;
  __shared__ double theCoords[2];
  __shared__ int theSec;
  __shared__ int oVars[4];
  __shared__ double theVars[5];
  if(threadIdx.x == 0){
    if(level == 0)
      lvl = level;
    else
      lvl = 1;
    npat = pOut->m_nPatterns;
    size = mdtData->m_nPattXY;
    int NA = 2*(int)(cfgPar.steps.ip.xy/(cfgPar.steps.stepsize.xy+0.));
    int theBin = b_max[max];

    theCoords[0] = __fma_rn(cfgPar.steps.stepsize.xy, ((theBin%NA)-0.5), -cfgPar.steps.ip.xy);
    theCoords[1] = __fma_rn(__fma_rn(cfgPar.steps.stepsize_angle.xyz, ((theBin/(NA+0.))-0.5), 0.), 0.0174533, 0);
    while(theCoords[1] > M_PI)
      theCoords[1] -= 2*M_PI;
    while(theCoords[1] < -M_PI)
      theCoords[1] += 2*M_PI;

    theSec = s_max[max];

    for(int v=0;v<4;v++)
      oVars[v] = 0;
    pOut->m_nHits[lvl * npat + max] = 0;
  }

  __syncthreads();

#pragma unroll
  for(int i=threadIdx.x; i<size; i += blockDim.x){
    double x = mdtData->m_hitx[a_data[i]], y = mdtData->m_hity[a_data[i]];
    double trig[2] = {sin(theCoords[1]), cos(theCoords[1])};
    double radius = __dsqrt_rn(__fma_rn(x, x, __fma_rn(y, y, 0)));
    double theta = atan2(radius, mdtData->m_hitz[a_data[i]]);
    int hSec = (int)(theta*cfgPar.steps.sectors.xyz*M_1_PI);
    int hSec_m = hSec > 0 ? hSec - 1 : cfgPar.steps.sectors.xyz - 1;
    int hSec_p = hSec < cfgPar.steps.sectors.xyz - 1 ? hSec + 1 : 0;
    double dotprod = __fma_rn(y, trig[0], __fma_rn(x, trig[1], 0));
    double scphimax = __fma_rn(x, trig[0], - __fma_rn(y, trig[1], 0));
    double resid = __dadd_rn(scphimax, -theCoords[0]);
    bool ass = mdtData->m_ass[a_data[i]];
    int c1 = (int)((theSec == hSec) || (theSec == hSec_m) || (theSec == hSec_p));
    int c2 = (int)(dotprod >= 0.);
    int c3 = (int)(fabs(resid) < cfgPar.algo.maximum_residu_mm);
    int c4 = (int)(!ass);
    int csqn = c1 + c2 + c3 + c4;
    if(csqn < 4)
      continue;
    mdtData->m_ass[a_data[i]] = csqn == 4 ? true : ass;
    int hidx = atomicAdd(&(pOut->m_nHits[lvl * npat + max]), csqn == 4 ? 1 : 0);
    if(csqn == 4)
      pOut->m_hitIdx[lvl * npat + max][hidx] = a_data[i];
    atomicAdd(&pOut->m_nTotHits, csqn == 4 ? 1 : 0);
    double phi = calcAngle(x, y, theCoords[0]);
    atomicAdd(oVars, (csqn == 4 ? (int)(scphimax*100000) : 0));
    atomicAdd(oVars+1, (csqn == 4 ? (int)(sin(phi)*100000) : 0));
    atomicAdd(oVars+2, (csqn == 4 ? (int)(cos(phi)*100000) : 0));
    atomicAdd(oVars+3, (csqn == 4 ? (int)(theta*100000) : 0));
  }

  __syncthreads();

  if(threadIdx.x == 0){
    int pat_hits = pOut->m_nHits[lvl * npat + max];
    pOut->m_algo[lvl * npat + max] = 0;//to be changed
    pOut->m_level[lvl * npat + max] = level;
    if(pat_hits > 0){
      theVars[0] = atan2((double)oVars[1]*0.00001, (double)oVars[2]*0.00001);
      theVars[1] = ((double)oVars[0]*0.00001)/pat_hits;
      theVars[4] = ((double)oVars[3]*0.00001)/pat_hits;

      pOut->m_ephi[lvl * npat + max] = pat_hits > 0 && fabs(theVars[0]-theCoords[1]) > 0.05 ? theCoords[1] : theVars[0];
      pOut->m_erphi[lvl * npat + max] = theCoords[0];
      pOut->m_etheta[lvl * npat + max] = theVars[4];
      pOut->m_maximumhistogram[lvl * npat + max] = v_max[max]*0.0001;
      atomicAdd(&pOut->m_nPatterns, 1);
    }
    int c1 = (int)(pat_hits >= cfgPar.algo.thresholdpattern_xyz);
    int c2 = (int)(cfgPar.algo.maximum_level-1 > lvl);
    int c3 = (int)(size - pOut->m_nTotHits == 0);
    int csqn = c1 + c2 + c3;
    nxt[0] = csqn == 3 ? 0 : 1;
   }

}

////////////////////////////////////// WRAPPER KERNEL //////////////////////////////////////

//__global__ void houghAlgo(int * devProps, int * controls, int * vot_xy_hits, int * ass_xy_hits, int * vot_cc_hits, //int * ass_cc_hits, INPUT_HT_DATA *mdtData, int * b_xy_maxes, int * v_xy_maxes, int * b_cc_maxes, int * v_cc_maxes, //int * s_xy_max, int * b_xy_max, int * v_xy_max, int * s_cc_max, int * b_cc_max, int * v_cc_max, MUON_HOUGH_RED_PATT//ERN *pOut)

__global__ void houghAlgo(int * devProps, int * controls, int * vot_xy_hits, int * ass_xy_hits, int * vot_cc_hits, int * ass_cc_hits, INPUT_HT_DATA *mdtData, int * b_cc_maxes, int * v_cc_maxes, int * s_cc_max, int * b_cc_max, int * v_cc_max, MUON_HOUGH_RED_PATTERN *pOut, int * monitor)
{

  int warps = devProps[0];
  int maxTrds = devProps[1];
  int *d_len = controls;
  int *lvl = controls+1;
  int *d_max = controls+2;
  int *nxt = controls+3;
  nxt[0] = 1;
  //0:xy 1:cc
  int Nsec[2] = {cfgPar.steps.sectors.xyz, cfgPar.steps.sectors.rz};
  int NA[2] = {(int)(2*cfgPar.steps.ip.xy/cfgPar.steps.stepsize.xy), (int)cfgPar.steps.nbins_curved};
  int NB[2] = {(int)(cfgPar.steps.angle.xyz/cfgPar.steps.stepsize_angle.xyz),
      	       (int)((cfgPar.steps.angle.rz/cfgPar.steps.stepsize_angle.rz))};
  dim3 xyGrid(NA[0], Nsec[0]);
  dim3 ccGrid(NA[1], Nsec[1]);
  dim3 sortBlock(Nsec[0], NA[0]);

  const int n_streams = 2;
  cudaStream_t streams[n_streams];

  for(int lev=0; lev < cfgPar.algo.maximum_level; lev++){

    siftHit<<< 1, maxTrds >>>(mdtData, vot_xy_hits, ass_xy_hits, vot_cc_hits, ass_cc_hits);
    cudaDeviceSynchronize();
    //__syncthreads();
    
    if((mdtData->m_nVoteXY == 0 || mdtData->m_nPattXY == 0) && (mdtData->m_nVoteRZ == 0 || mdtData->m_nPattRZ == 0))
      return;
  
    //voteHoughSpace<<< xyGrid, (int)(maxTrds*0.5), sizeof(int)*(1+NB[0]+maxTrds) >>>(vot_xy_hits, mdtData, b_xy_maxes, v_xy_maxes);
    houghCCSpace<<< ccGrid, (int)(maxTrds*0.5), sizeof(int)*(2+NB[1]+maxTrds) >>>(vot_cc_hits, mdtData, b_cc_maxes, v_cc_maxes, monitor);
    cudaDeviceSynchronize();
    //__syncthreads();
/*
    sorter<<< 1, sortBlock, (1+3*Nsec[0]+2*NA[0]*Nsec[0])*sizeof(int) >>>(b_xy_maxes, v_xy_maxes, s_xy_max, b_xy_max, v_xy_max, d_len);
    cudaDeviceSynchronize();
    //__syncthreads();

    int length = d_len[0];
    if(length == 0)
      break;

    lvl[0]=lev;

    int outThreads = mdtData->m_nPattXY < maxTrds ?
                     warps*((int)ceil(mdtData->m_nPattXY/(warps+0.))) : maxTrds;
    outThreads += outThreads == 0 ? 1 : 0;
    for(int max=0; max < length; max++){
      d_max[0] = max;
      computeOutput<<< 1, outThreads >>>(ass_xy_hits, mdtData, b_xy_max, s_xy_max, v_xy_max, pOut, lvl[0], d_max[0], nxt);
      cudaDeviceSynchronize();
    }
    //__syncthreads();

    if(nxt[0] == 0)
      break;
*/
  }

}
    /*/double theta[2] = {csqn ? hitth + rsin : -0.1, csqn ? hitth - rsin : -0.1 };
    double thetaSides[4] = {csqnSides[0] ? hitth + rsinSides[0] : -0.1, csqnSides[0] ? hitth - rsinSides[0] : -0.1,
         		    csqnSides[1] ? hitth + rsinSides[1] : -0.1, csqnSides[1] ? hitth - rsinSides[1] : -0.1};
    double crv_wg[2] = {csqn ? (0.5+0.5*sin(theta[0])) * wgh / (1.+sz_wght*rsin) : 0.,
    	   	      	csqn ? (0.5+0.5*sin(theta[1])) * wgh / (1.+sz_wght*rsin) : 0.};
    double crv_wgSides[4] = {csqnSides[0] ? (0.5+0.5*sin(thetaSides[0])) * wgh / (1.+sz_wght*rsinSides[0]) : 0.,
    	   		     csqnSides[0] ? (0.5+0.5*sin(thetaSides[1])) * wgh / (1.+sz_wght*rsinSides[0]) : 0.,
    	   		     csqnSides[1] ? (0.5+0.5*sin(thetaSides[2])) * wgh / (1.+sz_wght*rsinSides[1]) : 0.,
    	   		     csqnSides[1] ? (0.5+0.5*sin(thetaSides[3])) * wgh / (1.+sz_wght*rsinSides[1]) : 0.};
    bool csqn_th[2] = {theta[0] > 0. && theta[0] < M_PI, theta[1] > 0. && theta[1] < M_PI};
    bool csqn_thSides[4] = {thetaSides[0] > 0. && thetaSides[0] < M_PI, thetaSides[1] > 0. && thetaSides[1] < M_PI,
 	 		    thetaSides[2] > 0. && thetaSides[2] < M_PI, thetaSides[3] > 0. && thetaSides[3] < M_PI};
    int aBin[2] = {(int)(blockIdx.x+0.5+Amax), (int)(-blockIdx.x-0.5+Amax)};
    int tBin[2] = {(int)((theta[0]*57.29578)/Bwdt), (int)((theta[1]*57.29578)/Bwdt)};
    int tBinSides[4] = {(int)((thetaSides[0]*57.29578)/Bwdt), (int)((thetaSides[1]*57.29578)/Bwdt),
    		        (int)((thetaSides[2]*57.29578)/Bwdt), (int)((thetaSides[3]*57.29578)/Bwdt)};*/
