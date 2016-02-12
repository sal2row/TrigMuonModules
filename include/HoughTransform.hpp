#include "../../../DataStructures/TrigMuonDataStructs/TrigMuonDataStructs/TrigMuonAccelerationEDM.h"
#include "muonSimpleWork.hpp"

#include <vector>


namespace HoughTransform {
  void makeHTPatterns(APE::HoughTransformContext* m_context, INPUT_MDT_DATA* mdtHitArray, OutputDataContainer* outPatterns);
  
  
  /** resets members, called once per event */
  void reset();
  /** initiates members, called once per event */
  void init();
  /** clears houghpatterns, called in init() */
  APEMuonHoughPatternContainerShip emptyHoughPattern();
  /** calculates the mdt weight cut value */
  void setWeightMdtCutValue(/*const APEMuonHoughHitContainer* event*/);
  /** number of MDT hits */
  int getMDThitno();
  /** calculates new weights based on rejection factor (1-origweight) and number of hits in event, only done for MDTs */
  void calculateWeights(/*const APEMuonHoughHitContainer* event*/);
  /** method that builds the patterns */
  void makePatterns(int id_number);//was virtual MBDEVEL

  /** number of patterns build per id */
  std::vector <int> m_npatterns; // number_of_ids                                                                                                                                                               /** use cosmic settings (false) */
  bool m_use_cosmics;
  /** weight_cut for mdt hits in hough */
  bool m_weightcutmdt;
  /** value of mdt weight cut, dependent on # hits in event */
  double m_weightmdt;

  /** reconstructed patterns stored per [number_of_ids][level][which_segment] */
  APEMuonHoughPatternContainerShip m_houghpattern;
  /** maximum number of phi hits to do pattern recognition, if small zero no cut is applied */
  int m_maxNumberOfPhiHits;

  enum MuonHoughTransformers {hough_xy,hough_rzcosmics,hough_curved_at_a_cylinder,hough_rz,hough_rz_mdt,hough_rz_rpc,hough_yz};


  APE::HoughTransformContext* m_context;
  InputMdtData* m_mdtHitArray;
  OutputDataContainer* m_outputContainer;


  int sector(double, double, int);
  double calculateAngle(double, double, double);
  std::pair <double,double> getEndPointsFillLoop(double, double, int, float, float);
  int coordToBinx(double);
  int coordToBiny(double);
  std::pair <int,int> binToPair(int, int, int); // bin number, NA, NB
  int pairToBin(int, int, int, int); // bin num x, bin num y, NA, NB
  double binnumberToXCoord(int );
  double binnumberToYCoord(int );
  double signedDistanceOfLineToOrigin2D(double, double, double);
  inline int coordToBin(double x, double min, double max, int N) {return static_cast<int> (x-min)/((max-min)/N);}
  inline double binnumberToCoord(int binnumber, double min, double max, int N) {return min + (max-min)/N * (-0.5 + binnumber);}



  class maximaCompare{
  public:
    bool operator()(std::pair < std::pair <int,int> , double > lhs, std::pair < std::pair <int,int> , double > rhs)const
    {
      return lhs.second > rhs.second;
    }
  };


}

