#include "../include/HoughTransform.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <set>


void HoughTransform::makeHTPatterns(APE::HoughTransformContext* context, INPUT_MDT_DATA* mdtHitArray, OutputDataContainer* outPatterns){

  // Initialize everything
  m_context = context;
  m_mdtHitArray = mdtHitArray;
  m_outputContainer = outPatterns;

  //debug configuration
  bool verbose = false;


  //To fix soon
  m_use_cosmics = false;
  m_maxNumberOfPhiHits = -1;
  m_weightcutmdt = true;

  // const int Nsec=32; // Numero settori
  // const int Ntheta=18; // Numero settori in piano longitudinale
  const int NA=16; // Numero bin ascissa centro
  const int NB=1440; // Numero bin ordinata centro

  // const int Nrho=100; // Numero bin ascissa centro
  // const int Nphi=360; // Numero bin ordinata centro

  // const float Amin=-600.f; 
  // const float Amax=600.f; 
  const float Amin=-m_context->m_HTConfig.m_detectorsize_xy_full; 
  const float Amax=m_context->m_HTConfig.m_detectorsize_xy_full; 
  const float Bmin=0.f; 
  // const float Bmax=360.f; 
  const float Bmax=m_context->m_HTConfig.m_detectorsize_angle_xyz; 
  // const float rhomin=0.f; 
  //  const float thetamin=0.f; 
  //  const float thetamax=M_PI; 

  //  float m_pt;
  //  float dtheta=M_PI/Ntheta;
  // float dA=(Amax-Amin)/NA;
  // float dB=(Bmax-Bmin)/NB;
  // float rhomax=sqrt(Amax*Amax+Bmax*Bmax);
  // float rhobin=(rhomax-rhomin)/Nrho;
  // float phibin=2*M_PI/Nphi;


  int number_of_sectors=m_context->m_HTConfig.m_number_of_sectors_xyz; // 12 for MuonHoughTransformer_xyz, 32 for MuonHoughTransformer_CurvedAtACylinder
  float stepsize=m_context->m_HTConfig.m_stepsize_xy; // 75.0 for MuonHoughTransformer_xyz, 1.0 for MuonHoughTransformer_CurvedAtACylinder
  float stepsize_per_angle=m_context->m_HTConfig.m_stepsize_per_angle_xyz; // 0.250 for MuonHoughTransformer_xyz, 0.500 for MuonHoughTransformer_CurvedAtACylinder
  int number_of_maxima = m_context->m_HTConfig.m_number_of_maxima; // MuonHoughTransformer (Base class): getMaxima with max_patterns=5
  unsigned int threshold = m_context->m_HTConfig.m_thresholdhisto_xyz;//50; //mn Nel codice standard sono usati 2 valori: 9000 o 21000... DA CAPIRE!
  bool ip_setting = true; //missing, need to add
  double max_residu_mm = m_context->m_HTConfig.m_maximum_residu_mm;
  float scale = 1; // MuonHoughPatternEvent/MuonHoughHisto2D.h  CHECK!
  int nbins_plus3 = NA; // CHECK!
  bool use_negative_weights = false; // CHECK!
  double detectorsize_angle = m_context->m_HTConfig.m_detectorsize_angle_xyz;
  
  double binwidthx=(Amax-Amin)/NA;
  double binwidthy=(Bmax-Bmin)/NB;
  
  std::cout << "------------- Algorithm configuration " << std::endl;
  std::cout << "number_of_sectors: " << number_of_sectors << std::endl;
  std::cout << "stepsize: " << stepsize << std::endl;
  std::cout << "stepsize_per_angle: " << stepsize_per_angle << std::endl;
  std::cout << "max_residu_mm: " << max_residu_mm << std::endl;
  std::cout << "threshold(histo): " << threshold << std::endl;


  std::cout << "m_context->m_HTConfig.m_number_of_sectors_xyz = " << m_context->m_HTConfig.m_number_of_sectors_xyz << std::endl;
//   std::cout << "m_context->m_HTConfig.m_stepsize_xy = " << m_context->m_HTConfig.m_stepsize_xy << std::endl;
//   std::cout << "m_context->m_HTConfig.m_stepsize_per_angle_xyz = " << m_context->m_HTConfig.m_stepsize_per_angle_xyz << std::endl;
//   std::cout << "m_context->m_HTConfig.m_number_of_maxima = " << m_context->m_HTConfig.m_number_of_maxima << std::endl;
  std::cout << "m_context->m_HTConfig.m_thresholdhisto_xyz = " << m_context->m_HTConfig.m_thresholdhisto_xyz << std::endl;
//   std::cout << "m_context->m_HTConfig.m_use_ip = " << m_context->m_HTConfig.m_use_ip << std::endl;
  std::cout << "m_context->m_HTConfig.m_maximum_residu_mm = " << m_context->m_HTConfig.m_maximum_residu_mm << std::endl;
//   std::cout << "m_context->m_HTConfig.m_detectorsize_angle_xyz = " << m_context->m_HTConfig.m_detectorsize_angle_xyz << std::endl;
  
  
  
  float acc_Mat[NA][NB][Nsec] = {0.};
  
  std::vector<float> x_values;
  std::vector<float> y_values;
  std::vector<float> z_values;
  std::vector<float> r_values;
  std::vector<float> sector_values;



  
//  std::cout<<"Setup of muon HoughTransform with  NA="<< NA  << " NB=" << NB << " Amin=" << Amin << " Amax=" << Amax << " Bmin=" << Bmin << " Bmax=" << Bmax << " number_of_sectors=" << number_of_sectors <<std::endl;

  
//  tbb::tick_count tstart=tbb::tick_count::now();

  std::cout<<"running the job..." << std::endl;

  int nWords=m_mdtHitArray->m_nDataWords;
  std::cout<<"In Work: run nWords = "<<nWords<<std::endl;

  ///// Added from MB  
  
  // /** empty and reinitialize the houghpattern vectors */
  m_houghpattern=HoughTransform::emptyHoughPattern();
  reset();
  init();
  
  /** skip cosmic events that have more than 1000 phi hits */
  if (m_use_cosmics == true && m_maxNumberOfPhiHits >= 0 ) {
    int phihits = 0;
    for (unsigned int hitid=0; hitid<mdtHitArray->m_nDataWords; hitid++) {
      if (mdtHitArray->m_hit[hitid].m_measures_phi==1) {
        phihits++;
      }
    }
    if (phihits > m_maxNumberOfPhiHits ) {
      std::cout << "Cosmic event has more than 1000 phi hits: " << phihits << " event is not reconstructed!" << std::endl;
      return;
    }
  }
  
  if (m_weightcutmdt == true) {setWeightMdtCutValue();}
  
  std::cout << "Mdt Cut Value: " << m_weightmdt << std::endl;  // ATH_MSG_DEBUG("Mdt Cut Value: " << m_weightmdt);
  
  // reset weights, based on rejection factor and m_weightmdt
  calculateWeights();
  
  if( verbose ) {
    std::cout << "Event Info" << std::endl;
    std::cout << "Size: " << m_mdtHitArray->m_nDataWords << std::endl;
    
    for (unsigned int i=0; i<m_mdtHitArray->m_nDataWords; i++) {
      std::cout << m_mdtHitArray->m_hit[i].m_hitx << " "
		<< m_mdtHitArray->m_hit[i].m_hity << " "
		<< m_mdtHitArray->m_hit[i].m_hitz << " "
		<< m_mdtHitArray->m_hit[i].m_measures_phi << " "
		<< m_mdtHitArray->m_hit[i].m_detector_id << " "
		<< m_mdtHitArray->m_hit[i].m_probability << " "
		<< m_mdtHitArray->m_hit[i].m_weight << " "
		<< m_mdtHitArray->m_hit[i].m_associated << " "
		<< std::endl;
    }
  }
  //////
  
  
  //mn OLD Hough code starts here...
  for(int i=0;i<nWords;i++) {
    MuonHoughHitFlat mdtHit = m_mdtHitArray->m_hit[i];
    //    std::cout << "matteo -- MDT hit " << i << "    :  x=" << mdtHit.m_hitx << " y=" << mdtHit.m_hity << " z=" << mdtHit.m_hitz << " radius=" << mdtHit.m_radius << " sector=" << sector(mdtHit.m_radius,mdtHit.m_hitz,number_of_sectors) << std::endl;
    x_values.push_back(mdtHit.m_hitx);
    y_values.push_back(mdtHit.m_hity);
    z_values.push_back(mdtHit.m_hitz);
    r_values.push_back(mdtHit.m_radius);
    sector_values.push_back(sector(mdtHit.m_radius, mdtHit.m_hitz, number_of_sectors));
  }
  
  //   MuonHoughTransformer_xyz::fillHit  and  MuonHoughTransformer_xyz::fillHisto
  for(unsigned int i = 0; i < x_values.size(); i++){
  
    float radius=r_values.at(i);
    float hitx=x_values.at(i);
    float hity=y_values.at(i);
    int sectorhit = sector_values.at(i);
    double dotprod = 0.;
    float weight = m_mdtHitArray->m_hit[i].m_weight;
    //    std::cout << "DEBUG - weight (from hit): " << weight << std::endl;    
    std::pair <double,double> endpoints = getEndPointsFillLoop(radius,stepsize,sectorhit, Amin, Amax);
    
    for (double r0 = endpoints.first; r0<endpoints.second; r0+=stepsize) {
      
      double phi = calculateAngle(hitx, hity, r0);
      //      std::cout << " hitx: " << hitx << " hity: " << hity << " r0: " << r0 << " phi: " << phi << std::endl; 
      //      CxxUtils::sincos scphi(phi);
      //      dotprod = scphi.apply(hity,hitx); // hity * sincosphi[0] + hitx * sincosphi[1];
      dotprod = hity * sin(phi) + hitx * cos(phi); // hity * sincosphi[0] + hitx * sincosphi[1];
      if (dotprod >=0) {
        double phi_in_grad = phi*57.29578; //MuonHough::rad_degree_conversion_factor;
        //		double ip_weight = weight*weightHoughTransform(r0); // give mildly more importance to patterns closer to IP
        //mn: replace with matrix fillHisto(r0,phi_in_grad,weight,sectorhit);
        acc_Mat[coordToBin(r0, Amin, Amax, NA)][coordToBin(phi_in_grad, Bmin, Bmax, NB)][sectorhit]+=scale*weight; 
      }
    }
    
    double phi_in_grad = calculateAngle(hitx, hity, radius)*57.29578;
    int binhit = pairToBin(coordToBin(radius, Amin, Amax, NA), coordToBin(phi_in_grad, Bmin, Bmax, NB), NA, NB);
    int binhitx = binToPair(binhit, NA, NB).first;
    int binhity = binToPair(binhit, NA, NB).second;
    
    ///////////////////////////////////////////////////////
    ///MN - Placeholder
    ///   after this fills with different weights the neighboring bins
    ///   see:   MuonHoughTransformer_xyz::fillHisto
    /// ATTENZIONE: QUESTO E' TUTTO DA RIVEDERE!!!
    ///////////////////////////////////////////////////////
    // applying a 'butterfly' weighting effect:
    bool butterfly = true;
    if (butterfly == true)
      {
	// nearby sectors:
	if (number_of_sectors>=3)
	  {
	    double third_weight = weight/3.;
	    if (sectorhit != 0 && sectorhit != number_of_sectors - 1)
	      {
		acc_Mat[binhitx][binhity][sectorhit+1]+=scale*third_weight;
		acc_Mat[binhitx][binhity][sectorhit-1]+=scale*third_weight;
		//	      m_histos.getHisto(sector+1)->fill(filled_binnumber,third_weight);
		//	      m_histos.getHisto(sector-1)->fill(filled_binnumber,third_weight);
	      }
	    else if (sectorhit == 0)
	      {
		acc_Mat[binhitx][binhity][sectorhit+1]+=scale*third_weight;
		//	      m_histos.getHisto(sector+1)->fill(filled_binnumber,third_weight);
		////	      m_histos.getHisto(m_number_of_sectors-1)->fill(filled_binnumber,weight/6.);
	      }
	    else // sector == m_number_of_sectors - 1
	      {
		acc_Mat[binhitx][binhity][sectorhit-1]+=scale*third_weight;
		//	      m_histos.getHisto(sector-1)->fill(filled_binnumber,third_weight);
		////	      m_histos.getHisto(0)->fill(filled_binnumber,weight/6.);
	      }
	  }
	
	double half_weight = 0.5*weight;
	
	binhitx = binToPair(binhit-1, NA, NB).first;
	binhity = binToPair(binhit-1, NA, NB).second;
	acc_Mat[binhitx][binhity][sectorhit]+=scale*half_weight;
	//histo->fill(filled_binnumber-1, half_weight);
	binhitx = binToPair(binhit+1, NA, NB).first;
	binhity = binToPair(binhit+1, NA, NB).second;
	acc_Mat[binhitx][binhity][sectorhit]+=scale*half_weight;
	//histo->fill(filled_binnumber+1, half_weight);
	
	const int upperright = binhit + nbins_plus3;
	const int lowerleft = binhit - nbins_plus3;
	
	
	if (phi_in_grad - binwidthy < 0)
	  {
	    binhit = pairToBin(coordToBin(radius-binwidthx, Amin, Amax, NA), coordToBin(phi_in_grad, Bmin, Bmax, NB)-binwidthy+detectorsize_angle, NA, NB);
	    binhitx = binToPair(binhit, NA, NB).first;
	    binhity = binToPair(binhit, NA, NB).second;
	    acc_Mat[binhitx][binhity][sectorhit]+=scale*half_weight;
	    ////histo->fill(r0-m_binwidthx, phi-m_binwidthy + m_detectorsize_angle, half_weight); // should calculate binnumber..
	    binhitx = binToPair(upperright, NA, NB).first;
	    binhity = binToPair(upperright, NA, NB).second;
	    acc_Mat[binhitx][binhity][sectorhit]+=scale*half_weight;
	    ////histo->fill(upperright, half_weight);
	    // 	  if (use_negative_weights)
	    // 	    {	  
	    // 	      histo->fill(r0+m_binwidthx, phi-m_binwidthy + m_detectorsize_angle, -half_weight);
	    // 	      histo->fill(upperright-2, -half_weight);
	    // 	    }
	  }
	else if (phi_in_grad + binwidthy > detectorsize_angle)
	  {
	    binhitx = binToPair(lowerleft, NA, NB).first;
	    binhity = binToPair(lowerleft, NA, NB).second;
	    acc_Mat[binhitx][binhity][sectorhit]+=scale*half_weight;
	    ////histo->fill(lowerleft, half_weight);
	    binhit = pairToBin(coordToBin(radius+binwidthx, Amin, Amax, NA), coordToBin(phi_in_grad, Bmin, Bmax, NB)+binwidthy-detectorsize_angle, NA, NB);
	    binhitx = binToPair(binhit, NA, NB).first;
	    binhity = binToPair(binhit, NA, NB).second;
	    acc_Mat[binhitx][binhity][sectorhit]+=scale*half_weight;
	    ////histo->fill(r0+m_binwidthx, phi+m_binwidthy -m_detectorsize_angle, half_weight);
	    // 	  if (use_negative_weights)
	    // 	    {
	    // 	      histo->fill(lowerleft+2, -half_weight);
	    // 	      histo->fill(r0-m_binwidthx, phi+m_binwidthy -m_detectorsize_angle, -half_weight);
	    // 	    }
	  }
	else 
	  {
	    binhitx = binToPair(lowerleft, NA, NB).first;
	    binhity = binToPair(lowerleft, NA, NB).second;
	    acc_Mat[binhitx][binhity][sectorhit]+=scale*half_weight;
	    ////histo->fill(lowerleft, half_weight);
	    binhitx = binToPair(upperright, NA, NB).first;
	    binhity = binToPair(upperright, NA, NB).second;
	    acc_Mat[binhitx][binhity][sectorhit]+=scale*half_weight;
	    ////histo->fill(upperright, half_weight);
	    // 	  if (use_negative_weights)
	    // 	    {
	    // 	      histo->fill(lowerleft+2, -half_weight);
	    // 	      histo->fill(upperright-2, -half_weight);
	    // 	    }
	  }
      }
    
    ///////////////////////////////////////////////////////
    /// ATTENZIONE: RIVEDERE FINO A QUI!!!
    ///////////////////////////////////////////////////////
    
    
    
  }
  
  
  //    std::cout << "matteo -- Now looking for the maxima" << std::endl;
  
  std::vector <std::pair < std::pair <int,int> , double > > maxima;
  
  for (int sector=0; sector<number_of_sectors; sector++) // to be made more general when m_number_of_sectors ==1 e.g.
    {
      // pair (binnumber, content of that bin)
      //	std::pair <int, double> maximumbin = m_histos.getHisto(sector)->getMaximumBin();
      //std::pair <int,double> maximumbin = m_histos.getHisto(sector)->getMax();
      //mn here replace the getMax() method on the 2D histos
      int maxbin=-1;
      unsigned int maxval = threshold;
      for(unsigned int iA = 0; iA < NA; iA++){
	for(unsigned int iB = 0; iB < NB; iB++){
	  if (acc_Mat[iA][iB][sector] > maxval) {
	    maxbin=pairToBin(iA, iB, NA, NB);
	    maxval = acc_Mat[iA][iB][sector];
	  }
	}
      }
      //std::cout << "DEBUG - maxval=" << maxval << std::endl;
      std::pair < std::pair <int,int> , double > maximum;
      maximum.first.first = sector;
      maximum.first.second = maxbin;
      maximum.second = maxval;
      if (maxval>threshold) maxima.push_back(maximum);
    }
  
  std::sort(maxima.begin(),maxima.end(),maximaCompare());
  
  // Here we end up with the full list of maxima
  // then we choose among them
  std::cout << "DEBUG - maxima found: " << maxima.size() << std::endl;
  
  
  
  if (maxima.size()>0) { // maxima
    
    
    std::vector <std::pair <int,int> > maximumbins; // sorted - used in the subsequent processing
    
    unsigned int count_maxima = 0; // should be iterator
    int number_of_patterns = 0;
    std::set<int> sectors; // sectors that are already used
    const unsigned int size = maxima.size();
    while (count_maxima!=size && number_of_patterns!=number_of_maxima)
      {
  	std::pair <int,int> maximumbin = maxima[count_maxima].first; 
  	//	if (sectorNotInMaximumBinsYet(maximumbins,maximumbin.first))
	
  	bool check = true; // check if sector is not nearby a sector already chosen
  	int sector = maximumbin.first;
	
  	if (sectors.find(sector) != sectors.end()) {check = false;}
	
  	if (check == true)
  	  {
  	    maximumbins.push_back(maximumbin);
  	    sectors.insert(maximumbin.first);
	    
  	    int sectormin = sector-1;
  	    int sectorplus = sector+1;
  	    if (sectormin < 0) {sectormin = number_of_sectors;}
  	    if (sectorplus > number_of_sectors) {sectorplus=0;}
	    
  	    sectors.insert(sectormin);
  	    sectors.insert(sectorplus);
	    
  	    if (number_of_sectors > 20  && maximumbin.first%2 == 1) // hack for new single and overlap filling curved transform!
  	      {
  		int sectorminmin = sectormin - 1;
  		int sectorplusplus = sectorplus + 1;
		if (sectorminmin < 0) {sectorminmin = number_of_sectors;} // not necessary since those should be odd
		if (sectorplusplus > number_of_sectors) {sectorplusplus=0;}
  		sectors.insert(sectorminmin);
  		sectors.insert(sectorplusplus);
  	      }
	    
  	    count_maxima++;
  	    number_of_patterns++;
  	  }
  	else 
  	  {
  	    count_maxima++;
  	  }
      }
    std::cout << "DEBUG - maxima bin kept: " << maximumbins.size() << std::endl;

    //mn  maximumbins is returned from the original getMaxima(max_patterns) -> corresponds to "maxima" in MuonHoughTransformSteering.cxx
    //    ->  std::vector <std::pair <int, int> > maxima = m_houghtransformer->getMaxima(max_patterns); // sector,binnumber , sorted vector
    
    // Now should translate hookAssociateHitsToMaximum
    // to produces the houghpattern

    //MBDEVEL -- loop over maxima should start here...
    for(unsigned int cnt=0; cnt<maximumbins.size(); cnt++) {
      std::cout << "copying maximum n " << cnt << std::endl;
      int max_sector=maximumbins.at(cnt).first;
      std::pair <double,double> coordsmaximum;
      coordsmaximum.first = 99999.;  // initialization in case no maximum is found
      coordsmaximum.second = 99999.;  // initialization in case no maximum is found
      if (maximumbins.size()>0) {//shoudl be not needed - MB
	coordsmaximum.first = binnumberToCoord( binToPair(maximumbins.at(cnt).second, NA, NB).first, Amin, Amax, NA); // binx
	coordsmaximum.second = binnumberToCoord( binToPair(maximumbins.at(cnt).second, NA, NB).second, Bmin, Bmax, NB); // biny
      }
    
      double phimax = coordsmaximum.second*0.0174533; // phimax in rad
      int max_secmax = max_sector + 1;
      int max_secmin = max_sector - 1;
      if (max_secmin < 0) max_secmin = max_sector;
      if (max_secmax > number_of_sectors -1 ) max_secmax = max_sector;
      
      
      
      
      
      double ephi=0., eradius=0., sin_phi=0., cos_phi=0.;
      double dotprod=0.;
      double etheta=0.;
      double residu_distance=0.;
      int nhoughhits=0;
      for (unsigned int i=0; i<x_values.size(); i++)
	{
	  double hitx = x_values.at(i);
	  double hity = y_values.at(i);
	  double hitz = z_values.at(i);
	  
	  double radiushit = std::sqrt(hitx*hitx + hity*hity);
	  int sectorhit = sector_values.at(i);
	  
	  if (sectorhit == max_sector||sectorhit == max_secmin||sectorhit == max_secmax)
	    {
	      if (ip_setting == false){dotprod = 1.;}
	      else {
		////dotprod = scphimax.apply(getHitPos(event,i).second, getHitPos(event,i).first); //getHitPos(event,i).first * sincosphimax[1] + getHitPos(event,i).second * sincosphimax[0];
		dotprod = hitx * cos(phimax) + hity * sin(phimax);
	      }
	      if (dotprod>=0)
		{
		  //   residu_distance = m_muonhoughmathutils.distanceToLine(getHitPos(event,i).first,getHitPos(event,i).second,coordsmaximum_inglobalcoords.first,coordsmaximum_inglobalcoords.second);
		  //   residu_distance = - coordsmaximum.first + std::sin(phimax) * event->getHitx(i) - std::cos(phimax) * event->getHity(i); // phimax shoudl be in map, but there are rounding errors..
		  //residu_distance = - coordsmaximum.first + scphimax.apply(getHitPos(event,i).first,-getHitPos(event,i).second); //- coordsmaximum.first + sincosphimax[0] * getHitPos(event,i).first - sincosphimax[1] * getHitPos(event,i).second; // phimax shoudl be in map, but there are rounding errors..
		  residu_distance = - coordsmaximum.first + hitx * sin(phimax) - hity * cos(phimax); //- coordsmaximum.first + sincosphimax[0] * getHitPos(event,i).first - sincosphimax[1] * getHitPos(event,i).second; // phimax shoudl be in map, but there are rounding errors..	      
		  if(std::abs(residu_distance)<max_residu_mm) 
		    {
		      nhoughhits++;
		      // 		  houghpattern->addHit(event->getHit(i));
		      // 
		      // 		  event->getHit(i)->setAssociated(true);
		      // 
		      double phi = calculateAngle(hitx,hity,coordsmaximum.first);
		      
		      double thetah = atan2(radiushit,hitz);		  
		      
		      etheta  +=thetah; 
		      sin_phi += sin(phi);
		      cos_phi += cos(phi);
		      
		      double radius = signedDistanceOfLineToOrigin2D(hitx,hity,phimax);
		      eradius +=radius;
		      
		    }
		} //dotprod >=0
	    } // sector requirement
	} // size
      
      // eradius=eradius/(houghpattern->size()+0.0001);
      // etheta=etheta/(houghpattern->size()+0.0001);
      eradius=eradius/(nhoughhits+0.0001);
      etheta=etheta/(nhoughhits+0.0001);
      ephi = std::atan2(sin_phi,cos_phi);  
      
      std::cout << "matteo -- eradius=" << eradius << "  etheta=" << etheta << "  ephi=" << ephi << std::endl;
      
      
      // From MuonHoughTransformer_xyz::hookAssociateHitsToMaximum
      APE_MUON_HOUGH_PATTERN outhoughpattern;
      outhoughpattern.m_id_number = hough_xy;
      outhoughpattern.m_whichsegment = false;
      outhoughpattern.m_ephi = ephi;
      outhoughpattern.m_erphi = coordsmaximum.first;
      outhoughpattern.m_etheta = etheta;
      outhoughpattern.m_ertheta = 0;
      outhoughpattern.m_ecurvature = 0;
      outhoughpattern.m_maximumhistogram = 0;
      std::cout << "Test in HT.cxx: id_number, etheta: " << outhoughpattern.m_id_number << " " << outhoughpattern.m_etheta << std::endl;
      
      //    int cnt = 0;
      m_outputContainer->patterns[cnt] = outhoughpattern;
      
    }
    //MBDEVEL -- loop over maxima should end here...    
    m_outputContainer->m_N_maxima_found = maximumbins.size();

  } // maxima

// IN TrigMuonAccelerationEDM.h è definita la struttura APEMuonHoughPattern, da usare per l'output
      
//mn --- end  
    
//  SIMPLE_OUTPUT_DATA *pOutput = static_cast<SIMPLE_OUTPUT_DATA*>(m_buffer->getBuffer());
//  pOutput->m_nDataWords = nWords;
//  std::cout<<" processed "<<nWords<<" data words"<<std::endl;

//  duration=tbb::tick_count::now()-tstart;
//  m_stats.push_back(duration.seconds()*1000.0);

  std::cout << "That's all folks!" << std::endl;
}


void HoughTransform::reset()
{
  // ATH_MSG_VERBOSE("reset()");
  if (HoughTransform::m_houghpattern.size()!=0)
    {
      for (unsigned int i=0; i<HoughTransform::m_houghpattern.size(); i++)
	{
	  for (unsigned int j=0; j<HoughTransform::m_houghpattern[i].size(); j++)
	    {
	      for (unsigned int k=0; k<HoughTransform::m_houghpattern[i][j].size(); k++)
		{
		  delete HoughTransform::m_houghpattern[i][j][k];
		  HoughTransform::m_houghpattern[i][j][k]=0;
		}
	    }
	}
      // m_houghpattern.clear();
    }
}

void HoughTransform::init()
{
  // ATH_MSG_VERBOSE("init()");

  m_npatterns= std::vector<int> (m_context->m_HTConfig.m_number_of_ids);

}

APEMuonHoughPatternContainerShip HoughTransform::emptyHoughPattern()
{
  //  std::cout << "emptyHoughPattern() (start) "  << std::endl;
  APEMuonHoughPatternContainerShip houghpattern;
  houghpattern.reserve(m_context->m_HTConfig.m_number_of_ids);
  for (int i=0; i<m_context->m_HTConfig.m_number_of_ids; i++)
    {
      APEMuonHoughPatternContainer which_segment_vector;
      which_segment_vector.reserve(m_context->m_HTConfig.m_maximum_level);
      houghpattern.push_back(which_segment_vector);
      for(int lvl=0; lvl<m_context->m_HTConfig.m_maximum_level; lvl++)
        {
          APEMuonHoughPatternCollection level_vector;
          level_vector.reserve(m_context->m_HTConfig.m_number_of_maxima);
          houghpattern[i].push_back(level_vector);
          for (int maximum_number=0; maximum_number<m_context->m_HTConfig.m_number_of_maxima; maximum_number++)
            {
              APEMuonHoughPattern* houghpattern_level = 0;
              houghpattern[i][lvl].push_back(houghpattern_level);
            } // maximum_number
        } // maximum_level
    } // number_of_ids
  return houghpattern;
} //emptyHoughPattern     

void HoughTransform::setWeightMdtCutValue(/*const APEMuonHoughHitContainer* event*/)
{
  if (m_use_cosmics == true) {
    //To fix asap - MB DEVEL
    // m_context->m_HTConfig.m_weightmdt = 0.;
    return;
  }
  int mdthits = getMDThitno(); // slow function!
   m_weightmdt = mdthits > 0 ? 1. - 5./std::sqrt(mdthits) : 0;
}

int HoughTransform::getMDThitno()
{
  int mdthitno=0;
  for (unsigned int i=0; i<m_mdtHitArray->m_nDataWords; i++)
    {
      if (m_mdtHitArray->m_hit[i].m_detector_id==MDT)
	{
	  mdthitno++;
	}
    }
  return mdthitno;
}

void HoughTransform::calculateWeights(/*const APEMuonHoughHitContainer* event*/)
{
  if (m_weightmdt >= 0.5) { // else do nothing (e.g. cosmics case)
    for (unsigned int i=0; i<m_mdtHitArray->m_nDataWords; i++)
      {
	DetectorTechnology technology = m_mdtHitArray->m_hit[i].m_detector_id;
	if (technology == MDT)
          {
            // recalculate weight, especially important for cavern background MDT events
            double p_old = m_mdtHitArray->m_hit[i].m_orig_weight;
            double p_calc = 0.25*p_old*(1.- m_weightmdt);
            double p_new = p_calc/(p_calc + m_weightmdt*(1-p_old));
	    //std::cout << "MDT probability old " << p_old << " Recalculated " << p_new << std::endl;//ATH_MSG_VERBOSE(" MDT probability old " <<  p_old  << " Recalculated " << p_new);
            m_mdtHitArray->m_hit[i].m_weight = p_new;//event->getHit(i)->setWeight(p_new);
          }
      }
  }
}

void HoughTransform::makePatterns(int id_number)
{
  std::cout << "makePatterns" << std::endl;
  
  //  resetAssociation(); // resets association, for hits that are already assigned to pattern in a previous hough
  //  [...]
}

/*
void APEMuonHoughPatternTool::setWeightMdtCutValue(const APEMuonHoughHitContainer* mdtHitArray)
{
  if (m_use_cosmics == true) {
    m_weightmdt = 0.;
    return;
  }
  int mdthits = event->getMDThitno(); // slow function!                                                                                                                                                       
  m_weightmdt = mdthits > 0 ? 1. - 5./std::sqrt(mdthits) : 0;
}
*/
int HoughTransform::sector(double hitradius, double hitz, int number_of_sectors)
{

  //  returns  the sector number of the hit 0..number_of_sectors-1

  // Peter Kluit correction 
  double theta = std::atan2(hitradius,hitz); // radius>0 : theta: [0,Pi]

  int sectorhit = static_cast<int> (theta * number_of_sectors / M_PI);
  if (sectorhit == number_of_sectors) sectorhit += -1; // could happen in rare cases
  return sectorhit; // only valid for xy!! yz to be done (or to be abondoned) 
}

double HoughTransform::calculateAngle(double hitx, double hity, double r0)
{
  double phi=0;
  double heigth_squared = hitx*hitx + hity*hity - r0*r0;
  if (heigth_squared>=0)
    {
      double heigth = std::sqrt(heigth_squared);  
      
      phi = std::atan2(hity,hitx)+std::atan2(r0,heigth); 
    }

  else {
    phi = std::atan2(hity,hitx);
  }

  if (phi < 0)
    {phi += 2.*M_PI;}
  if (phi > 2.*M_PI)
    {phi -= 2.*M_PI;}

  return phi;
}

std::pair <double,double>  HoughTransform::getEndPointsFillLoop(double radius, double stepsize, int sector, float  Amin, float Amax)
{
  std::pair <double,double> endpoints (-radius+0.00001,radius);   // why +/-0.00001?

  if (-radius < Amin) // randomizer to avoid binning effects
    {
      //int floor = (int) std::floor(radius);
      //endpoints.first=m_histos.getHisto(sector)->getXmin() + floor%stepsize; //randomizer, is there a faster way?
      endpoints.first=Amin + 0.5*stepsize; // no randomizer! no radius constraint
    }

  if (radius > Amax)
    {
      endpoints.second = Amax;
    }
  return endpoints;
}

std::pair <int,int>  HoughTransform::binToPair(int ibin, int nx, int ny)
{
  std::pair <int,int>  binxy;
  binxy.first=ibin%nx;
  binxy.second=ibin/nx;
  return binxy;
}

int  HoughTransform::pairToBin(int ix, int iy, int nx, int ny)
{
  return iy*nx+ix;
}

double HoughTransform::signedDistanceOfLineToOrigin2D(double x, double y, double phi)
{
  return x*sin(phi)-y*cos(phi); 
}
