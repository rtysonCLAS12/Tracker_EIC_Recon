R__LOAD_LIBRARY(libfmt.so)
#include "fmt/core.h"

#include "Math/Vector3D.h"
#include "Math/Vector4D.h"
#include "Math/VectorUtil.h"
#include "TDecompSVD.h"
#include "TMatrixD.h"
//#include <Vector4D.h>

#include "ROOT/RDataFrame.hxx"
#include "ROOT/RVec.hxx"
#include "TCanvas.h"
#include "TChain.h"
#include "TF1.h"
#include "TFile.h"
#include "TH1D.h"
#include "TH1F.h"
#include "TString.h"
#include "TLegend.h"
#include "TMath.h"
#include "TRandom3.h"
#include "TTree.h"
#include "Math/Vector3D.h" 
#include "TLinearFitter.h"

#include <tuple>
#include <vector>
#include <cmath>

#include "DD4hep/Detector.h"
#include "DDRec/CellIDPositionConverter.h"
 
#include "edm4hep/MCParticleCollection.h"
#include "edm4hep/SimTrackerHitCollection.h"
#include "edm4hep/SimCalorimeterHitCollection.h"
using namespace ROOT::Math;

struct track{  
  XYZVector pos;
  XYZVector vec;
  double    chi2;
  ROOT::VecOps::RVec<int> clusters;
};

struct truth{  
  int    id;
  double x;
  double y;
  double z;
};

using RVecT  = ROOT::VecOps::RVec<track>;
using RVecTruth  = ROOT::VecOps::RVec<truth>;
using RVec4  = ROOT::VecOps::RVec<ROOT::Math::PxPyPzMVector>;
using RVecMC = ROOT::VecOps::RVec<edm4hep::MCParticleData>;
using RVecI  = ROOT::VecOps::RVec<int>;
using RVecB  = ROOT::VecOps::RVec<bool>;
using RVecD  = ROOT::VecOps::RVec<double>;
using RVecUL = ROOT::VecOps::RVec<unsigned long>;
  
   int beamID = 4;
//int beamID = 3;
   int simID  = 1;
   //int simID  = 1;   
//-----------------------------------------------------------------------------------------
// Grab Component functor
//-----------------------------------------------------------------------------------------
  struct getSubID{
    getSubID(std::string cname, dd4hep::Detector& det, std::string rname = "TaggerTrackerHits") : componentName(cname), detector(det), readoutName(rname){}
    
    ROOT::VecOps::RVec<int> operator()(const std::vector<edm4hep::SimTrackerHitData>& evt) {
      auto decoder = detector.readout(readoutName).idSpec().decoder();
      auto indexID = decoder->index(componentName);
      ROOT::VecOps::RVec<int> result;
      for(const auto& h: evt) {
	result.push_back(decoder->get(h.cellID,indexID));      
      }
      return result;    
    };
    
    void SetComponent(std::string cname){
      componentName = cname;
    }
    void SetReadout(std::string rname){
      readoutName = rname;
    }

    private: 
    std::string componentName;
    dd4hep::Detector& detector;
    std::string readoutName;
  };

//-----------------------------------------------------------------------------------------
// Grab Particle functor
//-----------------------------------------------------------------------------------------
struct getParticle{
  getParticle(int genStat, int pdg) : generatorStatus(genStat), PDG(pdg){}
  
  ROOT::VecOps::RVec<bool> operator()(const vector<edm4hep::MCParticleData>& evt) {
    ROOT::VecOps::RVec<bool> particles;
    for(const auto& h: evt) {
      particles.push_back(h.PDG==PDG && h.generatorStatus==generatorStatus);      
    }
    return particles;
  };
  
  void SetGeneratorStatus(int genStat){
    generatorStatus = genStat;
  }
  void SetPDG(int pdg){
    PDG = pdg;
  }
  
private: 
  int generatorStatus;
  int PDG;
};


  

// Particle definitions and frame names.
struct partDetails{
  std::string fName;
  int pdg;
  int genStatus;
};




void ProcessTaggerG4GraphSingle(TString inName,
			  TString outName,
			  dd4hep::Detector& detector){
  
  ROOT::EnableImplicitMT();

  using namespace ROOT::Math;
  using namespace std;

  // Input Data Chain
  TChain* t = new TChain("events");
  t->Add(inName);

  ROOT::RDataFrame d0(*t, {"TaggerTrackerHits", "MCParticles"});
  //d0.Range(0,10000); // Limit events to analyse 

  // -------------------------
  // Get the DD4hep instance
  // Load the compact XML file
  // Initialize the position converter tool
  dd4hep::rec::CellIDPositionConverter cellid_converter(detector);
  // -------------------------

  //--------------------------------------------------------------------------------------------
  // Lambda Functions
  //--------------------------------------------------------------------------------------------
  
  // -------------------------
  // Beam Vector 
  // -------------------------
  auto beamVertex = [&](const std::vector<edm4hep::MCParticleData>& evt) {    

    for (const auto& h : evt) {
      if(h.generatorStatus!=simID) continue;
      return h.vertex;
      break;
    }
    return edm4hep::Vector3d();
  };

  // -------------------------
  // Beam Time 
  // -------------------------
  auto beamTime = [&](const std::vector<edm4hep::MCParticleData>& evt) {    

    for (const auto& h : evt) {
      if(h.generatorStatus!=simID) continue;
      return h.time;
    }
    return (float)0.0;
  };


  // -------------------------
  // Real Hit Positions
  // -------------------------
  auto real_position = [&](const std::vector<edm4hep::SimTrackerHitData>& hits) {
    
    std::vector<XYZVector> positions;

    for (const auto& h : hits) {
      XYZVector result(h.position.x,h.position.y,h.position.z);
      positions.push_back(result);
    }
    return positions;
  };

  // -------------------------
  // Cell Hit Positions
  // -------------------------
  auto cell_position = [&](const std::vector<edm4hep::SimTrackerHitData>& hits) {
    
    std::vector<XYZVector> positions;

    for (const auto& h : hits) {
      auto pos1 = cellid_converter.position(h.cellID);
      XYZVector result(pos1.x()*10,pos1.y()*10,pos1.z()*10);
      positions.push_back(result);
    }
    return positions;
  };



  //----------------------------------------------------------------------------
  // Fill Dataframes
  //----------------------------------------------------------------------------
  

   //-----------------------------------------
   // Hits positions by real/cell positions
   //-----------------------------------------
   auto d1 = d0.Define("nHits", "TaggerTrackerHits.size()")
     .Define("real_position",     real_position,           {"TaggerTrackerHits"})
     .Define("cell_position",     cell_position,           {"TaggerTrackerHits"})
     .Define("real_time",         "TaggerTrackerHits.time")
     .Define("real_EDep",         "TaggerTrackerHits.EDep");
   
   //-----------------------------------------
   // Chosen Particles to save
   //-----------------------------------------
   std::vector<partDetails> parts = {{"beamElectron",11,beamID},{"beamProton",2212,beamID},{"scatteredElectron",11,simID}};
   std::vector<std::string> Part_Vec;
   for(auto part: parts){
     std::string colName   = part.fName;
     std::string colNameFilter = colName+"_filter";
     std::string colNameX  = colName+"_x";
     std::string colNameY  = colName+"_y";
     std::string colNameZ  = colName+"_z";
     std::string colNameID = colName+"_id";
     std::string colX      = colName+".momentum.x";
     std::string colY      = colName+".momentum.y";
     std::string colZ      = colName+".momentum.z";
     //     std::string colID      = colName+".momentum.z";
     Part_Vec.push_back(colName);
     d1 = d1.Define(colNameFilter,getParticle(part.genStatus,part.pdg),{"MCParticles"})
       .Define(colName,  "MCParticles["+colNameFilter+"]")
       .Define(colNameX, [](RVecMC tracks){RVecD outvec; for(auto track:tracks) outvec.push_back(track.momentum.x); return outvec;},{colName})
       .Define(colNameY, [](RVecMC tracks){RVecD outvec; for(auto track:tracks) outvec.push_back(track.momentum.y); return outvec;},{colName})
       .Define(colNameZ, [](RVecMC tracks){RVecD outvec; for(auto track:tracks) outvec.push_back(track.momentum.z); return outvec;},{colName})
       .Define(colNameID,"Nonzero("+colNameFilter+")");
   }

   //-----------------------------------------
   // Hit detector IDs
   //-----------------------------------------
   auto ids = detector.readout("TaggerTrackerHits").idSpec().fields();
   std::vector<std::string> ID_Vec;
   for(auto id: ids){
     std::string colName = id.first+"ID";
     ID_Vec.push_back(colName);
     d1 = d1.Define(colName,getSubID(id.first,detector),{"TaggerTrackerHits"});
   }

   //Get hits
   d1 = d1.Alias("hitIndex","TaggerTrackerHits#0.index")
     //     .Define("truth",[](RVecI hitindex,RVecUL particleindex,RVecD x,RVecD y,RVecD z){RVecTruth outvec; for(auto inx:hitindex){ if(Any(particleindex==inx)) outvec.push_back({inx,x,y,z});else outvec.push_back({-1,0,0,0});} return outvec;},{"hitIndex","scatteredElectron_id","scatteredElectron_x","scatteredElectron_y","scatteredElectron_z"})
     .Define("truth_index",[](RVecI hitindex,RVecUL particleindex){RVecI outvec; for(int i=0; i<hitindex.size(); i++){ RVecB checkVec = particleindex==hitindex[i]; if(Any(checkVec)) outvec.push_back(ArgMax(checkVec));else outvec.push_back(-1);} return outvec;},{"hitIndex","scatteredElectron_id"})
     .Define("real_hitID",
	     [](RVecI hitindex,RVecUL particleindex){RVecI outvec; for(auto idx:hitindex){ if(idx>=0) outvec.push_back(particleindex[idx]); else outvec.push_back(-1);} return outvec;},
	     {"truth_index","scatteredElectron_id"})
     .Define("real_mom_x",
	     [](RVecI hitindex,RVecD particleindex){RVecD outvec; for(auto idx:hitindex){ if(idx>=0) outvec.push_back(particleindex[idx]); else outvec.push_back(0);} return outvec;},
	     {"truth_index","scatteredElectron_x"})
     .Define("real_mom_y",
	     [](RVecI hitindex,RVecD particleindex){RVecD outvec; for(auto idx:hitindex){ if(idx>=0) outvec.push_back(particleindex[idx]); else outvec.push_back(0);} return outvec;},
	     {"truth_index","scatteredElectron_y"})
     .Define("real_mom_z",
	     [](RVecI hitindex,RVecD particleindex){RVecD outvec; for(auto idx:hitindex){ if(idx>=0) outvec.push_back(particleindex[idx]); else outvec.push_back(0);} return outvec;},
	     {"truth_index","scatteredElectron_z"});
//      .Define("real_mom_x",[](RVecTruth truths){RVecD outvec; for(auto truth:truths) outvec.push_back(truth.x); return outvec;},{"truth"})
//      .Define("real_mom_y",[](RVecTruth truths){RVecD outvec; for(auto truth:truths) outvec.push_back(truth.y); return outvec;},{"truth"})
//      .Define("real_mom_z",[](RVecTruth truths){RVecD outvec; for(auto truth:truths) outvec.push_back(truth.z); return outvec;},{"truth"});


   d1 = d1.Define("NLayerHitsMod1",[](RVecUL scatindex, RVecI hitindex, RVecI layerindex, RVecI modindex){
		    ROOT::VecOps::RVec<int> outvec;
		    for(auto part: scatindex){
		      int count = 0;
		      for(int i=0; i<4; i++){

			if(Any(hitindex[modindex==1&&layerindex==i]==part)) count ++;
		      }
		      outvec.push_back(count);
		    }
		    return outvec;
		  },{"scatteredElectron_id","real_hitID","layerID","moduleID"})
     .Define("NLayerHitsMod2",[](RVecUL scatindex, RVecI hitindex, RVecI layerindex, RVecI modindex){
		    ROOT::VecOps::RVec<int> outvec;
		    for(auto part: scatindex){
		      int count = 0;
		      for(int i=0; i<4; i++){

			if(Any(hitindex[modindex==2&&layerindex==i]==part)) count ++;
		      }
		      outvec.push_back(count);
		    }
		    return outvec;
		  },{"scatteredElectron_id","real_hitID","layerID","moduleID"})
     .Define("real_mod1hits",
	     [](RVecI hitindex,RVecI particleindex){RVecI outvec; for(auto idx:hitindex){ if(idx>=0) outvec.push_back(particleindex[idx]); else outvec.push_back(0);} return outvec;},
	     {"truth_index","NLayerHitsMod1"})
     .Define("real_mod2hits",
	     [](RVecI hitindex,RVecI particleindex){RVecI outvec; for(auto idx:hitindex){ if(idx>=0) outvec.push_back(particleindex[idx]); else outvec.push_back(0);} return outvec;},
	     {"truth_index","NLayerHitsMod2"});
     
   d1.Snapshot("particles",outName,{"scatteredElectron_x","scatteredElectron_y","scatteredElectron_z","scatteredElectron_id","NLayerHitsMod1","NLayerHitsMod2"});
  
   ROOT::RDF::RSnapshotOptions opts;
   opts.fMode = "UPDATE";

   d1.Snapshot("hits",outName,{"moduleID","layerID","xID","yID","real_time","real_EDep","real_hitID","real_mom_x","real_mom_y","real_mom_z","real_mod1hits","real_mod2hits"},opts);

}

//--------------------------------------------------------------------------------------------

//-----------------------------------------------------------------------------------------
// Main Analysis Call
//-----------------------------------------------------------------------------------------
void ProcessTaggerG4Graph(){

  string compactName = "/home/simon/EIC/epic/epic_18x275.xml";
  dd4hep::Detector& detector = dd4hep::Detector::getInstance();
  detector.fromCompact(compactName);

  ProcessTaggerG4GraphSingle("/scratch/EIC/G4out/qr_bx_18x2750.edm4hep.root",
 			     "/scratch/EIC/GraphData/out_graph_0.root",
 			     detector
 			     );
  ProcessTaggerG4GraphSingle("/scratch/EIC/G4out/qr_bx_18x2751.edm4hep.root",
			     "/scratch/EIC/GraphData/out_graph_1.root",
			     detector
			     );
  ProcessTaggerG4GraphSingle("/scratch/EIC/G4out/qr_bx_18x2752.edm4hep.root",
			     "/scratch/EIC/GraphData/out_graph_2.root",
			     detector
			     );
  ProcessTaggerG4GraphSingle("/scratch/EIC/G4out/qr_bx_18x2753.edm4hep.root",
			     "/scratch/EIC/GraphData/out_graph_3.root",
			     detector
			     );
  ProcessTaggerG4GraphSingle("/scratch/EIC/G4out/qr_bx_18x2754.edm4hep.root",
			     "/scratch/EIC/GraphData/out_graph_4.root",
			     detector
			     );
  ProcessTaggerG4GraphSingle("/scratch/EIC/G4out/qr_bx_18x2755.edm4hep.root",
			     "/scratch/EIC/GraphData/out_graph_5.root",
			     detector
			     );
  ProcessTaggerG4GraphSingle("/scratch/EIC/G4out/qr_bx_18x2756.edm4hep.root",
			     "/scratch/EIC/GraphData/out_graph_6.root",
			     detector
			     );
  ProcessTaggerG4GraphSingle("/scratch/EIC/G4out/qr_bx_18x2757.edm4hep.root",
			     "/scratch/EIC/GraphData/out_graph_7.root",
			     detector
			     );
  ProcessTaggerG4GraphSingle("/scratch/EIC/G4out/qr_bx_18x2758.edm4hep.root",
			     "/scratch/EIC/GraphData/out_graph_8.root",
			     detector
			     );
  ProcessTaggerG4GraphSingle("/scratch/EIC/G4out/qr_bx_18x2759.edm4hep.root",
			     "/scratch/EIC/GraphData/out_graph_9.root",
			     detector
			     );
  ProcessTaggerG4GraphSingle("/scratch/EIC/G4out/qr_bx_18x27510.edm4hep.root",
			     "/scratch/EIC/GraphData/out_graph_10.root",
			     detector
			     );
  ProcessTaggerG4GraphSingle("/scratch/EIC/G4out/qr_bx_18x27511.edm4hep.root",
			     "/scratch/EIC/GraphData/out_graph_11.root",
			     detector
			     );
  ProcessTaggerG4GraphSingle("/scratch/EIC/G4out/qr_bx_18x27512.edm4hep.root",
			     "/scratch/EIC/GraphData/out_graph_12.root",
			     detector
			     );
  ProcessTaggerG4GraphSingle("/scratch/EIC/G4out/qr_bx_18x27513.edm4hep.root",
			     "/scratch/EIC/GraphData/out_graph_13.root",
			     detector
			     );
  ProcessTaggerG4GraphSingle("/scratch/EIC/G4out/qr_bx_18x27514.edm4hep.root",
			     "/scratch/EIC/GraphData/out_graph_14.root",
			     detector
			     );
  ProcessTaggerG4GraphSingle("/scratch/EIC/G4out/qr_bx_18x27515.edm4hep.root",
			     "/scratch/EIC/GraphData/out_graph_15.root",
			     detector
			     );
  ProcessTaggerG4GraphSingle("/scratch/EIC/G4out/qr_bx_18x27516.edm4hep.root",
			     "/scratch/EIC/GraphData/out_graph_16.root",
			     detector
			     );
  ProcessTaggerG4GraphSingle("/scratch/EIC/G4out/qr_bx_18x27517.edm4hep.root",
			     "/scratch/EIC/GraphData/out_graph_17.root",
			     detector
			     );
}
