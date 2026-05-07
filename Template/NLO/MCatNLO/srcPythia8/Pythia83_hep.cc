// Driver for Pythia 8. Reads an input file dynamically created on
// the basis of the inputs specified in MCatNLO_MadFKS_PY8.Script
#include "Pythia8/Pythia.h"
#include "Pythia8Plugins/HepMC2.h"
#include "Pythia8Plugins/aMCatNLOHooks.h"
#include "Pythia8Plugins/CombineMatchingInput.h"
#include "HepMC/GenEvent.h"
#include "HepMC/IO_GenEvent.h"
#include <fstream>
#include <vector>

using namespace Pythia8;

int main() {
  Pythia pythia;

  // Register the q-cut variation list before reading the Pythia card so user
  // input "JetMatching:qCutList = 15 20 30 40" is accepted by the parser.
  // Empty default ⇒ legacy single-q-cut mode.
  pythia.settings.addPVec("JetMatching:qCutList", std::vector<double>(),
                          false, false, 0., 0.);

  string inputname="Pythia8.cmd",outputname="Pythia8.hep";

  pythia.readFile(inputname.c_str());

  //Create UserHooks pointer for the FxFX matching. Stop if it failed. Pass pointer to Pythia.
  CombineMatchingInput combined;
  //UserHooks* matching            = NULL;

  int nAbort=10;
  int nPrintLHA=1;
  int iAbort=0;
  int iPrintLHA=0;
  int iEventtot=pythia.mode("Main:numberOfEvents");
  int iEventshower=pythia.mode("Main:spareMode1");
  string evt_norm=pythia.word("Main:spareWord1");

  //FxFx merging
  bool isFxFx=pythia.flag("JetMatching:doFxFx");
  if (isFxFx) {
    combined.setHook(pythia);
    //matching = combined->getHook(pythia);
    //if (!matching) {
    //  std::cout << " Failed to initialise jet matching structures.\n"
    //            << " Program stopped.";
    //  return 1;
    //}
    //pythia.setUserHooksPtr(matching);
    int nJmax=pythia.mode("JetMatching:nJetMax");
    double Qcut=pythia.parm("JetMatching:qCut");
    double PTcut=pythia.parm("JetMatching:qCutME");
    if (Qcut <= PTcut || Qcut <= 0.) {
      std::cout << " \n";
      std::cout << "Merging scale (shower_card.dat) smaller than pTcut (run_card.dat)"
		<< Qcut << " " << PTcut << "\n";
      return 0;
    }
  }

  // q-cut variation: detect non-empty qCutList, grab a typed pointer to the
  // matching hook, and open the sidecar accept-flag file. Each row in the
  // sidecar carries the HepMC event_number and one accept flag (1 = kept,
  // 0 = vetoed) per q-cut variation, in the same order as qCutList. The
  // q-cut grid is recorded in the header comment.
  std::vector<double> qCutList = pythia.settings.pvec("JetMatching:qCutList");
  bool doQCutVariation = isFxFx && !qCutList.empty();
  std::ofstream qcutSidecar;
  JetMatchingMadgraph* jmHook = nullptr;
  if (doQCutVariation) {
    jmHook = dynamic_cast<JetMatchingMadgraph*>(combined.hook.get());
    if (!jmHook) {
      std::cout << "Error: JetMatching:qCutList requires "
                << "JetMatchingMadgraph hook (FxFx scheme = 1)." << std::endl;
      return 1;
    }
    qcutSidecar.open("Pythia8.qcut_accept");
    qcutSidecar << "# qcut_grid:";
    for (double q : qCutList) qcutSidecar << ' ' << q;
    qcutSidecar << "\n# evt_idx";
    for (double q : qCutList) qcutSidecar << " a_" << q;
    qcutSidecar << '\n';
  }

  // Initialise Pythia.
  if (!pythia.init()) {
    cout << "Error: could not initialise Pythia" << endl;
    return 0;
  };

  HepMC::Pythia8ToHepMC ToHepMC;
  HepMC::IO_GenEvent ascii_io(outputname.c_str(), std::ios::out);
  // Do not store cross section information, as this will be done manually.
  ToHepMC.set_store_pdf(false);
  ToHepMC.set_store_proc(false);
  ToHepMC.set_store_xsec(false);

  // Cross section an error.
  double sigmaTotal  = 0.;
  double errorTotal  = 0.;

  for (int iEvent = 0; ; ++iEvent) {
    if (!pythia.next()) {
      if (++iAbort < nAbort) continue;
      break;
    }
    // the number of events read by Pythia so far
    int nSelected=pythia.info.nSelected();

    if (nSelected > iEventshower) break;
    if (pythia.info.isLHA() && iPrintLHA < nPrintLHA) {
      pythia.LHAeventList();
      pythia.info.list();
      pythia.process.list();
      pythia.event.list();
      ++iPrintLHA;
    }

    HepMC::GenEvent* hepmcevt = new HepMC::GenEvent();
    double evtweight = pythia.info.weight();
    double normhepmc;
    // ALWAYS NORMALISE HEPMC WEIGHTS TO SUM TO THE CROSS SECTION
    if (evt_norm != "sum") {
      normhepmc = 1. / double(iEventshower);
    } else {
      normhepmc = double(iEventtot) / double(iEventshower);
    }
    sigmaTotal += evtweight*normhepmc;
    hepmcevt->weights().push_back(evtweight*normhepmc);
    ToHepMC.fill_next_event( pythia, hepmcevt );
    // Add the weight of the current event to the cross section.
    // Report cross section to hepmc
    HepMC::GenCrossSection xsec;
    xsec.set_cross_section( sigmaTotal, pythia.info.sigmaErr() );
    hepmcevt->set_cross_section( xsec );
    // Write the HepMC event to file. Done with it.
    ascii_io << hepmcevt;

    // q-cut variation: write per-event accept-flag row to the sidecar.
    // flags[i] = 1 means the matching at qCutList[i] vetoed the event;
    // we write (1 - flags[i]) so that "1" denotes "kept".
    if (doQCutVariation) {
      const std::vector<int>& flags = jmHook->getVetoVector();
      qcutSidecar << hepmcevt->event_number();
      for (size_t i = 0; i < flags.size(); ++i)
        qcutSidecar << ' ' << (1 - flags[i]);
      qcutSidecar << '\n';
    }

    delete hepmcevt;
  }
  if (doQCutVariation) qcutSidecar.close();

  pythia.stat();
  if (isFxFx){
    std::cout << " \n";
    std::cout << "*********************************************************************** \n";
    std::cout << "*********************************************************************** \n";
    std::cout << "Cross section, including FxFx merging is: "
	      << sigmaTotal << "\n";
    std::cout << "*********************************************************************** \n";
    std::cout << "*********************************************************************** \n";
  }

  return 0;
}
