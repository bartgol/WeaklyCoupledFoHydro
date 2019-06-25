#include <Albany_config.h>
#include <Albany_SolverFactory.hpp>
#include <Albany_Session.hpp>
#include <Albany_CommUtils.hpp>
#include <Albany_ModelEvaluator.hpp>
#include <Albany_ScalarOrdinalTypes.hpp>
#include <Albany_STKDiscretization.hpp>

#include <Piro_PerformSolve.hpp>
#include <Piro_PerformAnalysis.hpp>
#include <Piro_NOXSolver.hpp>
#include <Thyra_DefaultModelEvaluatorWithSolveFactory.hpp>

#include <Teuchos_YamlParameterListHelpers.hpp>
#include <Teuchos_GlobalMPISession.hpp>
#include <Teuchos_VerboseObject.hpp>

struct AlbanySolvers {
  // TODO: if, upon solver creation, you never use the factories
  //       other than for getting parameters, consider storing just
  //       a parameter list

  using TROME = Thyra::ResponseOnlyModelEvaluatorBase<ST>;

  Teuchos::RCP<Albany::SolverFactory> m_inv_fo_factory;
  Teuchos::RCP<Albany::SolverFactory> m_inv_hydro_factory;
  Teuchos::RCP<Albany::SolverFactory> m_fwd_fo_factory;

  Teuchos::RCP<TROME> m_inv_fo_solver;
  Teuchos::RCP<TROME> m_inv_hydro_solver;
  Teuchos::RCP<TROME> m_fwd_fo_solver;

  int m_max_iters;
  int m_tolerance;
};

AlbanySolvers create_solvers(const std::string& fo_inv_fname,
                             const std::string& hydro_inv_fname,
                             const std::string& fo_fwd_fname,
                             const std::string& coupling_params_fname);

int run_initialization_problem (const AlbanySolvers& solvers,
                                Teuchos::RCP<Teuchos::FancyOStream> out);
int run_iterative_inversion_problem (const AlbanySolvers& solvers,
                                     Teuchos::RCP<Teuchos::FancyOStream> out);

/////////////// IMPLEMENTATIONS /////////////////////

int main(int argc, char** argv) {
  int status = 0;

  // Init MPI and Kokkos
  Teuchos::GlobalMPISession mpiSession(&argc, &argv);
  Kokkos::initialize(argc,argv);

  // Get the comm
  Teuchos::RCP<const Teuchos_Comm> comm = Albany::getDefaultComm();

  // Init output stream
  Teuchos::RCP<Teuchos::FancyOStream> out;
  out = Teuchos::VerboseObjectBase::getDefaultOStream();
  out->setProcRankAndSize(comm->getRank(),comm->getSize());
  out->setOutputToRootOnly(0);

  *out << "   +------------------------------------------------------------+\n"
       << "   |   Weakly coupled StokesFO-Hydrology parameter inference    |\n"
       << "   +------------------------------------------------------------+\n"
       << "\n"
       << " This script will attempt to infer parameters in the subglacial hydrology   \n"
       << " model as well as in the Schoof's sliding law for the ice.                  \n"
       << " To this end, we first solve an inverse problem for the ice, assimilating   \n"
       << " surface velocity measures, and estimating the basal traction field.        \n"
       << " Then, we start our iterative procedure: at step k, estimate the parameters \n"
       << " solving an inverse problem for the hydrology, trying to match the traction \n"
       << " field produced by the ice solver at step k-1 with the one obtained using   \n"
       << " Schoof's sliding law with the estimated parameters. Then use the estimated \n"
       << " parameters for the sliding law to solve a forward problem for the ice.     \n"
       << " The iterative procedure continues until a maximum number of iterations is  \n"
       << " reached, or the quantity ||u(k) - u(k-1)||/||u(k)|| drops below a given    \n"
       << " tolerance, (here u is the sliding velocity of the ice).                    \n"
       << "\n";

  // Read input line
  std::string fo_inv_fname = "input_fo_inverse.yaml";
  std::string fo_fwd_fname = "input_hydro_inverse.yaml";
  std::string hydro_inv_fname = "input_fo_forward.yaml";
  std::string coupling_fname = "input_coupling.yaml";

  if (argc>1) {
    if (!std::strcmp(argv[1],"-h") ||
        !std::strcmp(argv[1],"--help"))
    {
      *out << " Usage:\n\n"
           << "   weakly_coupled_fo_hydro [inverse_fo_input_file [inverse_hydro_input_file [forward_fo_input_file [weak_coupling_input_file]]]]\n"
           << "\n"
           << " The input file names default to the following:\n"
           << "  inverse_fo_input_file: input_fo_inverse.yaml\n" 
           << "  inverse_fo_input_file: input_hydro_inverse.yaml\n" 
           << "  inverse_fo_input_file: input_fo_forward.yaml\n"
           << "  weak_coupling_input_file: input_coupling.yaml\n\n";

      Kokkos::finalize_all();
      return 0;
    }
    fo_inv_fname = argv[1];
    if (argc>2) {
      hydro_inv_fname = argv[2];
      if (argc>3) {
        fo_fwd_fname = argv[3];
        if (argc>4) {
          coupling_fname = argv[4];
        }
      }
    }
  }

  // Create solvers
  AlbanySolvers solvers = create_solvers(fo_inv_fname, hydro_inv_fname, fo_fwd_fname, coupling_fname);

  // Run initialization phase
  status = run_initialization_problem (solvers, out);
  if (status!=0) {
    return status;
  }

  // Run inverse problem
  status = run_iterative_inversion_problem (solvers, out);
  Kokkos::finalize_all();

  return status;
}

AlbanySolvers create_solvers(const std::string& fo_inv_fname,
                             const std::string& hydro_inv_fname,
                             const std::string& fo_fwd_fname,
                             const std::string& coupling_params_fname)
{
  using namespace Albany;

  // Get the comm
  Teuchos::RCP<const Teuchos_Comm> comm = Albany::getDefaultComm();

  AlbanySolvers solvers;

  // Create solver factories
  solvers.m_inv_fo_factory    = Teuchos::rcp( new Albany::SolverFactory(fo_inv_fname,comm) );
  solvers.m_inv_hydro_factory = Teuchos::rcp( new Albany::SolverFactory(hydro_inv_fname,comm) );
  solvers.m_fwd_fo_factory    = Teuchos::rcp( new Albany::SolverFactory(fo_fwd_fname,comm) );

  Session::reset_build_type(solvers.m_inv_fo_factory->getParameters().get<std::string>("Build Type"));
  solvers.m_inv_fo_solver    = solvers.m_inv_fo_factory->create(comm,comm);

  Session::reset_build_type(solvers.m_inv_hydro_factory->getParameters().get<std::string>("Build Type"));
  solvers.m_inv_hydro_solver = solvers.m_inv_hydro_factory->create(comm,comm);

  Session::reset_build_type(solvers.m_fwd_fo_factory->getParameters().get<std::string>("Build Type"));
  solvers.m_fwd_fo_solver    = solvers.m_fwd_fo_factory->create(comm,comm);

  Teuchos::RCP<Teuchos::ParameterList> params = Teuchos::createParameterList("");
  Teuchos::updateParametersFromYamlFileAndBroadcast(coupling_params_fname,params.ptr(),*comm);

  solvers.m_max_iters = params->get<int>("Max Iters",10);
  solvers.m_tolerance = params->get<RealType>("Relative Tolerance",1e-3);

  return solvers;
}

int run_initialization_problem (const AlbanySolvers& solvers,
                                Teuchos::RCP<Teuchos::FancyOStream> out)
{
  using namespace Albany;

  *out << "   +------------------------------------------------------------+\n"
       << "   |            Initializing with inverse FO problem            |\n"
       << "   +------------------------------------------------------------+\n"
       << "\n";

  // Get the comm
  Teuchos::RCP<const Teuchos_Comm> comm = Albany::getDefaultComm();

  auto fo_solver  = solvers.m_inv_fo_solver;
  auto fo_factory = solvers.m_inv_fo_factory;
  auto fo_model   = fo_factory->returnModel();

  // Solve
  Teuchos::Array<Teuchos::RCP<const Thyra_Vector>> responses;
  Teuchos::Array<Teuchos::Array<Teuchos::RCP<const Thyra_MultiVector>>> sensitivities;
  Piro::PerformSolve(*fo_solver,fo_factory->getAnalysisParameters(),responses,sensitivities);

  // Set basal traction and sliding velocity in the hydrology discretization
  auto hydro_solver  = solvers.m_inv_hydro_solver;
  auto hydro_factory = solvers.m_inv_hydro_factory;
  auto hydro_model   = hydro_factory->returnModel();

  auto basalSideName = fo_model->getApp()->getProblemPL()->get<std::string>("Basal Side Name");

  auto fo_basal_disc = fo_model->getApp()->getDiscretization()->getSideSetDiscretizations().at(basalSideName);
  auto hydro_disc    = hydro_model->getApp()->getDiscretization();

  auto fo_basal_states = fo_basal_disc->getStateArrays().nodeStateArrays;
  auto hydro_states = hydro_disc->getStateArrays().nodeStateArrays;

  const size_t num_buckets = fo_basal_states.size();
  const std::string slidingVelName = "sliding_velocity";
  const std::string tractionName   = "basal_traction";
  for (size_t ib=0; ib<num_buckets; ++ib) {
    // Set computed sliding velocity on basal mesh
    auto sliding_v_hydro = hydro_states[ib].at(slidingVelName);
    auto sliding_v_fo = fo_basal_states[ib].at(slidingVelName+"_"+basalSideName);

    ALBANY_ASSERT(sliding_v_hydro.size()==sliding_v_fo.size(),
                  "Error! Something is wrong with states dimensions.\n");
    for (auto i = decltype(sliding_v_fo.size())(0); i<sliding_v_fo.size(); ++i) {
      sliding_v_hydro(i) = sliding_v_fo(i);
    }

    // Set computed traction field on mesh
    auto traction_hydro = hydro_states[ib].at(slidingVelName);
    auto traction_fo = fo_basal_states[ib].at(slidingVelName+"_"+basalSideName);

    ALBANY_ASSERT(traction_hydro.size()==traction_fo.size(),
                  "Error! Something is wrong with states dimensions.\n");
    for (auto i = decltype(traction_fo.size())(0); i<traction_fo.size(); ++i) {
      traction_hydro(i) = traction_fo(i);
    }
  }

  return 0;
}

int run_iterative_inversion_problem (const AlbanySolvers& solvers,
                                     Teuchos::RCP<Teuchos::FancyOStream> out)
{
  using namespace Albany;

  *out << "   +------------------------------------------------------------+\n"
       << "   |              Running fixed point algorithm                 |\n"
       << "   +------------------------------------------------------------+\n"
       << "\n";


  // Get the comm
  Teuchos::RCP<const Teuchos_Comm> comm = Albany::getDefaultComm();

  auto hydro_solver  = solvers.m_inv_hydro_solver;
  auto hydro_factory = solvers.m_inv_hydro_factory;
  auto hydro_model   = hydro_factory->returnModel();

  auto fo_solver  = solvers.m_fwd_fo_solver;
  auto fo_factory = solvers.m_fwd_fo_factory;
  auto fo_model   = fo_factory->returnModel();

  auto hydro_bt = solvers.m_inv_hydro_factory->getParameters().get<std::string>("Build Type");
  auto fo_bt    = solvers.m_fwd_fo_factory->getParameters().get<std::string>("Build Type");

  auto basalSideName = fo_model->getApp()->getProblemPL()->get<std::string>("Basal Side Name");

  auto fo_basal_disc = fo_model->getApp()->getDiscretization()->getSideSetDiscretizations().at(basalSideName);
  auto hydro_disc    = hydro_model->getApp()->getDiscretization();

  auto fo_basal_states = fo_basal_disc->getStateArrays().nodeStateArrays;
  auto hydro_states = hydro_disc->getStateArrays().nodeStateArrays;

  const size_t num_buckets = fo_basal_states.size();
  ALBANY_ASSERT (num_buckets==hydro_states.size(),
                 "Error! Something is wrong with the buckets size.\n");

  const std::string effPressName   = "effective_pressure";
  const std::string slidingVelName = "sliding_velocity";

  RealType error = 1;
  int status = 0;
  Teuchos::RCP<Thyra_Vector> p;
  for (int iter=0; iter<solvers.m_max_iters && error>solvers.m_tolerance; ++iter) {
    // Solve inverse hydrology
    Session::reset_build_type(hydro_bt);

    // Set inputs in hydrology discretization
    for (size_t ib=0; ib<num_buckets; ++ib) {
      // Set computed sliding velocity on basal mesh
      auto sliding_v_hydro = hydro_states[ib].at(slidingVelName);
      auto sliding_v_fo = fo_basal_states[ib].at(slidingVelName+"_"+basalSideName);

      ALBANY_ASSERT(sliding_v_hydro.size()==sliding_v_fo.size(),
                    "Error! Something is wrong with states dimensions.\n");
      for (auto i = decltype(sliding_v_fo.size())(0); i<sliding_v_fo.size(); ++i) {
        sliding_v_hydro(i) = sliding_v_fo(i);
      }

      // Set computed traction field on mesh
      auto traction_hydro = hydro_states[ib].at(slidingVelName);
      auto traction_fo = fo_basal_states[ib].at(slidingVelName+"_"+basalSideName);

      ALBANY_ASSERT(traction_hydro.size()==traction_fo.size(),
                    "Error! Something is wrong with states dimensions.\n");
      for (auto i = decltype(traction_fo.size())(0); i<traction_fo.size(); ++i) {
        traction_hydro(i) = traction_fo(i);
      }
    }
    status = Piro::PerformAnalysis(*hydro_solver,hydro_factory->getAnalysisParameters(),p);

    // If something went wrong, quit
    if (status!=0) {
      return status;
    }

    // Solve forward ice
    Session::reset_build_type(fo_bt);

    // Set parameters in ice problem (only first block, corresponding to sliding law)
    auto p_prod = Teuchos::rcp_dynamic_cast<Thyra_ProductVector>(p,true);
    auto p_schoof = p_prod->getVectorBlock(0);
    auto nominalValues = fo_model->getNominalValues();
    nominalValues.set_p(0,p_schoof);
    fo_model->setNominalValues(nominalValues);

    for (size_t ib=0; ib<num_buckets; ++ib) {
      // Set computed effective pressure in ice mesh
      auto eff_press_hydro = hydro_states[ib].at(effPressName);
      auto eff_press_fo = fo_basal_states[ib].at(effPressName+"_"+basalSideName);

      ALBANY_ASSERT(eff_press_hydro.size()==eff_press_fo.size(),
                    "Error! Something is wrong with states dimensions.\n");
      for (auto i = decltype(eff_press_fo.size())(0); i<eff_press_fo.size(); ++i) {
        eff_press_fo(i) = eff_press_hydro(i);
      }
    }

    Piro::PerformSolve(*fo_solver,fo_factory->getAnalysisParameters().sublist("Solve"),p);
  }
  return 0;
}
