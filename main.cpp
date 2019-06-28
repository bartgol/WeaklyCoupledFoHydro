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

  Teuchos::RCP<Albany::CombineAndScatterManager> m_sliding_cas;
  Teuchos::RCP<Albany::CombineAndScatterManager> m_traction_cas;
  Teuchos::RCP<Albany::CombineAndScatterManager> m_eff_press_cas;

  Teuchos::RCP<Thyra_Vector> m_sliding_fo;
  Teuchos::RCP<Thyra_Vector> m_traction_fo;
  Teuchos::RCP<Thyra_Vector> m_eff_press_fo;
  Teuchos::RCP<Thyra_Vector> m_sliding_hydro;
  Teuchos::RCP<Thyra_Vector> m_traction_hydro;
  Teuchos::RCP<Thyra_Vector> m_eff_press_hydro;

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

void copyToHydro (const AlbanySolvers& solvers,
                  const Teuchos::RCP<Albany::AbstractDiscretization> fo_basal_disc,
                  const Teuchos::RCP<Albany::AbstractDiscretization> hydro_disc);

void copyFromHydro (const AlbanySolvers& solvers,
                    const Teuchos::RCP<Albany::AbstractDiscretization> fo_basal_disc,
                    const Teuchos::RCP<Albany::AbstractDiscretization> hydro_disc);

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
  std::string fo_fwd_fname = "input_fo_forward.yaml";
  std::string hydro_inv_fname = "input_hydro_inverse.yaml";
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

  Teuchos::RCP<Teuchos::FancyOStream> out;
  out = Teuchos::VerboseObjectBase::getDefaultOStream();
  out->setProcRankAndSize(comm->getRank(),comm->getSize());
  out->setOutputToRootOnly(0);
  *out << "   +------------------------------------------------------------+\n"
       << "   |            Creating the ice and hydrology solvers          |\n"
       << "   +------------------------------------------------------------+\n"
       << "\n";

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

  // Create cas managers
  auto basalSideName = solvers.m_fwd_fo_factory->returnModel()->getApp()->getProblemPL()->get<std::string>("Basal Side Name");
  const auto fo_basal_disc = solvers.m_fwd_fo_factory->returnModel()->getApp()->getDiscretization()->getSideSetDiscretizations().at(basalSideName);
  const auto hydro_disc    = solvers.m_inv_hydro_factory->returnModel()->getApp()->getDiscretization();

  const std::string slidingVelName = "sliding_velocity";
  const std::string tractionName   = "basal_traction";
  const std::string effPressName   = "effective_pressure";

  solvers.m_sliding_cas = Albany::createCombineAndScatterManager(fo_basal_disc->getVectorSpace(slidingVelName),
                                                                 hydro_disc->getOverlapVectorSpace(slidingVelName));
  solvers.m_traction_cas = Albany::createCombineAndScatterManager(fo_basal_disc->getVectorSpace(tractionName),
                                                                  hydro_disc->getOverlapVectorSpace(tractionName));
  solvers.m_eff_press_cas = Albany::createCombineAndScatterManager(fo_basal_disc->getVectorSpace(effPressName),
                                                                   hydro_disc->getOverlapVectorSpace(effPressName));

  // Create thyra vectors to help with cas import/export
  solvers.m_sliding_fo      = Thyra::createMember(solvers.m_sliding_cas->getOwnedVectorSpace());
  solvers.m_sliding_hydro   = Thyra::createMember(solvers.m_sliding_cas->getOverlappedVectorSpace());
  solvers.m_traction_fo     = Thyra::createMember(solvers.m_traction_cas->getOwnedVectorSpace());
  solvers.m_traction_hydro  = Thyra::createMember(solvers.m_traction_cas->getOverlappedVectorSpace());
  solvers.m_eff_press_fo    = Thyra::createMember(solvers.m_eff_press_cas->getOwnedVectorSpace());
  solvers.m_eff_press_hydro = Thyra::createMember(solvers.m_eff_press_cas->getOverlappedVectorSpace());

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

  auto basalSideName = fo_model->getApp()->getProblemPL()->get<std::string>("Basal Side Name");

  auto fo_basal_disc = fo_model->getApp()->getDiscretization()->getSideSetDiscretizations().at(basalSideName);
  auto hydro_disc    = solvers.m_inv_hydro_factory->returnModel()->getApp()->getDiscretization();
  copyToHydro(solvers, fo_basal_disc, hydro_disc);

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

  auto fo_basal_states = fo_basal_disc->getStateArrays().elemStateArrays;
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
    copyToHydro(solvers, fo_basal_disc, hydro_disc);

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

    copyFromHydro(solvers, fo_basal_disc, hydro_disc);

    Piro::PerformSolve(*fo_solver,fo_factory->getAnalysisParameters().sublist("Solve"),p);
  }
  return 0;
}

void copyToHydro (const AlbanySolvers& solvers,
                  const Teuchos::RCP<Albany::AbstractDiscretization> fo_basal_disc,
                  const Teuchos::RCP<Albany::AbstractDiscretization> hydro_disc) {
  const std::string slidingVelName = "sliding_velocity";
  const std::string tractionName   = "basal_traction";

  solvers.m_sliding_fo->assign(0.0);
  solvers.m_traction_fo->assign(0.0);

  // Copy from fo stk structures into Thyra_Vector's
  const auto fo_basal_states  = fo_basal_disc->getStateArrays().elemStateArrays;
  const size_t num_fo_buckets = fo_basal_states.size();
  auto sliding_fo_thyra_data  = Albany::getNonconstLocalData(solvers.m_sliding_fo);
  auto traction_fo_thyra_data = Albany::getNonconstLocalData(solvers.m_traction_fo);
  auto traction_fo_dof_manager   = fo_basal_disc->getDOFManager(tractionName);
  for (size_t ib=0; ib<num_fo_buckets; ++ib) {
    // Get arrays from basal mesh
    const auto sliding_v_fo = fo_basal_states[ib].at(slidingVelName);
    const auto traction_fo  = fo_basal_states[ib].at(slidingVelName);

    // Loop over elem/nodes in the workset, and copy over to Thyra 
    const int num_elems = sliding_v_fo.dimension(0);
    const int num_nodes = sliding_v_fo.dimension(1);
    const auto& ElNodeID = fo_basal_disc->getWsElNodeID()[ib];
    for (int ielem=0; ielem<num_elems; ++ielem) {
      for (int inode=0; inode<num_nodes; ++inode) {
        const GO node_gid = ElNodeID[ielem][inode];
        const LO node_lid = Albany::getLocalElement(solvers.m_sliding_fo->space(),node_gid);

        // Only process local nodes, ghosted ones will be imported
        if (Albany::locallyOwnedComponent(Albany::getSpmdVectorSpace(solvers.m_sliding_fo->space()),node_gid)) {
          sliding_fo_thyra_data[node_lid] = sliding_v_fo(ielem,inode);

          // For velocity, use its dof manager, cause you don't know if it's interleaved
          for (int icomp=0; icomp<2; ++icomp) {
            const GO dof_gid = traction_fo_dof_manager.getGlobalDOF(node_gid,icomp);
            const LO dof_lid = Albany::getLocalElement(solvers.m_traction_fo->space(),dof_gid);
            traction_fo_thyra_data[dof_lid] = traction_fo(ielem,inode,icomp);
          }
        }
      }
    }
  }

  // Export
  solvers.m_sliding_hydro->assign(0.0);
  solvers.m_traction_hydro->assign(0.0);
  solvers.m_sliding_cas->scatter(solvers.m_sliding_fo,solvers.m_sliding_hydro,Albany::CombineMode::INSERT);
  solvers.m_traction_cas->scatter(solvers.m_traction_fo,solvers.m_traction_hydro,Albany::CombineMode::INSERT);

  // Copy from Thyra_Vector's into hydro stk structures
  auto hydro_states = hydro_disc->getStateArrays().elemStateArrays;
  const size_t num_hydro_buckets  = hydro_states.size();
  auto sliding_hydro_thyra_data   = Albany::getLocalData(solvers.m_sliding_hydro.getConst());
  auto traction_hydro_thyra_data  = Albany::getLocalData(solvers.m_traction_hydro.getConst());
  auto traction_hydro_dof_manager = hydro_disc->getDOFManager(tractionName);
  for (size_t ib=0; ib<num_hydro_buckets; ++ib) {
    // Get arrays from mesh
    const auto sliding_v_hydro = hydro_states[ib].at(slidingVelName);
    const auto traction_hydro  = hydro_states[ib].at(slidingVelName);

    // Loop over elem/nodes in the workset, and copy over to Thyra 
    const int num_elems = sliding_v_hydro.dimension(0);
    const int num_nodes = sliding_v_hydro.dimension(1);
    const auto& ElNodeID = hydro_disc->getWsElNodeID()[ib];
    for (int ielem=0; ielem<num_elems; ++ielem) {
      for (int inode=0; inode<num_nodes; ++inode) {
        const GO node_gid = ElNodeID[ielem][inode];
        const LO node_lid = Albany::getLocalElement(solvers.m_sliding_hydro->space(),node_gid);
        sliding_v_hydro (ielem,inode) = sliding_hydro_thyra_data[node_lid] ;

        // For velocity, use its dof manager, cause you don't know if it's interleaved
        for (int icomp=0; icomp<2; ++icomp) {
          const GO dof_gid = traction_hydro_dof_manager.getGlobalDOF(node_gid,icomp);
          const LO dof_lid = Albany::getLocalElement(solvers.m_traction_hydro->space(),dof_gid);
          traction_hydro(ielem,inode,icomp) = traction_hydro_thyra_data[dof_lid];
        }
      }
    }
  }
}

void copyFromHydro (const AlbanySolvers& solvers,
                  const Teuchos::RCP<Albany::AbstractDiscretization> fo_basal_disc,
                  const Teuchos::RCP<Albany::AbstractDiscretization> hydro_disc) {
  const std::string effPressName   = "effective_pressure";

  solvers.m_eff_press_hydro->assign(0.0);

  // Copy from fo stk structures into Thyra_Vector's
  auto hydro_states = hydro_disc->getStateArrays().elemStateArrays;
  const size_t num_hydro_buckets  = hydro_states.size();
  auto eff_press_hydro_thyra_data   = Albany::getNonconstLocalData(solvers.m_eff_press_hydro);
  for (size_t ib=0; ib<num_hydro_buckets; ++ib) {
    // Get arrays from basal mesh
    const auto eff_press_hydro = hydro_states[ib].at(effPressName);

    // Loop over elem/nodes in the workset, and copy over to Thyra 
    const int num_elems = eff_press_hydro.dimension(0);
    const int num_nodes = eff_press_hydro.dimension(1);
    const auto& ElNodeID = hydro_disc->getWsElNodeID()[ib];
    for (int ielem=0; ielem<num_elems; ++ielem) {
      for (int inode=0; inode<num_nodes; ++inode) {
        const GO node_gid = ElNodeID[ielem][inode];

        // Only process local nodes, ghosted ones will be imported
        if (Albany::locallyOwnedComponent(Albany::getSpmdVectorSpace(solvers.m_eff_press_hydro->space()),node_gid)) {
          const LO node_lid = Albany::getLocalElement(solvers.m_eff_press_hydro->space(),node_gid);
          eff_press_hydro_thyra_data[node_lid] = eff_press_hydro(ielem,inode);
        }
      }
    }
  }

  // Export
  solvers.m_eff_press_fo->assign(0.0);
  solvers.m_eff_press_cas->scatter(solvers.m_eff_press_hydro,solvers.m_eff_press_fo,Albany::CombineMode::INSERT);

  // Copy from Thyra_Vector's into hydro stk structures
  const auto fo_basal_states  = fo_basal_disc->getStateArrays().elemStateArrays;
  const size_t num_fo_buckets = fo_basal_states.size();
  auto eff_press_fo_thyra_data  = Albany::getNonconstLocalData(solvers.m_eff_press_fo);
  for (size_t ib=0; ib<num_fo_buckets; ++ib) {
    // Get arrays from mesh
    const auto eff_press_fo = fo_basal_states[ib].at(effPressName);

    // Loop over elem/nodes in the workset, and copy over to Thyra 
    const int num_elems = eff_press_fo.dimension(0);
    const int num_nodes = eff_press_fo.dimension(1);
    const auto& ElNodeID = fo_basal_disc->getWsElNodeID()[ib];
    for (int ielem=0; ielem<num_elems; ++ielem) {
      for (int inode=0; inode<num_nodes; ++inode) {
        const GO node_gid = ElNodeID[ielem][inode];
        const LO node_lid = Albany::getLocalElement(solvers.m_eff_press_fo->space(),node_gid);
        eff_press_fo (ielem,inode) = eff_press_fo_thyra_data[node_lid] ;
      }
    }
  }
}
