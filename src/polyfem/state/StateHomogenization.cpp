#include <polyfem/State.hpp>
#include <polyfem/solver/NLProblem.hpp>

#include <polyfem/utils/StringUtils.hpp>
#include <polyfem/utils/MaybeParallelFor.hpp>

#include <polyfem/solver/NonlinearSolver.hpp>
#include <polyfem/solver/LBFGSSolver.hpp>
#include <polyfem/solver/SparseNewtonDescentSolver.hpp>

#include <polyfem/solver/forms/ContactForm.hpp>
#include <polyfem/solver/forms/ElasticForm.hpp>
#include <polyfem/solver/forms/FrictionForm.hpp>
#include <polyfem/solver/forms/LaggedRegForm.hpp>

#include <polysolve/LinearSolver.hpp>
#include <polysolve/FEMSolver.hpp>
#include <unsupported/Eigen/SparseExtra>

#include <ipc/ipc.hpp>

// map BroadPhaseMethod values to JSON as strings
namespace ipc
{
	NLOHMANN_JSON_SERIALIZE_ENUM(
		ipc::BroadPhaseMethod,
		{{ipc::BroadPhaseMethod::HASH_GRID, "hash_grid"}, // also default
		 {ipc::BroadPhaseMethod::HASH_GRID, "HG"},
		 {ipc::BroadPhaseMethod::BRUTE_FORCE, "brute_force"},
		 {ipc::BroadPhaseMethod::BRUTE_FORCE, "BF"},
		 {ipc::BroadPhaseMethod::SPATIAL_HASH, "spatial_hash"},
		 {ipc::BroadPhaseMethod::SPATIAL_HASH, "SH"},
		 {ipc::BroadPhaseMethod::SWEEP_AND_TINIEST_QUEUE, "sweep_and_tiniest_queue"},
		 {ipc::BroadPhaseMethod::SWEEP_AND_TINIEST_QUEUE, "STQ"},
		 {ipc::BroadPhaseMethod::SWEEP_AND_TINIEST_QUEUE_GPU, "sweep_and_tiniest_queue_gpu"},
		 {ipc::BroadPhaseMethod::SWEEP_AND_TINIEST_QUEUE_GPU, "STQ_GPU"}})
} // namespace ipc

namespace polyfem {

using namespace assembler;
using namespace mesh;
using namespace solver;
using namespace utils;
using namespace quadrature;

namespace
{
	template <typename ProblemType>
	std::shared_ptr<cppoptlib::NonlinearSolver<ProblemType>> make_nl_homo_solver(const json &solver_args)
	{
		const std::string name = solver_args["nonlinear"]["solver"];
		if (name == "newton" || name == "Newton")
		{
			return std::make_shared<cppoptlib::SparseNewtonDescentSolver<ProblemType>>(
				solver_args["nonlinear"], solver_args["linear"]);
		}
		else if (name == "lbfgs" || name == "LBFGS" || name == "L-BFGS")
		{
			return std::make_shared<cppoptlib::LBFGSSolver<ProblemType>>(solver_args["nonlinear"]);
		}
		else
		{
			throw std::invalid_argument(fmt::format("invalid nonlinear solver type: {}", name));
		}
	}

    Eigen::MatrixXd generate_linear_field(const State &state, const Eigen::MatrixXd &grad)
    {
        const int problem_dim = grad.rows();
        const int dim = state.mesh->dimension();
        assert(dim == grad.cols());

        Eigen::MatrixXd func(state.n_bases * problem_dim, 1);
        func.setZero();

        for (int i = 0; i < state.n_bases; i++)
        {
            func.block(i * problem_dim, 0, problem_dim, 1) = grad * state.mesh_nodes->node_position(i).transpose();
        }

        return func;
    }
}

void State::solve_homogenized_field(const Eigen::MatrixXd &disp_grad, const Eigen::MatrixXd &target, Eigen::MatrixXd &sol_)
{
    if (formulation() != "NeoHookean" && formulation() != "LinearElasticity")
    {
        log_and_throw_error("Nonlinear homogenization only supports NeoHookean and linear elasticity!");
    }

    const int dim = mesh->dimension();
    const int ndof = n_bases * dim;

    if (sol_.rows() != ndof || sol_.cols() != 1)
        sol_.setZero(ndof, 1);

    std::vector<std::shared_ptr<Form>> forms;

    std::shared_ptr<ElasticForm> elastic_form = std::make_shared<ElasticForm>(
        n_bases, bases, geom_bases(),
        assembler, ass_vals_cache,
        formulation(),
        problem->is_time_dependent() ? args["time"]["dt"].get<double>() : 0.0,
        mesh->is_volume());
    forms.push_back(elastic_form);

    std::shared_ptr<LaggedRegForm> lag_form = nullptr;
    if (target.size() == sol_.size())
    {
        lag_form = std::make_shared<LaggedRegForm>(1);
        lag_form->init_lagging(target);
        lag_form->disable();
        forms.push_back(lag_form);
    }

    std::shared_ptr<ContactForm> contact_form = nullptr;
    std::shared_ptr<FrictionForm> friction_form = nullptr;
    if (args["contact"]["enabled"])
    {

        const bool use_adaptive_barrier_stiffness = !args["solver"]["contact"]["barrier_stiffness"].is_number();

        contact_form = std::make_shared<ContactForm>(
            collision_mesh, boundary_nodes_pos,
            args["contact"]["dhat"],
            avg_mass,
            use_adaptive_barrier_stiffness,
            /*is_time_dependent=*/solve_data.time_integrator != nullptr,
            args["solver"]["contact"]["CCD"]["broad_phase"],
            args["solver"]["contact"]["CCD"]["tolerance"],
            args["solver"]["contact"]["CCD"]["max_iterations"]);

        if (use_adaptive_barrier_stiffness)
        {
            contact_form->set_weight(1);
            logger().debug("Using adaptive barrier stiffness");
        }
        else
        {
            contact_form->set_weight(args["solver"]["contact"]["barrier_stiffness"]);
            logger().debug("Using fixed barrier stiffness of {}", contact_form->barrier_stiffness());
        }

        forms.push_back(contact_form);

        // ----------------------------------------------------------------

        if (args["contact"]["friction_coefficient"].get<double>() != 0)
        {
            friction_form = std::make_shared<FrictionForm>(
                collision_mesh,
                boundary_nodes_pos,
                args["contact"]["epsv"],
                args["contact"]["friction_coefficient"],
                args["contact"]["dhat"],
                args["solver"]["contact"]["CCD"]["broad_phase"],
                args.value("/time/dt"_json_pointer, 1.0), // dt=1.0 if static
                *contact_form,
                args["solver"]["contact"]["friction_iterations"]);
            forms.push_back(friction_form);
        }
    }

    std::shared_ptr<NLProblem> homo_problem = std::make_shared<NLProblem>(
        ndof,
        formulation(),
        boundary_nodes,
        local_boundary,
        n_boundary_samples(),
        *solve_data.rhs_assembler, *this, 0, forms);
    
    Eigen::VectorXd macro_field = generate_linear_field(*this, disp_grad);
    homo_problem->set_disp_offset(macro_field);

    std::shared_ptr<cppoptlib::NonlinearSolver<NLProblem>> nl_solver = make_nl_homo_solver<NLProblem>(args["solver"]);
    
    Eigen::VectorXd tmp_sol;
    if (lag_form)
    {
        for (auto form : forms)
            form->set_weight(1e-8); // do not disable so it detects nan
        lag_form->set_weight(1);
        lag_form->enable();

        tmp_sol = homo_problem->full_to_reduced(sol_);
        homo_problem->init(homo_problem->reduced_to_full(tmp_sol));
        nl_solver->minimize(*homo_problem, tmp_sol);
        sol_ = homo_problem->reduced_to_full(tmp_sol) - macro_field;

        for (auto form : forms)
            form->set_weight(1);
        lag_form->disable();
    }
    
    tmp_sol = homo_problem->full_to_reduced(sol_);
    export_data(homo_problem->reduced_to_full(tmp_sol), Eigen::MatrixXd());
    
    homo_problem->init(homo_problem->reduced_to_full(tmp_sol));
    nl_solver->minimize(*homo_problem, tmp_sol);
    sol_ = homo_problem->reduced_to_full(tmp_sol);

    // static int index = 0;
    // StiffnessMatrix H;
    // homo_problem->hessian(tmp_sol, H);
    // Eigen::saveMarket(H, "H" + std::to_string(index) + ".mat");
    // index++;
}

}