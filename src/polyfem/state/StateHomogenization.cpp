#include <polyfem/State.hpp>
#include <polyfem/solver/NLHomoProblem.hpp>

#include <polyfem/utils/StringUtils.hpp>
#include <polyfem/utils/MaybeParallelFor.hpp>

#include <polyfem/solver/NonlinearSolver.hpp>
#include <polyfem/solver/LBFGSSolver.hpp>
#include <polyfem/solver/SparseNewtonDescentSolver.hpp>
#include <polyfem/solver/GradientDescentSolver.hpp>

#include <polyfem/solver/forms/PeriodicContactForm.hpp>
#include <polyfem/solver/forms/ElasticForm.hpp>
#include <polyfem/solver/forms/FrictionForm.hpp>

#include <polysolve/LinearSolver.hpp>
#include <polysolve/FEMSolver.hpp>
#include <unsupported/Eigen/SparseExtra>

#include <polyfem/io/Evaluator.hpp>

#include <ipc/ipc.hpp>

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
		if (name == "GradientDescent" || name == "gradientdescent" || name == "gradient")
		{
			return std::make_shared<cppoptlib::GradientDescentSolver<ProblemType>>(
				solver_args["nonlinear"], 0);
		}
		else if (name == "newton" || name == "Newton")
		{
			return std::make_shared<cppoptlib::SparseNewtonDescentSolver<ProblemType>>(
				solver_args["nonlinear"], solver_args["linear"], 0);
		}
		else if (name == "lbfgs" || name == "LBFGS" || name == "L-BFGS")
		{
			return std::make_shared<cppoptlib::LBFGSSolver<ProblemType>>(solver_args["nonlinear"], 0);
		}
		else
		{
			throw std::invalid_argument(fmt::format("invalid nonlinear solver type: {}", name));
		}
	}
}

void State::solve_homogenized_field(const Eigen::MatrixXd &disp_grad, Eigen::MatrixXd &sol_, const std::vector<int> &fixed_entry, bool for_bistable)
{
    const int dim = mesh->dimension();
    const int ndof = n_bases * dim;

    Eigen::MatrixXd pressure;
    init_solve(sol_, pressure);

    solver::SolveData solve_data_tmp = solve_data;
    const std::vector<std::shared_ptr<Form>> forms = solve_data_tmp.init_forms(
        // General
        mesh->dimension(), 0,
        // Elastic form
        n_bases, bases, geom_bases(), assembler, ass_vals_cache, formulation(),
        // Body form
        n_pressure_bases, boundary_nodes, local_boundary, local_neumann_boundary,
        n_boundary_samples(), rhs, sol_,
        // Inertia form
        args["solver"]["ignore_inertia"], mass,
        // Lagged regularization form
        args["solver"]["advanced"]["lagged_regularization_weight"],
        args["solver"]["advanced"]["lagged_regularization_iterations"],
        // Augmented lagrangian form
        obstacle,
        // Contact form
        args["contact"]["enabled"], args["contact"]["periodic"].get<bool>() ? periodic_collision_mesh : collision_mesh, args["contact"]["dhat"],
        avg_mass, args["contact"]["use_convergent_formulation"],
        args["solver"]["contact"]["barrier_stiffness"],
        args["solver"]["contact"]["CCD"]["broad_phase"],
        args["solver"]["contact"]["CCD"]["tolerance"],
        args["solver"]["contact"]["CCD"]["max_iterations"],
        // Periodic contact
        args["contact"]["periodic"], tiled_to_periodic,
        // Friction form
        args["contact"]["friction_coefficient"],
        args["contact"]["epsv"],
        args["solver"]["contact"]["friction_iterations"],
        // Rayleigh damping form
        args["solver"]["rayleigh_damping"]);

    std::vector<int> boundary_nodes_tmp = boundary_nodes;
    full_to_periodic(boundary_nodes_tmp);

    std::shared_ptr<NLHomoProblem> homo_problem = std::make_shared<NLHomoProblem>(
        ndof,
        boundary_nodes_tmp,
        local_boundary,
        n_boundary_samples(),
        *solve_data_tmp.rhs_assembler, *this, 0, forms, solve_data_tmp.periodic_contact_form);
    solve_data_tmp.nl_problem = homo_problem;

    if (args["optimization"]["enabled"])
    {
        solve_data = solve_data_tmp;
    }

    homo_problem->set_fixed_entry(fixed_entry);
    {
        bool flag = false;
        for (int i = 0; i < dim; i++)
        {
            for (int j = 0; j < i; j++)
            {
                if (std::find(fixed_entry.begin(), fixed_entry.end(), i + j * dim) == fixed_entry.end() &&
                    std::find(fixed_entry.begin(), fixed_entry.end(), j + i * dim) == fixed_entry.end())
                {
                    logger().info("Strain entry [{},{}] and [{},{}] are not fixed, solve for symmetric strain...", i, j, j, i);
                    flag = true;
                    break;
                }
            }
        }
        if (flag)
        {
            if ((disp_grad - disp_grad.transpose()).norm() > 1e-10)
                log_and_throw_error("Macro strain is not symmetric!");
            homo_problem->set_only_symmetric();
        }
    }

    Eigen::VectorXd tmp_sol = homo_problem->full_to_reduced(sol_, Eigen::MatrixXd::Zero(dim, dim));
    Eigen::VectorXd tail = homo_problem->macro_full_to_reduced(utils::flatten(disp_grad));
    tmp_sol.tail(tail.size()) = tail;

    homo_problem->init(homo_problem->reduced_to_full(tmp_sol));
    std::shared_ptr<cppoptlib::NonlinearSolver<NLHomoProblem>> nl_solver = make_nl_homo_solver<NLHomoProblem>(args["solver"]);
    nl_solver->minimize(*homo_problem, tmp_sol);
    
    if (for_bistable)
    {
        homo_problem->set_fixed_entry({});
        nl_solver->minimize(*homo_problem, tmp_sol);
    }

    sol_ = homo_problem->reduced_to_full(tmp_sol);

    Eigen::MatrixXd disp_grad_out = homo_problem->reduced_to_disp_grad(tmp_sol);

    logger().info("displacement grad {}", utils::flatten(disp_grad_out).transpose());

    if (args["optimization"]["enabled"])
        cache_transient_adjoint_quantities(0, sol_, disp_grad_out);

    // static int index = 0;
    // StiffnessMatrix H;
    // homo_problem->hessian(tmp_sol, H);
    // Eigen::saveMarket(H, "H" + std::to_string(index) + ".mat");
    // index++;
}

}