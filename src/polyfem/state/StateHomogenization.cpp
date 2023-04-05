#include <polyfem/State.hpp>
#include <polyfem/solver/NLHomoProblem.hpp>

#include <polyfem/utils/StringUtils.hpp>
#include <polyfem/utils/MaybeParallelFor.hpp>

#include <polyfem/solver/NonlinearSolver.hpp>
#include <polyfem/solver/LBFGSSolver.hpp>
#include <polyfem/solver/SparseNewtonDescentSolver.hpp>
#include <polyfem/solver/GradientDescentSolver.hpp>

#include <polyfem/solver/forms/PeriodicContactForm.hpp>
#include <polyfem/solver/forms/MacroStrainALForm.hpp>
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

    bool solve_symmetric_flag = false;
    {
        for (int i = 0; i < dim; i++)
        {
            for (int j = 0; j < i; j++)
            {
                if (std::find(fixed_entry.begin(), fixed_entry.end(), i + j * dim) == fixed_entry.end() &&
                    std::find(fixed_entry.begin(), fixed_entry.end(), j + i * dim) == fixed_entry.end())
                {
                    logger().info("Strain entry [{},{}] and [{},{}] are not fixed, solve for symmetric strain...", i, j, j, i);
                    solve_symmetric_flag = true;
                    break;
                }
            }
        }
        if (solve_symmetric_flag)
        {
            if ((disp_grad - disp_grad.transpose()).norm() > 1e-10)
                log_and_throw_error("Macro strain is not symmetric!");
        }
    }

    std::vector<int> boundary_nodes_tmp = boundary_nodes;
    full_to_periodic(boundary_nodes_tmp);

    std::shared_ptr<NLHomoProblem> homo_problem = std::make_shared<NLHomoProblem>(
        ndof,
        boundary_nodes_tmp,
        local_boundary,
        n_boundary_samples(),
        *solve_data_tmp.rhs_assembler, *this, 0, forms, solve_symmetric_flag, solve_data_tmp.periodic_contact_form);
    solve_data_tmp.nl_problem = homo_problem;

    if (args["optimization"]["enabled"])
    {
        solve_data = solve_data_tmp;
    }

    std::shared_ptr<cppoptlib::NonlinearSolver<NLHomoProblem>> nl_solver = make_nl_homo_solver<NLHomoProblem>(args["solver"]);

    bool force_al = args["solver"]["augmented_lagrangian"]["force"];
    Eigen::VectorXd extended_sol(sol_.size() + dim * dim);
    extended_sol << sol_, Eigen::VectorXd::Zero(dim * dim);
    Eigen::MatrixXd disp_grad_out = disp_grad;
    if (force_al)
    {
        Eigen::VectorXi al_indices;
        Eigen::VectorXd al_values;
        // from full to symmetric indices
        {
            Eigen::VectorXd fixed_mask;
            fixed_mask.setZero(dim * dim);
            fixed_mask(fixed_entry).setOnes();
            fixed_mask = homo_problem->macro_full_to_mid(fixed_mask);
            
            al_indices.setZero((int)std::round(fixed_mask.sum()));
            for (int i = 0, j = 0; i < fixed_mask.size(); i++)
                if (abs(fixed_mask(i)) > 1e-8)
                    al_indices(j++) = i;
            
            al_values = homo_problem->macro_full_to_mid(utils::flatten(disp_grad));
        }
        std::shared_ptr<MacroStrainALForm> al_form = std::make_shared<MacroStrainALForm>(dim, al_indices, al_values);
        homo_problem->set_al_form(al_form);

        const double initial_weight = args["solver"]["augmented_lagrangian"]["initial_weight"];
        const double scaling = args["solver"]["augmented_lagrangian"]["scaling"];
        const int max_al_steps = args["solver"]["augmented_lagrangian"]["max_steps"];
        double al_weight = initial_weight;

        Eigen::VectorXd tmp_sol = homo_problem->full_to_reduced(sol_, Eigen::MatrixXd::Zero(dim, dim));
        Eigen::VectorXd reduced_sol = tmp_sol;
        for (int i = 0; i < al_indices.size(); i++)
            reduced_sol(al_indices(i) + homo_problem->reduced_size()) = al_values(i);

        homo_problem->line_search_begin(tmp_sol, reduced_sol);
        int al_steps = 0;
        while (force_al
            || !std::isfinite(homo_problem->value(tmp_sol))
            || !homo_problem->is_step_valid(tmp_sol, reduced_sol)
            || !homo_problem->is_step_collision_free(tmp_sol, reduced_sol))
        {
            force_al = false;
            homo_problem->line_search_end();

            {
                for (auto form : forms)
                    form->set_weight(al_weight);
                solve_data_tmp.periodic_contact_form->set_weight(al_weight);
                al_form->set_weight(1 - al_weight);
            }
            logger().debug("Solving AL Problem with weight {}", al_weight);

            homo_problem->init(tmp_sol);
            nl_solver->minimize(*homo_problem, tmp_sol);

            reduced_sol = tmp_sol;
            for (int i = 0; i < al_indices.size(); i++)
                reduced_sol(al_indices(i) + homo_problem->reduced_size()) = al_values(i);
            
            logger().debug("Current macro strain: {}", tmp_sol.tail(homo_problem->macro_reduced_size()));

            homo_problem->line_search_begin(tmp_sol, reduced_sol);

            al_weight /= scaling;
			if (al_steps >= max_al_steps)
			{
				log_and_throw_error(fmt::format("Unable to solve AL problem, out of iterations {} (current weight = {}), stopping", max_al_steps, al_weight));
				break;
			}

            ++al_steps;
        }
        homo_problem->line_search_end();
        extended_sol = homo_problem->reduced_to_extended(tmp_sol);
        disp_grad_out = utils::unflatten(extended_sol.tail(dim * dim), dim);
        {
            al_weight = 1;
            for (auto form : forms)
                form->set_weight(al_weight);
            solve_data_tmp.periodic_contact_form->set_weight(al_weight);
            al_form->set_weight(1 - al_weight);
            al_form->disable();
        }
    }

    homo_problem->set_fixed_entry(fixed_entry, utils::flatten(disp_grad));

    Eigen::VectorXd reduced_sol = homo_problem->NLProblem::full_to_reduced(extended_sol.head(ndof));
    Eigen::VectorXd tail = homo_problem->macro_full_to_reduced(utils::flatten(disp_grad_out));
    reduced_sol.tail(tail.size()) = tail;

    homo_problem->init(reduced_sol);
    nl_solver->minimize(*homo_problem, reduced_sol);
    
    if (for_bistable)
    {
        homo_problem->set_fixed_entry({}, utils::flatten(disp_grad_out));
        nl_solver->minimize(*homo_problem, reduced_sol);
    }

    sol_ = homo_problem->reduced_to_full(reduced_sol);
    disp_grad_out = homo_problem->reduced_to_disp_grad(reduced_sol);

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