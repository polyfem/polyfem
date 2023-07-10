#include <polyfem/State.hpp>
#include <polyfem/solver/NLHomoProblem.hpp>
#include <polyfem/assembler/Mass.hpp>
#include <polyfem/utils/StringUtils.hpp>
#include <polyfem/utils/MaybeParallelFor.hpp>

#include <polyfem/solver/NonlinearSolver.hpp>
#include <polyfem/solver/LBFGSSolver.hpp>
#include <polyfem/solver/SparseNewtonDescentSolver.hpp>
#include <polyfem/solver/GradientDescentSolver.hpp>

#include <polyfem/solver/forms/PeriodicContactForm.hpp>
#include <polyfem/solver/forms/MacroStrainALForm.hpp>
#include <polyfem/solver/forms/MacroStrainLagrangianForm.hpp>

#include <polysolve/FEMSolver.hpp>
#include <unsupported/Eigen/SparseExtra>

#include <polyfem/io/OBJWriter.hpp>
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

void State::solve_homogenized_field(Eigen::MatrixXd &disp_grad, Eigen::MatrixXd &sol_, const std::vector<int> &fixed_entry, bool for_bistable)
{
    const int dim = mesh->dimension();
    const int ndof = n_bases * dim;

    solver::SolveData solve_data_tmp = solve_data;
    const std::vector<std::shared_ptr<Form>> forms = solve_data_tmp.init_forms(
        // General
        mesh->dimension(), 0,
        // Elastic form
        n_bases, bases, geom_bases(), *assembler, ass_vals_cache, mass_ass_vals_cache,
        // Body form
        n_pressure_bases, boundary_nodes, local_boundary, local_neumann_boundary,
        n_boundary_samples(), rhs, Eigen::VectorXd::Zero(ndof) /* only to set neumann BC, not used*/, mass_matrix_assembler->density(),
        // Inertia form
        args["solver"]["ignore_inertia"], mass, nullptr,
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
        args["optimization"]["enabled"],
        // Periodic contact
        args["contact"]["periodic"], tiled_to_single,
        // Friction form
        args["contact"]["friction_coefficient"],
        args["contact"]["epsv"],
        args["solver"]["contact"]["friction_iterations"],
        // Rayleigh damping form
        args["solver"]["rayleigh_damping"]);
    
    solve_data_tmp.named_forms().at("augmented_lagrangian_lagr")->disable();
    solve_data_tmp.named_forms().at("augmented_lagrangian_lagr")->set_weight(0);
    solve_data_tmp.named_forms().at("augmented_lagrangian_penalty")->disable();
    solve_data_tmp.named_forms().at("augmented_lagrangian_penalty")->set_weight(0);

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
        *solve_data_tmp.rhs_assembler, *this, 0, forms, solve_symmetric_flag);
    if (solve_data_tmp.periodic_contact_form)
        homo_problem->add_form(solve_data_tmp.periodic_contact_form);
    solve_data_tmp.nl_problem = homo_problem;

    if (args["optimization"]["enabled"])
    {
        solve_data = solve_data_tmp;
    }

    std::shared_ptr<cppoptlib::NonlinearSolver<NLHomoProblem>> nl_solver = make_nl_homo_solver<NLHomoProblem>(args["solver"]);
    
    bool force_al = args["solver"]["augmented_lagrangian"]["force"];

    Eigen::VectorXd extended_sol;
    extended_sol.setZero(ndof + dim * dim);
    if (!force_al)
        extended_sol.tail(dim * dim) = utils::flatten(disp_grad); // if not forcing AL, start solve from pure compression
    
    if (initial_guess.size() == extended_sol.size())
        extended_sol = initial_guess;

    Eigen::MatrixXd disp_grad_out = disp_grad;
    {
        Eigen::VectorXi al_indices;
        Eigen::VectorXd al_values;
        // from full to symmetric indices
        {
            al_indices.setZero(fixed_entry.size());
            for (int i = 0; i < fixed_entry.size(); i++)
                al_indices(i) = fixed_entry[i] + homo_problem->full_size();
            
            al_values = utils::flatten(disp_grad)(fixed_entry);
        }
        logger().debug("AL indices: {}, AL values: {}", al_indices.transpose(), al_values.transpose());
        std::shared_ptr<MacroStrainALForm> al_form = std::make_shared<MacroStrainALForm>(al_indices, al_values);
        std::shared_ptr<MacroStrainLagrangianForm> lagr_form = std::make_shared<MacroStrainLagrangianForm>(al_indices, al_values);
        homo_problem->add_form(al_form);
        homo_problem->add_form(lagr_form);

        const double initial_weight = args["solver"]["augmented_lagrangian"]["initial_weight"];
        const double max_weight = args["solver"]["augmented_lagrangian"]["max_weight"];
        const double eta_tol = args["solver"]["augmented_lagrangian"]["eta"];
        const double scaling = args["solver"]["augmented_lagrangian"]["scaling"];
        const int max_al_steps = args["solver"]["augmented_lagrangian"]["max_solver_iters"];
        const double error_tol = args["solver"]["augmented_lagrangian"]["error"];
        double al_weight = initial_weight;

        Eigen::VectorXd tmp_sol = homo_problem->extended_to_reduced(extended_sol);
        const Eigen::VectorXd initial_sol = tmp_sol;
        const double initial_error = (extended_sol(al_indices).array() - al_values.array()).matrix().squaredNorm();
        double current_error = initial_error;
        
        // try to enforce fixed values on macro strain
        extended_sol(al_indices) = al_values;
        Eigen::VectorXd reduced_sol = homo_problem->extended_to_reduced(extended_sol);

        homo_problem->line_search_begin(tmp_sol, reduced_sol);
        int al_steps = 0;
		while (force_al
			   || !std::isfinite(homo_problem->value(reduced_sol))
			   || !homo_problem->is_step_valid(tmp_sol, reduced_sol)
			   || !homo_problem->is_step_collision_free(tmp_sol, reduced_sol)
               || current_error > error_tol)
        {
            force_al = false;
            homo_problem->line_search_end();

            al_form->set_weight(al_weight);
            logger().info("Solving AL Problem with weight {}", al_weight);

            homo_problem->init(tmp_sol);
			try
			{
				nl_solver->minimize(*homo_problem, tmp_sol);
			}
			catch (const std::runtime_error &e)
			{
                logger().error("AL solve failed!");
                export_data(homo_problem->reduced_to_full(tmp_sol), Eigen::MatrixXd());
			}

            extended_sol = homo_problem->reduced_to_extended(tmp_sol);
            logger().debug("Current macro strain: {}", extended_sol.tail(dim * dim));

            current_error = (extended_sol(al_indices).array() - al_values.array()).matrix().squaredNorm();
            const double eta = 1 - sqrt(current_error / initial_error);

            logger().info("Current eta = {}, current error = {}, initial error = {}", eta, current_error, initial_error);

			if (eta < eta_tol && al_weight < max_weight)
            	al_weight *= scaling;
            else
                lagr_form->update_lagrangian(extended_sol, al_weight);
            
            if (eta <= 0)
            {
                args["solver"]["augmented_lagrangian"]["initial_weight"] = args["solver"]["augmented_lagrangian"]["initial_weight"].get<double>() * scaling;
                {
                    json tmp = json::object();
                    tmp["/solver/augmented_lagrangian/initial_weight"_json_pointer] = args["solver"]["augmented_lagrangian"]["initial_weight"];
                    in_args.merge_patch(tmp);
                }
                logger().warn("AL weight too small, increase weight and revert solution, new initial weight is {}", args["solver"]["augmented_lagrangian"]["initial_weight"]);
                tmp_sol = initial_sol;
            }
            else if (current_error <= error_tol && al_steps == 0)
            {
                args["solver"]["augmented_lagrangian"]["initial_weight"] = args["solver"]["augmented_lagrangian"]["initial_weight"].get<double>() / scaling;
                {
                    json tmp = json::object();
                    tmp["/solver/augmented_lagrangian/initial_weight"_json_pointer] = args["solver"]["augmented_lagrangian"]["initial_weight"];
                    in_args.merge_patch(tmp);
                }
                logger().warn("AL weight too large, decrease initial weight to {}", args["solver"]["augmented_lagrangian"]["initial_weight"]);
            }

            // try to enforce fixed values on macro strain
            extended_sol(al_indices) = al_values;
            reduced_sol = homo_problem->extended_to_reduced(extended_sol);

            homo_problem->line_search_begin(tmp_sol, reduced_sol);

			if (al_steps++ >= max_al_steps)
				log_and_throw_error(fmt::format("Unable to solve AL problem, out of iterations {} (current weight = {}), stopping", max_al_steps, al_weight));
        }
        homo_problem->line_search_end();
        // extended_sol = homo_problem->reduced_to_extended(tmp_sol);
        // disp_grad_out = utils::unflatten(extended_sol.tail(dim * dim), dim);
        {
            al_weight = 1;
            for (auto form : forms)
                form->set_weight(al_weight);
            if (solve_data_tmp.periodic_contact_form)
                solve_data_tmp.periodic_contact_form->set_weight(al_weight);
            al_form->set_weight(1 - al_weight);
            al_form->disable();
            lagr_form->set_weight(1 - al_weight);
            lagr_form->disable();
        }
    }

    homo_problem->set_fixed_entry(fixed_entry, utils::flatten(disp_grad));

    Eigen::VectorXd reduced_sol = homo_problem->extended_to_reduced(extended_sol);

    // const Eigen::MatrixXd displaced = periodic_collision_mesh.displace_vertices(utils::unflatten(solve_data_tmp.periodic_contact_form->single_to_tiled(homo_problem->reduced_to_extended(reduced_sol)), dim));

    // static int debug_id = 0;
    // io::OBJWriter::write(
    //     "tiled" + std::to_string(debug_id++) + ".obj", displaced,
    //     periodic_collision_mesh.edges(), periodic_collision_mesh.faces());

    homo_problem->init(reduced_sol);
    nl_solver->minimize(*homo_problem, reduced_sol);
    
    if (for_bistable)
    {
        homo_problem->set_fixed_entry({}, utils::flatten(disp_grad));
        nl_solver->minimize(*homo_problem, reduced_sol);
    }

    disp_grad = homo_problem->reduced_to_disp_grad(reduced_sol);
    logger().info("displacement grad {}", utils::flatten(disp_grad).transpose());

    if (args["optimization"]["enabled"])
        cache_transient_adjoint_quantities(0, homo_problem->reduced_to_full(reduced_sol), disp_grad);
    
    sol_ = homo_problem->reduced_to_extended(reduced_sol);
    
    // initial_guess = homo_problem->reduced_to_extended(reduced_sol);

    // static int index = 0;
    // StiffnessMatrix H;
    // homo_problem->hessian(tmp_sol, H);
    // Eigen::saveMarket(H, "H" + std::to_string(index) + ".mat");
    // index++;
}

}