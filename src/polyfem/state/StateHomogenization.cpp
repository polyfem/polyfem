#include <polyfem/State.hpp>
#include <polyfem/solver/NLHomoProblem.hpp>
#include <polyfem/assembler/Mass.hpp>
#include <polyfem/utils/StringUtils.hpp>
#include <polyfem/utils/MaybeParallelFor.hpp>
#include <polyfem/utils/Timer.hpp>

#include <polysolve/nonlinear/Solver.hpp>
// #include <polyfem/solver/LBFGSSolver.hpp>
// #include <polyfem/solver/SparseNewtonDescentSolver.hpp>
// #include <polyfem/solver/GradientDescentSolver.hpp>

#include <polyfem/assembler/ViscousDamping.hpp>
#include <polyfem/solver/forms/PeriodicContactForm.hpp>
#include <polyfem/solver/forms/MacroStrainALForm.hpp>
#include <polyfem/solver/forms/MacroStrainLagrangianForm.hpp>

// #include <polysolve/FEMSolver.hpp>
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

void State::init_homogenization_solve(const std::vector<int> &fixed_entry, const double t)
{
    const int dim = mesh->dimension();
    const int ndof = n_bases * dim;

    const std::vector<std::shared_ptr<Form>> forms = solve_data.init_forms(
        // General
		units,
        mesh->dimension(), t,
        // Elastic form
        n_bases, bases, geom_bases(), *assembler, ass_vals_cache, mass_ass_vals_cache,
        // Body form
        n_pressure_bases, boundary_nodes, local_boundary, local_neumann_boundary,
        n_boundary_samples(), rhs, Eigen::VectorXd::Zero(ndof) /* only to set neumann BC, not used*/, mass_matrix_assembler->density(),
        // Inertia form
        args.value("/time/quasistatic"_json_pointer, true), mass,
        nullptr,
        // Lagged regularization form
        args["solver"]["advanced"]["lagged_regularization_weight"],
        args["solver"]["advanced"]["lagged_regularization_iterations"],
        // Augmented lagrangian form
        obstacle.ndof(),
        // Contact form
        args["contact"]["enabled"], args["contact"]["periodic"].get<bool>() ? periodic_collision_mesh : collision_mesh, args["contact"]["dhat"],
        avg_mass, args["contact"]["use_convergent_formulation"],
        args["solver"]["contact"]["barrier_stiffness"],
        args["solver"]["contact"]["CCD"]["broad_phase"],
        args["solver"]["contact"]["CCD"]["tolerance"],
        args["solver"]["contact"]["CCD"]["max_iterations"],
        optimization_enabled == CacheLevel::Derivatives,
        // Periodic contact
        args["contact"]["periodic"], tiled_to_single,
        // Friction form
        args["contact"]["friction_coefficient"],
        args["contact"]["epsv"],
        args["solver"]["contact"]["friction_iterations"],
        // Rayleigh damping form
        args["solver"]["rayleigh_damping"]);
    
    solve_data.named_forms().at("augmented_lagrangian_lagr")->disable();
    solve_data.named_forms().at("augmented_lagrangian_lagr")->set_weight(0);
    solve_data.named_forms().at("augmented_lagrangian_penalty")->disable();
    solve_data.named_forms().at("augmented_lagrangian_penalty")->set_weight(0);

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
    }

    std::shared_ptr<NLHomoProblem> homo_problem = std::make_shared<NLHomoProblem>(
        ndof,
        boundary_nodes,
        local_boundary,
        n_boundary_samples(),
        *solve_data.rhs_assembler, *this, t, forms, solve_symmetric_flag);
    if (solve_data.periodic_contact_form)
        homo_problem->add_form(solve_data.periodic_contact_form);
    solve_data.nl_problem = homo_problem;
}

void State::solve_homogenization_step(Eigen::MatrixXd &sol, const Eigen::MatrixXd &disp_grad, const std::vector<int> &fixed_entry, const int t, bool adaptive_initial_weight)
{
    const int dim = mesh->dimension();
    const int ndof = n_bases * dim;

    auto homo_problem = std::dynamic_pointer_cast<NLHomoProblem>(solve_data.nl_problem);

    if (homo_problem->has_symmetry_constraint() && (disp_grad - disp_grad.transpose()).norm() > 1e-8)
        log_and_throw_error("Macro strain is not symmetric!");

    std::shared_ptr<polysolve::nonlinear::Solver> nl_solver = make_nl_solver();

    Eigen::VectorXd extended_sol;
    extended_sol.setZero(ndof + dim * dim);
    
    if (sol.size() == extended_sol.size())
        extended_sol = sol;

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
        bool force_al = true;
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
                if (adaptive_initial_weight)
                {
                    args["solver"]["augmented_lagrangian"]["initial_weight"] = args["solver"]["augmented_lagrangian"]["initial_weight"].get<double>() * scaling;
                    {
                        json tmp = json::object();
                        tmp["/solver/augmented_lagrangian/initial_weight"_json_pointer] = args["solver"]["augmented_lagrangian"]["initial_weight"];
                    }
                    logger().warn("AL weight too small, increase weight and revert solution, new initial weight is {}", args["solver"]["augmented_lagrangian"]["initial_weight"].get<double>());
                }
                tmp_sol = initial_sol;
            }
            else if (current_error <= error_tol && al_steps == 0)
            {
                args["solver"]["augmented_lagrangian"]["initial_weight"] = args["solver"]["augmented_lagrangian"]["initial_weight"].get<double>() / scaling;
                {
                    json tmp = json::object();
                    tmp["/solver/augmented_lagrangian/initial_weight"_json_pointer] = args["solver"]["augmented_lagrangian"]["initial_weight"];
                }
                logger().warn("AL weight too large, decrease initial weight to {}", args["solver"]["augmented_lagrangian"]["initial_weight"].get<double>());
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
            for (auto& [name, form] : solve_data.named_forms())
                if (form)
                    form->set_weight(al_weight);
            if (solve_data.periodic_contact_form)
                solve_data.periodic_contact_form->set_weight(al_weight);
            al_form->set_weight(1 - al_weight);
            al_form->disable();
            lagr_form->set_weight(1 - al_weight);
            lagr_form->disable();
        }
    }

    homo_problem->set_fixed_entry(fixed_entry, utils::flatten(disp_grad));

    Eigen::VectorXd reduced_sol = homo_problem->extended_to_reduced(extended_sol);

    homo_problem->init(reduced_sol);
    nl_solver->minimize(*homo_problem, reduced_sol);

    logger().info("displacement grad {}", extended_sol.tail(dim * dim).transpose());

    if (optimization_enabled != CacheLevel::None)
        cache_transient_adjoint_quantities(t, homo_problem->reduced_to_full(reduced_sol), utils::unflatten(extended_sol.tail(dim * dim), dim));
    
    sol = homo_problem->reduced_to_extended(reduced_sol);
}

void State::solve_homogenization(const int time_steps, const double t0, const double dt, const std::vector<int> &fixed_entry, Eigen::MatrixXd &sol)
{
    bool is_static = !is_param_valid(args, "time");
    if (!is_static && !args["time"]["quasistatic"])
        log_and_throw_error("Transient homogenization can only do quasi-static!");
    
    init_homogenization_solve(fixed_entry, t0);

    Eigen::MatrixXd extended_sol;
    for (int t = 0; t <= time_steps; ++t)
    {
        double forward_solve_time = 0, remeshing_time = 0, global_relaxation_time = 0;

        const Eigen::MatrixXd disp_grad = macro_strain_constraint.eval(mesh->dimension(), t0 + dt * t);
        {
            POLYFEM_SCOPED_TIMER(forward_solve_time);
            solve_homogenization_step(extended_sol, disp_grad, fixed_entry, t, false);
        }
        sol = extended_sol.topRows(extended_sol.size()-disp_grad.size()) + io::Evaluator::generate_linear_field(n_bases, mesh_nodes, utils::unflatten(extended_sol.bottomRows(disp_grad.size()), mesh->dimension()));
        
        if (is_static)
            return;

        // Always save the solution for consistency
        save_timestep(t0 + dt * t, t, t0, dt, sol, Eigen::MatrixXd()); // no pressure

        {
            POLYFEM_SCOPED_TIMER("Update quantities");

        //     solve_data.time_integrator->update_quantities(sol);

            solve_data.nl_problem->update_quantities(t0 + (t + 1) * dt, sol);

            solve_data.update_dt();
            solve_data.update_barrier_stiffness(sol);
        }

        logger().info("{}/{}  t={}", t, time_steps, t0 + dt * t);

        // const std::string rest_mesh_path = args["output"]["data"]["rest_mesh"].get<std::string>();
        // if (!rest_mesh_path.empty())
        // {
        //     Eigen::MatrixXd V;
        //     Eigen::MatrixXi F;
        //     build_mesh_matrices(V, F);
        //     io::MshWriter::write(
        //         resolve_output_path(fmt::format(args["output"]["data"]["rest_mesh"], t)),
        //         V, F, mesh->get_body_ids(), mesh->is_volume(), /*binary=*/true);
        // }

        // const std::string &state_path = resolve_output_path(fmt::format(args["output"]["data"]["state"], t));
        // if (!state_path.empty())
        //     solve_data.time_integrator->save_state(state_path);

        // save restart file
        save_restart_json(t0, dt, t);
        // stats_csv.write(t, forward_solve_time, remeshing_time, global_relaxation_time, sol);
    }
}

}