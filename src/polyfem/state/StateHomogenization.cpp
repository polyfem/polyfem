#include <polyfem/State.hpp>
#include <polyfem/solver/NLHomoProblem.hpp>
#include <polyfem/assembler/Mass.hpp>
#include <polyfem/utils/StringUtils.hpp>
#include <polyfem/utils/MaybeParallelFor.hpp>
#include <polyfem/utils/Timer.hpp>
#include <polysolve/linear/FEMSolver.hpp>
#include <polysolve/nonlinear/Solver.hpp>

#include <polyfem/assembler/ViscousDamping.hpp>
#include <polyfem/solver/forms/PeriodicContactForm.hpp>
#include <polyfem/solver/forms/lagrangian/MacroStrainLagrangianForm.hpp>

#include <unsupported/Eigen/SparseExtra>

#include <polyfem/io/OBJWriter.hpp>
#include <polyfem/io/Evaluator.hpp>

#include <ipc/ipc.hpp>

namespace polyfem
{

	using namespace assembler;
	using namespace mesh;
	using namespace solver;
	using namespace utils;
	using namespace quadrature;

	void State::init_homogenization_solve(const double t)
	{
		const int dim = mesh->dimension();
		const int ndof = n_bases * dim;

		const std::vector<std::shared_ptr<Form>> forms = solve_data.init_forms(
			// General
			units,
			mesh->dimension(), t,
			// Elastic form
			n_bases, bases, geom_bases(), *assembler, ass_vals_cache, mass_ass_vals_cache,
			args["solver"]["advanced"]["jacobian_threshold"], args["solver"]["advanced"]["check_inversion"],
			// Body form
			n_pressure_bases, boundary_nodes, local_boundary, local_neumann_boundary,
			n_boundary_samples(), rhs, Eigen::VectorXd::Zero(ndof) /* only to set neumann BC, not used*/, mass_matrix_assembler->density(),
			// Pressure form
			local_pressure_boundary, local_pressure_cavity, elasticity_pressure_assembler,
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
			optimization_enabled == solver::CacheLevel::Derivatives,
			args["contact"],
			// Homogenization
			macro_strain_constraint,
			// Periodic contact
			args["contact"]["periodic"], periodic_collision_mesh_to_basis, periodic_bc,
			// Friction form
			args["contact"]["friction_coefficient"],
			args["contact"]["epsv"],
			args["solver"]["contact"]["friction_iterations"],
			// Rayleigh damping form
			args["solver"]["rayleigh_damping"]);

		for (const auto &[name, form] : solve_data.named_forms())
		{
			if (name == "augmented_lagrangian")
			{
				form->set_weight(0);
				form->disable();
			}
		}

		bool solve_symmetric_flag = false;
		{
			const auto &fixed_entry = macro_strain_constraint.get_fixed_entry();
			for (int i = 0; i < dim; i++)
			{
				for (int j = 0; j < i; j++)
				{
					if (std::find(fixed_entry.data(), fixed_entry.data() + fixed_entry.size(), i + j * dim) == fixed_entry.data() + fixed_entry.size() && std::find(fixed_entry.data(), fixed_entry.data() + fixed_entry.size(), j + i * dim) == fixed_entry.data() + fixed_entry.size())
					{
						logger().info("Strain entry [{},{}] and [{},{}] are not fixed, solve for symmetric strain...", i, j, j, i);
						solve_symmetric_flag = true;
						break;
					}
				}
				if (solve_symmetric_flag)
					break;
			}
		}

		std::shared_ptr<NLHomoProblem> homo_problem = std::make_shared<NLHomoProblem>(
			ndof,
			macro_strain_constraint,
			*this, t, forms, solve_data.al_form, solve_symmetric_flag);
		if (solve_data.periodic_contact_form)
			homo_problem->add_form(solve_data.periodic_contact_form);
		if (solve_data.strain_al_lagr_form)
			homo_problem->add_form(solve_data.strain_al_lagr_form);

		solve_data.nl_problem = homo_problem;
		solve_data.nl_problem->init(Eigen::VectorXd::Zero(homo_problem->reduced_size() + homo_problem->macro_reduced_size()));
		solve_data.nl_problem->update_quantities(t, Eigen::VectorXd::Zero(homo_problem->reduced_size() + homo_problem->macro_reduced_size()));
	}

	void State::solve_homogenization_step(Eigen::MatrixXd &sol, const int t, bool adaptive_initial_weight)
	{
		const int dim = mesh->dimension();
		const int ndof = n_bases * dim;

		auto homo_problem = std::dynamic_pointer_cast<NLHomoProblem>(solve_data.nl_problem);

		Eigen::VectorXd extended_sol;
		extended_sol.setZero(ndof + dim * dim);

		if (sol.size() == extended_sol.size())
			extended_sol = sol;

		const auto &fixed_entry = macro_strain_constraint.get_fixed_entry();
		homo_problem->set_fixed_entry({});
		{
			std::shared_ptr<polysolve::nonlinear::Solver> nl_solver = make_nl_solver(true);

			Eigen::VectorXi al_indices = fixed_entry.array() + homo_problem->full_size();
			Eigen::VectorXd al_values = utils::flatten(macro_strain_constraint.eval(t))(fixed_entry);

			std::shared_ptr<MacroStrainLagrangianForm> lagr_form = solve_data.strain_al_lagr_form;
			lagr_form->enable();

			const double initial_weight = args["solver"]["augmented_lagrangian"]["initial_weight"];
			const double max_weight = args["solver"]["augmented_lagrangian"]["max_weight"];
			const double eta_tol = args["solver"]["augmented_lagrangian"]["eta"];
			const double scaling = args["solver"]["augmented_lagrangian"]["scaling"];
			double al_weight = initial_weight;

			Eigen::VectorXd tmp_sol = homo_problem->extended_to_reduced(extended_sol);
			const Eigen::VectorXd initial_sol = tmp_sol;
			const double initial_error = lagr_form->compute_error(extended_sol);
			double current_error = initial_error;

			// try to enforce fixed values on macro strain
			extended_sol(al_indices) = al_values;
			Eigen::VectorXd reduced_sol = homo_problem->extended_to_reduced(extended_sol);

			homo_problem->line_search_begin(tmp_sol, reduced_sol);
			int al_steps = 0;
			bool force_al = true;

			lagr_form->set_initial_weight(al_weight);

			while (force_al
				   || !std::isfinite(homo_problem->value(reduced_sol))
				   || !homo_problem->is_step_valid(tmp_sol, reduced_sol)
				   || !homo_problem->is_step_collision_free(tmp_sol, reduced_sol))
			{
				force_al = false;
				homo_problem->line_search_end();

				logger().info("Solving AL Problem with weight {}", al_weight);

				homo_problem->init(tmp_sol);
				try
				{
					nl_solver->minimize(*homo_problem, tmp_sol);
				}
				catch (const std::runtime_error &e)
				{
					logger().error("AL solve failed!");
				}

				extended_sol = homo_problem->reduced_to_extended(tmp_sol);
				logger().debug("Current macro strain: {}", extended_sol.tail(dim * dim));

				current_error = lagr_form->compute_error(extended_sol);
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

				// try to enforce fixed values on macro strain
				extended_sol(al_indices) = al_values;
				reduced_sol = homo_problem->extended_to_reduced(extended_sol);

				homo_problem->line_search_begin(tmp_sol, reduced_sol);
			}
			homo_problem->line_search_end();
			lagr_form->disable();
		}

		homo_problem->set_fixed_entry(fixed_entry);

		Eigen::VectorXd reduced_sol = homo_problem->extended_to_reduced(extended_sol);

		homo_problem->init(reduced_sol);
		std::shared_ptr<polysolve::nonlinear::Solver> nl_solver = make_nl_solver(false);
		nl_solver->minimize(*homo_problem, reduced_sol);

		logger().info("Macro Strain: {}", extended_sol.tail(dim * dim).transpose());

		// check saddle point
		{
			json linear_args = args["solver"]["linear"];
			std::string solver_name = linear_args["solver"];
			if (solver_name.find("Pardiso") != std::string::npos)
			{
				linear_args["solver"] = "Eigen::PardisoLLT";
				std::unique_ptr<polysolve::linear::Solver> solver =
					polysolve::linear::Solver::create(linear_args, logger());

				StiffnessMatrix A;
				homo_problem->hessian(reduced_sol, A);
				Eigen::VectorXd x, b = Eigen::VectorXd::Zero(A.rows());
				try
				{
					dirichlet_solve(
						*solver, A, b, {}, x, A.rows(), args["output"]["data"]["stiffness_mat"], false, false, false);
				}
				catch (const std::runtime_error &error)
				{
					logger().error("The solution is a saddle point!");
				}
			}
		}

		sol = homo_problem->reduced_to_extended(reduced_sol);
		if (args["/boundary_conditions/periodic_boundary/force_zero_mean"_json_pointer].get<bool>())
		{
			Eigen::VectorXd integral = io::Evaluator::integrate_function(bases, geom_bases(), sol, dim, dim);
			double area = io::Evaluator::integrate_function(bases, geom_bases(), Eigen::VectorXd::Ones(n_bases), dim, 1)(0);
			for (int d = 0; d < dim; d++)
				sol(Eigen::seqN(d, n_bases, dim), 0).array() -= integral(d) / area;

			reduced_sol = homo_problem->extended_to_reduced(sol);
		}

		if (optimization_enabled != solver::CacheLevel::None)
			cache_transient_adjoint_quantities(t, homo_problem->reduced_to_full(reduced_sol), utils::unflatten(sol.bottomRows(dim * dim), dim));
	}

	void State::solve_homogenization(const int time_steps, const double t0, const double dt, Eigen::MatrixXd &sol)
	{
		bool is_static = !is_param_valid(args, "time");
		if (!is_static && !args["time"]["quasistatic"])
			log_and_throw_error("Transient homogenization can only do quasi-static!");

		init_homogenization_solve(t0);

		const int dim = mesh->dimension();
		Eigen::MatrixXd extended_sol;
		for (int t = 0; t <= time_steps; ++t)
		{
			double forward_solve_time = 0, remeshing_time = 0, global_relaxation_time = 0;

			{
				POLYFEM_SCOPED_TIMER(forward_solve_time);
				solve_homogenization_step(extended_sol, t, false);
			}
			sol = extended_sol.topRows(extended_sol.size() - dim * dim) + io::Evaluator::generate_linear_field(n_bases, mesh_nodes, utils::unflatten(extended_sol.bottomRows(dim * dim), dim));

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

} // namespace polyfem