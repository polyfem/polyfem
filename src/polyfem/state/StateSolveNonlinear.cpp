#include <polyfem/State.hpp>

#include <polyfem/assembler/Mass.hpp>
#include <polyfem/assembler/ViscousDamping.hpp>

#include <polyfem/solver/forms/BodyForm.hpp>
#include <polyfem/solver/forms/ContactForm.hpp>
#include <polyfem/solver/forms/ElasticForm.hpp>
#include <polyfem/solver/forms/FrictionForm.hpp>
#include <polyfem/solver/forms/InertiaForm.hpp>
#include <polyfem/solver/forms/LaggedRegForm.hpp>
#include <polyfem/solver/forms/RayleighDampingForm.hpp>
#include <polyfem/solver/forms/lagrangian/BCLagrangianForm.hpp>

#include <polyfem/solver/NLProblem.hpp>
#include <polyfem/solver/ALSolver.hpp>
#include <polyfem/solver/SolveData.hpp>
#include <polyfem/io/MshWriter.hpp>
#include <polyfem/io/OBJWriter.hpp>
#include <polyfem/io/OutData.hpp>
#include <polyfem/utils/MatrixUtils.hpp>
#include <polyfem/utils/Timer.hpp>
#include <polyfem/utils/JSONUtils.hpp>
#include <polyfem/utils/BoundarySampler.hpp>

#include <ipc/ipc.hpp>

namespace polyfem
{
	using namespace mesh;
	using namespace solver;
	using namespace time_integrator;
	using namespace io;
	using namespace utils;

	std::shared_ptr<polysolve::nonlinear::Solver> State::make_nl_solver(bool for_al) const
	{
		return polysolve::nonlinear::Solver::create(for_al ? args["solver"]["augmented_lagrangian"]["nonlinear"] : args["solver"]["nonlinear"], args["solver"]["linear"], units.characteristic_length(), logger());
	}

	void State::solve_transient_tensor_nonlinear(const int time_steps, const double t0, const double dt, Eigen::MatrixXd &sol)
	{
		init_nonlinear_tensor_solve(sol, t0 + dt);

		// Write the total energy to a CSV file
		int save_i = 0;
		EnergyCSVWriter energy_csv(resolve_output_path("energy.csv"), solve_data);
		RuntimeStatsCSVWriter stats_csv(resolve_output_path("stats.csv"), *this, t0, dt);
		const bool remesh_enabled = args["space"]["remesh"]["enabled"];
		// const double save_dt = remesh_enabled ? (dt / 3) : dt;

		// Save the initial solution
		energy_csv.write(save_i, sol);
		save_timestep(t0, save_i, t0, dt, sol, Eigen::MatrixXd()); // no pressure
		save_i++;

		if (optimization_enabled != solver::CacheLevel::None)
			cache_transient_adjoint_quantities(0, sol, Eigen::MatrixXd::Zero(mesh->dimension(), mesh->dimension()));

		for (int t = 1; t <= time_steps; ++t)
		{
			double forward_solve_time = 0, remeshing_time = 0, global_relaxation_time = 0;

			{
				POLYFEM_SCOPED_TIMER(forward_solve_time);
				solve_tensor_nonlinear(sol, t);
			}

			if (remesh_enabled)
			{
				energy_csv.write(save_i, sol);
				// save_timestep(t0 + dt * t, save_i, t0, save_dt, sol, Eigen::MatrixXd()); // no pressure
				save_i++;

				bool remesh_success;
				{
					POLYFEM_SCOPED_TIMER(remeshing_time);
					remesh_success = this->remesh(t0 + dt * t, dt, sol);
				}

				// Save the solution after remeshing
				energy_csv.write(save_i, sol);
				// save_timestep(t0 + dt * t, save_i, t0, save_dt, sol, Eigen::MatrixXd()); // no pressure
				save_i++;

				// Only do global relaxation if remeshing was successful
				if (remesh_success)
				{
					POLYFEM_SCOPED_TIMER(global_relaxation_time);
					solve_tensor_nonlinear(sol, t, false); // solve the scene again after remeshing
				}
			}

			// Always save the solution for consistency
			energy_csv.write(save_i, sol);
			save_timestep(t0 + dt * t, t, t0, dt, sol, Eigen::MatrixXd()); // no pressure
			save_i++;

			if (optimization_enabled != solver::CacheLevel::None)
			{
				cache_transient_adjoint_quantities(t, sol, Eigen::MatrixXd::Zero(mesh->dimension(), mesh->dimension()));
			}

			{
				POLYFEM_SCOPED_TIMER("Update quantities");

				solve_data.time_integrator->update_quantities(sol);

				solve_data.nl_problem->update_quantities(t0 + (t + 1) * dt, sol);

				solve_data.update_dt();
				solve_data.update_barrier_stiffness(sol);
			}

			logger().info("{}/{}  t={}", t, time_steps, t0 + dt * t);

			const std::string rest_mesh_path = args["output"]["data"]["rest_mesh"].get<std::string>();
			if (!rest_mesh_path.empty())
			{
				Eigen::MatrixXd V;
				Eigen::MatrixXi F;
				build_mesh_matrices(V, F);
				io::MshWriter::write(
					resolve_output_path(fmt::format(args["output"]["data"]["rest_mesh"], t)),
					V, F, mesh->get_body_ids(), mesh->is_volume(), /*binary=*/true);
			}

			const std::string &state_path = resolve_output_path(fmt::format(args["output"]["data"]["state"], t));
			if (!state_path.empty())
				solve_data.time_integrator->save_state(state_path);

			// save restart file
			save_restart_json(t0, dt, t);
			if (remesh_enabled)
				stats_csv.write(t, forward_solve_time, remeshing_time, global_relaxation_time, sol);
		}
	}

	void State::init_nonlinear_tensor_solve(Eigen::MatrixXd &sol, const double t, const bool init_time_integrator)
	{
		assert(sol.cols() == 1);
		assert(!assembler->is_linear() || is_contact_enabled()); // non-linear
		assert(!problem->is_scalar());                           // tensor
		assert(mixed_assembler == nullptr);

		if (optimization_enabled != solver::CacheLevel::None)
		{
			if (initial_sol_update.size() == ndof())
				sol = initial_sol_update;
			else
				initial_sol_update = sol;
		}

		// --------------------------------------------------------------------
		// Check for initial intersections
		if (is_contact_enabled())
		{
			POLYFEM_SCOPED_TIMER("Check for initial intersections");

			const Eigen::MatrixXd displaced = collision_mesh.displace_vertices(
				utils::unflatten(sol, mesh->dimension()));

			if (ipc::has_intersections(collision_mesh, displaced, args["solver"]["contact"]["CCD"]["broad_phase"]))
			{
				OBJWriter::write(
					resolve_output_path("intersection.obj"), displaced,
					collision_mesh.edges(), collision_mesh.faces());
				log_and_throw_error("Unable to solve, initial solution has intersections!");
			}
		}

		// --------------------------------------------------------------------
		// Initialize time integrator
		if (problem->is_time_dependent())
		{
			if (init_time_integrator)
			{
				POLYFEM_SCOPED_TIMER("Initialize time integrator");
				solve_data.time_integrator = ImplicitTimeIntegrator::construct_time_integrator(args["time"]["integrator"]);

				Eigen::MatrixXd solution, velocity, acceleration;
				initial_solution(solution); // Reload this because we need all previous solutions
				solution.col(0) = sol;      // Make sure the current solution is the same as `sol`
				assert(solution.rows() == sol.size());
				initial_velocity(velocity);
				assert(velocity.rows() == sol.size());
				initial_acceleration(acceleration);
				assert(acceleration.rows() == sol.size());

				if (optimization_enabled != solver::CacheLevel::None)
				{
					if (initial_vel_update.size() == ndof())
						velocity = initial_vel_update;
					else
						initial_vel_update = velocity;
				}

				const double dt = args["time"]["dt"];
				solve_data.time_integrator->init(solution, velocity, acceleration, dt);
			}
			assert(solve_data.time_integrator != nullptr);
		}
		else
		{
			solve_data.time_integrator = nullptr;
		}

		// --------------------------------------------------------------------
		// Initialize forms

		damping_assembler = std::make_shared<assembler::ViscousDamping>();
		set_materials(*damping_assembler);

		elasticity_pressure_assembler = build_pressure_assembler();

		// for backward solve
		damping_prev_assembler = std::make_shared<assembler::ViscousDampingPrev>();
		set_materials(*damping_prev_assembler);

		const ElementInversionCheck check_inversion = args["solver"]["advanced"]["check_inversion"];
		const std::vector<std::shared_ptr<Form>> forms = solve_data.init_forms(
			// General
			units,
			mesh->dimension(), t,
			// Elastic form
			n_bases, bases, geom_bases(), *assembler, ass_vals_cache, mass_ass_vals_cache, args["solver"]["advanced"]["jacobian_threshold"], check_inversion,
			// Body form
			n_pressure_bases, boundary_nodes, local_boundary,
			local_neumann_boundary,
			n_boundary_samples(), rhs, sol, mass_matrix_assembler->density(),
			// Pressure form
			local_pressure_boundary, local_pressure_cavity, elasticity_pressure_assembler,
			// Inertia form
			args.value("/time/quasistatic"_json_pointer, true), mass,
			damping_assembler->is_valid() ? damping_assembler : nullptr,
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

		for (const auto &form : forms)
			form->set_output_dir(output_dir);

		if (solve_data.contact_form != nullptr)
			solve_data.contact_form->save_ccd_debug_meshes = args["output"]["advanced"]["save_ccd_debug_meshes"];

		// --------------------------------------------------------------------
		// Initialize nonlinear problems

		const int ndof = n_bases * mesh->dimension();
		solve_data.nl_problem = std::make_shared<NLProblem>(
			ndof, periodic_bc, t, forms, solve_data.al_form);
		solve_data.nl_problem->init(sol);
		solve_data.nl_problem->update_quantities(t, sol);
		// --------------------------------------------------------------------

		stats.solver_info = json::array();
	}

	void State::solve_tensor_nonlinear(Eigen::MatrixXd &sol, const int t, const bool init_lagging)
	{
		assert(solve_data.nl_problem != nullptr);
		NLProblem &nl_problem = *(solve_data.nl_problem);

		assert(sol.size() == rhs.size());

		if (nl_problem.uses_lagging())
		{
			if (init_lagging)
			{
				POLYFEM_SCOPED_TIMER("Initializing lagging");
				nl_problem.init_lagging(sol); // TODO: this should be u_prev projected
			}
			logger().info("Lagging iteration 1:");
		}

		// ---------------------------------------------------------------------

		// Save the subsolve sequence for debugging
		int subsolve_count = 0;
		save_subsolve(subsolve_count, t, sol, Eigen::MatrixXd()); // no pressure

		// ---------------------------------------------------------------------

		std::shared_ptr<polysolve::nonlinear::Solver> nl_solver = make_nl_solver(true);

		ALSolver al_solver(
			solve_data.al_form,
			args["solver"]["augmented_lagrangian"]["initial_weight"],
			args["solver"]["augmented_lagrangian"]["scaling"],
			args["solver"]["augmented_lagrangian"]["max_weight"],
			args["solver"]["augmented_lagrangian"]["eta"],
			[&](const Eigen::VectorXd &x) {
				this->solve_data.update_barrier_stiffness(sol);
			});

		al_solver.post_subsolve = [&](const double al_weight) {
			stats.solver_info.push_back(
				{{"type", al_weight > 0 ? "al" : "rc"},
				 {"t", t}, // TODO: null if static?
				 {"info", nl_solver->info()}});
			if (al_weight > 0)
				stats.solver_info.back()["weight"] = al_weight;
			if (solve_data.contact_form)
				stats.solver_info.back()["contact_pairs"] = solve_data.contact_form->n_contact_pairs();
			save_subsolve(++subsolve_count, t, sol, Eigen::MatrixXd()); // no pressure
		};

		Eigen::MatrixXd prev_sol = sol;
		al_solver.solve_al(nl_solver, nl_problem, sol);

		nl_solver = make_nl_solver(false);
		al_solver.solve_reduced(nl_solver, nl_problem, sol);

		if (args["space"]["advanced"]["count_flipped_els_continuous"])
		{
			const auto invalidList = count_invalid(mesh->dimension(), bases, geom_bases(), sol);
			logger().debug("Flipped elements (cnt {}) : {}", invalidList.size(), invalidList);
		}

		// ---------------------------------------------------------------------

		// TODO: Make this more general
		const double lagging_tol = args["solver"]["contact"].value("friction_convergence_tol", 1e-2) * units.characteristic_length();

		if (optimization_enabled != solver::CacheLevel::Derivatives)
		{
			// Lagging loop (start at 1 because we already did an iteration above)
			bool lagging_converged = !nl_problem.uses_lagging();
			for (int lag_i = 1; !lagging_converged; lag_i++)
			{
				Eigen::VectorXd tmp_sol = nl_problem.full_to_reduced(sol);

				// Update the lagging before checking for convergence
				nl_problem.update_lagging(tmp_sol, lag_i);

				// Check if lagging converged
				Eigen::VectorXd grad;
				nl_problem.gradient(tmp_sol, grad);
				const double delta_x_norm = (prev_sol - sol).lpNorm<Eigen::Infinity>();
				logger().debug("Lagging convergence grad_norm={:g} tol={:g} (||Δx||={:g})", grad.norm(), lagging_tol, delta_x_norm);
				if (grad.norm() <= lagging_tol)
				{
					logger().info(
						"Lagging converged in {:d} iteration(s) (grad_norm={:g} tol={:g})",
						lag_i, grad.norm(), lagging_tol);
					lagging_converged = true;
					break;
				}

				if (delta_x_norm <= 1e-12)
				{
					logger().warn(
						"Lagging produced tiny update between iterations {:d} and {:d} (grad_norm={:g} grad_tol={:g} ||Δx||={:g} Δx_tol={:g}); stopping early",
						lag_i - 1, lag_i, grad.norm(), lagging_tol, delta_x_norm, 1e-6);
					lagging_converged = false;
					break;
				}

				// Check for convergence first before checking if we can continue
				if (lag_i >= nl_problem.max_lagging_iterations())
				{
					logger().warn(
						"Lagging failed to converge with {:d} iteration(s) (grad_norm={:g} tol={:g})",
						lag_i, grad.norm(), lagging_tol);
					lagging_converged = false;
					break;
				}

				// Solve the problem with the updated lagging
				logger().info("Lagging iteration {:d}:", lag_i + 1);
				nl_problem.init(sol);
				solve_data.update_barrier_stiffness(sol);
				nl_solver->minimize(nl_problem, tmp_sol);
				nl_problem.finish();
				prev_sol = sol;
				sol = nl_problem.reduced_to_full(tmp_sol);

				// Save the subsolve sequence for debugging and info
				stats.solver_info.push_back(
					{{"type", "rc"},
					 {"t", t}, // TODO: null if static?
					 {"lag_i", lag_i},
					 {"info", nl_solver->info()}});
				save_subsolve(++subsolve_count, t, sol, Eigen::MatrixXd()); // no pressure
			}
		}
	}
} // namespace polyfem
