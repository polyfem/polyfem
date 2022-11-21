#include <polyfem/State.hpp>

#include <polyfem/solver/forms/BodyForm.hpp>
#include <polyfem/solver/forms/ContactForm.hpp>
#include <polyfem/solver/forms/ElasticForm.hpp>
#include <polyfem/solver/forms/FrictionForm.hpp>
#include <polyfem/solver/forms/InertiaForm.hpp>
#include <polyfem/solver/forms/LaggedRegForm.hpp>
#include <polyfem/solver/forms/ALForm.hpp>
#include <polyfem/solver/forms/RayleighDampingForm.hpp>

#include <polyfem/solver/NonlinearSolver.hpp>
#include <polyfem/solver/LBFGSSolver.hpp>
#include <polyfem/solver/SparseNewtonDescentSolver.hpp>
#include <polyfem/solver/NLProblem.hpp>
#include <polyfem/solver/ALSolver.hpp>
#include <polyfem/solver/SolveData.hpp>
#include <polyfem/io/OBJWriter.hpp>
#include <polyfem/mesh/remesh/Remesh.hpp>
#include <polyfem/utils/MatrixUtils.hpp>
#include <polyfem/utils/Timer.hpp>
#include <polyfem/utils/JSONUtils.hpp>

#include <ipc/ipc.hpp>

namespace polyfem
{
	using namespace mesh;
	using namespace solver;
	using namespace time_integrator;
	using namespace io;
	using namespace utils;

	template <typename ProblemType>
	std::shared_ptr<cppoptlib::NonlinearSolver<ProblemType>> State::make_nl_solver() const
	{
		const std::string name = args["solver"]["nonlinear"]["solver"];
		if (name == "newton" || name == "Newton")
		{
			return std::make_shared<cppoptlib::SparseNewtonDescentSolver<ProblemType>>(
				args["solver"]["nonlinear"], args["solver"]["linear"]);
		}
		else if (name == "lbfgs" || name == "LBFGS" || name == "L-BFGS")
		{
			return std::make_shared<cppoptlib::LBFGSSolver<ProblemType>>(args["solver"]["nonlinear"]);
		}
		else
		{
			throw std::invalid_argument(fmt::format("invalid nonlinear solver type: {}", name));
		}
	}

	void State::solve_transient_tensor_nonlinear(const int time_steps, const double t0, const double dt, Eigen::MatrixXd &sol)
	{
		init_nonlinear_tensor_solve(sol, t0 + dt);

		const double save_dt = dt / 3; // dt;
		int save_i = 0;

		save_timestep(t0, save_i++, t0, save_dt, sol, Eigen::MatrixXd()); // no pressure

		// Write the total energy to a CSV file
		std::ofstream energy_file(resolve_output_path("energy.csv"));
		energy_file << "i,elastic_energy,body_energy,inertia,contact_form,AL_energy,total_energy" << std::endl;
		const auto save_energy = [&](int i) {
			energy_file << fmt::format(
				"{},{},{},{},{},{},{}\n", i,
				solve_data.elastic_form->value(sol),
				solve_data.body_form->value(sol),
				solve_data.inertia_form ? solve_data.inertia_form->value(sol) : 0,
				solve_data.contact_form ? solve_data.contact_form->value(sol) : 0,
				solve_data.al_form->value(sol),
				solve_data.nl_problem->value(sol));
			energy_file.flush();
		};
		std::ofstream relax_diff_file(resolve_output_path("relax_diff.csv"));
		relax_diff_file << "L2,Linf" << std::endl;

		for (int t = 1; t <= time_steps; ++t)
		{
			solve_tensor_nonlinear(sol, t);

			if (t0 + dt * t >= args["space"]["remesh"]["t0"].get<double>())
			{
				save_energy(save_i);
				save_timestep(t0 + save_dt * t, save_i++, t0, save_dt, sol, Eigen::MatrixXd()); // no pressure

				// const double kappa_before = solve_data.contact_form->barrier_stiffness();
				mesh::remesh(*this, sol, t0 + dt * (t + 0), dt);
				// solve_data.updated_barrier_stiffness(sol);

				save_energy(save_i);
				save_timestep(t0 + save_dt * t, save_i++, t0, save_dt, sol, Eigen::MatrixXd()); // no pressure

				const Eigen::MatrixXd loc_relax_sol = sol;
				solve_tensor_nonlinear(sol, t); // solve the scene again after remeshing
				relax_diff_file << fmt::format("{},{}\n", (loc_relax_sol - sol).norm(), (loc_relax_sol - sol).lpNorm<Eigen::Infinity>());
				relax_diff_file.flush();

				save_energy(save_i);
				save_timestep(t0 + save_dt * t, save_i++, t0, save_dt, sol, Eigen::MatrixXd()); // no pressure
			}
			else
			{
				save_energy(save_i);
				save_timestep(t0 + save_dt * t, save_i++, t0, save_dt, sol, Eigen::MatrixXd()); // no pressure
				save_energy(save_i);
				save_timestep(t0 + save_dt * t, save_i++, t0, save_dt, sol, Eigen::MatrixXd()); // no pressure
				save_energy(save_i);
				save_timestep(t0 + save_dt * t, save_i++, t0, save_dt, sol, Eigen::MatrixXd()); // no pressure
			}

			{
				POLYFEM_SCOPED_TIMER("Update quantities");

				solve_data.time_integrator->update_quantities(sol, args["time"]["quasistatic"]);

				solve_data.nl_problem->update_quantities(t0 + (t + 1) * dt, sol);

				solve_data.update_dt();
				solve_data.updated_barrier_stiffness(sol);
			}

			logger().info("{}/{}  t={}", t, time_steps, t0 + dt * t);
		}

		solve_data.time_integrator->save_raw(
			resolve_output_path(args["output"]["data"]["u_path"]),
			resolve_output_path(args["output"]["data"]["v_path"]),
			resolve_output_path(args["output"]["data"]["a_path"]));
	}

	void State::init_nonlinear_tensor_solve(Eigen::MatrixXd &sol, const double t)
	{
		assert(!assembler.is_linear(formulation()) || is_contact_enabled()); // non-linear
		assert(!problem->is_scalar());                                       // tensor
		assert(!assembler.is_mixed(formulation()));

		// --------------------------------------------------------------------
		// Check for initial intersections
		if (is_contact_enabled())
		{
			POLYFEM_SCOPED_TIMER("Check for initial intersections");

			const Eigen::MatrixXd displaced = collision_mesh.displace_vertices(
				utils::unflatten(sol, mesh->dimension()));

			if (ipc::has_intersections(collision_mesh, displaced))
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
			POLYFEM_SCOPED_TIMER("Initialize time integrator");
			solve_data.time_integrator = ImplicitTimeIntegrator::construct_time_integrator(args["time"]["integrator"]);

			Eigen::MatrixXd velocity, acceleration;
			initial_velocity(velocity);
			assert(velocity.size() == sol.size());
			initial_velocity(acceleration);
			assert(acceleration.size() == sol.size());

			const double dt = args["time"]["dt"];
			solve_data.time_integrator->init(sol, velocity, acceleration, dt);
		}

		// --------------------------------------------------------------------
		// Initialize forms

		const int ndof = n_bases * mesh->dimension();

		// const std::vector<std::shared_ptr<Form>> forms = solve_data.init_forms(
		// 	n_bases, bases, geom_bases(), assembler, ass_vals_cache, formulation(),
		// 	mesh->dimension(), n_pressure_bases, boundary_nodes, local_boundary,
		// 	local_neumann_boundary, n_boundary_samples(), rhs, t, sol, args, mass,
		// 	obstacle, collision_mesh, avg_mass);

		assert(solve_data.rhs_assembler != nullptr);

		std::vector<std::shared_ptr<Form>> forms;
		solve_data.elastic_form = std::make_shared<ElasticForm>(
			n_bases, bases, geom_bases(),
			assembler, ass_vals_cache,
			formulation(),
			problem->is_time_dependent() ? args["time"]["dt"].get<double>() : 0.0,
			mesh->is_volume());
		forms.push_back(solve_data.elastic_form);

		solve_data.body_form = std::make_shared<BodyForm>(
			ndof, n_pressure_bases,
			boundary_nodes, local_boundary, local_neumann_boundary, n_boundary_samples(),
			rhs, *solve_data.rhs_assembler,
			assembler.density(),
			/*apply_DBC=*/true, /*is_formulation_mixed=*/false, problem->is_time_dependent());
		solve_data.body_form->update_quantities(t, sol);
		forms.push_back(solve_data.body_form);

		solve_data.inertia_form = nullptr;
		solve_data.damping_form = nullptr;
		if (problem->is_time_dependent())
		{
			solve_data.time_integrator = time_integrator::ImplicitTimeIntegrator::construct_time_integrator(args["time"]["integrator"]);
			if (!args["solver"]["ignore_inertia"])
			{
				solve_data.inertia_form = std::make_shared<InertiaForm>(mass, *solve_data.time_integrator);
				forms.push_back(solve_data.inertia_form);
			}
			if (assembler.has_damping())
			{
				solve_data.damping_form = std::make_shared<ElasticForm>(
					n_bases, bases, geom_bases(),
					assembler, ass_vals_cache,
					"Damping",
					args["time"]["dt"],
					mesh->is_volume());
				forms.push_back(solve_data.damping_form);
			}
		}
		else
		{
			const double lagged_regularization_weight = args["solver"]["advanced"]["lagged_regularization_weight"];
			if (lagged_regularization_weight > 0)
			{
				forms.push_back(std::make_shared<LaggedRegForm>(args["solver"]["advanced"]["lagged_regularization_iterations"]));
				forms.back()->set_weight(lagged_regularization_weight);
			}
		}

		solve_data.al_form = std::make_shared<ALForm>(
			ndof,
			boundary_nodes, local_boundary, local_neumann_boundary, n_boundary_samples(),
			mass,
			*solve_data.rhs_assembler,
			obstacle,
			problem->is_time_dependent(),
			t);
		forms.push_back(solve_data.al_form);

		solve_data.contact_form = nullptr;
		solve_data.friction_form = nullptr;
		if (args["contact"]["enabled"])
		{

			const bool use_adaptive_barrier_stiffness = !args["solver"]["contact"]["barrier_stiffness"].is_number();

			solve_data.contact_form = std::make_shared<ContactForm>(
				collision_mesh,
				args["contact"]["dhat"],
				avg_mass,
				use_adaptive_barrier_stiffness,
				/*is_time_dependent=*/solve_data.time_integrator != nullptr,
				args["solver"]["contact"]["CCD"]["broad_phase"],
				args["solver"]["contact"]["CCD"]["tolerance"],
				args["solver"]["contact"]["CCD"]["max_iterations"]);

			if (use_adaptive_barrier_stiffness)
			{
				solve_data.contact_form->set_weight(1);
				logger().debug("Using adaptive barrier stiffness");
			}
			else
			{
				solve_data.contact_form->set_weight(args["solver"]["contact"]["barrier_stiffness"]);
				logger().debug("Using fixed barrier stiffness of {}", solve_data.contact_form->barrier_stiffness());
			}

			forms.push_back(solve_data.contact_form);

			// ----------------------------------------------------------------

			if (args["contact"]["friction_coefficient"].get<double>() != 0)
			{
				solve_data.friction_form = std::make_shared<FrictionForm>(
					collision_mesh,
					args["contact"]["epsv"],
					args["contact"]["friction_coefficient"],
					args["contact"]["dhat"],
					args["solver"]["contact"]["CCD"]["broad_phase"],
					args.value("/time/dt"_json_pointer, 1.0), // dt=1.0 if static
					*solve_data.contact_form,
					args["solver"]["contact"]["friction_iterations"]);
				forms.push_back(solve_data.friction_form);
			}
		}

		std::vector<json> rayleigh_damping_jsons;
		if (args["solver"]["rayleigh_damping"].is_array())
			rayleigh_damping_jsons = args["solver"]["rayleigh_damping"].get<std::vector<json>>();
		else
			rayleigh_damping_jsons.push_back(args["solver"]["rayleigh_damping"]);
		if (problem->is_time_dependent())
		{
			// Map from form name to form so RayleighDampingForm::create can get the correct form to damp
			const std::unordered_map<std::string, std::shared_ptr<Form>> possible_forms_to_damp = {
				{"elasticity", solve_data.elastic_form},
				{"contact", solve_data.contact_form},
			};

			for (const json &params : rayleigh_damping_jsons)
			{
				forms.push_back(RayleighDampingForm::create(
					params, possible_forms_to_damp,
					*solve_data.time_integrator));
			}
		}
		else if (rayleigh_damping_jsons.size() > 0)
		{
			log_and_throw_error("Rayleigh damping is only supported for time-dependent problems");
		}

		// --------------------------------------------------------------------
		// Initialize nonlinear problems

		// const int ndof = n_bases * mesh->dimension();
		solve_data.nl_problem = std::make_shared<NLProblem>(
			ndof, boundary_nodes, local_boundary, n_boundary_samples(),
			*solve_data.rhs_assembler, t, forms);

		///////////////////////////////////////////////////////////////////////
		// Initialize time integrator
		if (problem->is_time_dependent())
		{
			POLYFEM_SCOPED_TIMER("Initialize time integrator");

			Eigen::MatrixXd velocity, acceleration;
			initial_velocity(velocity);
			assert(velocity.size() == sol.size());
			initial_velocity(acceleration);
			assert(acceleration.size() == sol.size());

			const double dt = args["time"]["dt"];
			solve_data.time_integrator->init(sol, velocity, acceleration, dt);
		}
		solve_data.update_dt();

		///////////////////////////////////////////////////////////////////////

		stats.solver_info = json::array();
	}

	void State::solve_tensor_nonlinear(Eigen::MatrixXd &sol, const int t)
	{

		assert(solve_data.nl_problem != nullptr);
		NLProblem &nl_problem = *(solve_data.nl_problem);

		assert(sol.size() == rhs.size());

		if (nl_problem.uses_lagging())
		{
			POLYFEM_SCOPED_TIMER("Initializing lagging");
			nl_problem.init_lagging(sol); // TODO: this should be u_prev projected
			logger().info("Lagging iteration {:d}:", 1);
		}

		// ---------------------------------------------------------------------

		// Save the subsolve sequence for debugging
		int subsolve_count = 0;
		save_subsolve(subsolve_count, t, sol, Eigen::MatrixXd()); // no pressure

		// ---------------------------------------------------------------------

		std::shared_ptr<cppoptlib::NonlinearSolver<NLProblem>> nl_solver = make_nl_solver<NLProblem>();

		ALSolver al_solver(
			nl_solver, solve_data.al_form,
			args["solver"]["augmented_lagrangian"]["initial_weight"],
			args["solver"]["augmented_lagrangian"]["max_weight"],
			[&](const Eigen::VectorXd &x) {
				this->solve_data.updated_barrier_stiffness(sol);
			});

		al_solver.post_subsolve = [&](const double al_weight) {
			json info;
			nl_solver->get_info(info);
			stats.solver_info.push_back(
				{{"type", al_weight > 0 ? "al" : "rc"},
				 {"t", t}, // TODO: null if static?
				 {"info", info}});
			if (al_weight > 0)
				stats.solver_info.back()["weight"] = al_weight;
			save_subsolve(++subsolve_count, t, sol, Eigen::MatrixXd()); // no pressure
		};

		al_solver.solve(nl_problem, sol, args["solver"]["augmented_lagrangian"]["force"]);

		// ---------------------------------------------------------------------

		// TODO: Make this more general
		const double lagging_tol = args["solver"]["contact"].value("friction_convergence_tol", 1e-2);

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
			logger().debug("Lagging convergence grad_norm={:g} tol={:g}", grad.norm(), lagging_tol);
			if (grad.norm() <= lagging_tol)
			{
				logger().info(
					"Lagging converged in {:d} iteration(s) (grad_norm={:g} tol={:g})",
					lag_i, grad.norm(), lagging_tol);
				lagging_converged = true;
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
			solve_data.updated_barrier_stiffness(sol);
			nl_solver->minimize(nl_problem, tmp_sol);
			sol = nl_problem.reduced_to_full(tmp_sol);

			// Save the subsolve sequence for debugging and info
			json info;
			nl_solver->get_info(info);
			stats.solver_info.push_back(
				{{"type", "rc"},
				 {"t", t}, // TODO: null if static?
				 {"lag_i", lag_i},
				 {"info", info}});
			save_subsolve(++subsolve_count, t, sol, Eigen::MatrixXd()); // no pressure
		}
	}

	////////////////////////////////////////////////////////////////////////
	// Template instantiations
	template std::shared_ptr<cppoptlib::NonlinearSolver<NLProblem>> State::make_nl_solver() const;
} // namespace polyfem
