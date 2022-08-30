#include <polyfem/State.hpp>

#include <polyfem/solver/NonlinearSolver.hpp>
#include <polyfem/solver/LBFGSSolver.hpp>
#include <polyfem/solver/SparseNewtonDescentSolver.hpp>
#include <polyfem/solver/NLProblem.hpp>
#include <polyfem/solver/ALNLProblem.hpp>
#include <polyfem/utils/MatrixUtils.hpp>
#include <polyfem/utils/OBJ_IO.hpp>
#include <polyfem/utils/Timer.hpp>

#include <ipc/ipc.hpp>

namespace polyfem
{
	using namespace solver;
	using namespace utils;

	template <typename ProblemType>
	std::shared_ptr<cppoptlib::NonlinearSolver<ProblemType>> State::make_nl_solver() const
	{
		const std::string name = args["solver"]["nonlinear"]["solver"];
		if (name == "newton" || name == "Newton")
		{
			return std::make_shared<cppoptlib::SparseNewtonDescentSolver<ProblemType>>(
				args["solver"]["nonlinear"], args["solver"]["linear"]["solver"], args["solver"]["linear"]["precond"]);
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

	void State::solve_transient_tensor_nonlinear(const int time_steps, const double t0, const double dt)
	{
		init_nonlinear_tensor_solve(t0 + dt);

		save_timestep(t0, 0, t0, dt);

		if (args["differentiable"])
		{
			cache_transient_adjoint_quantities(0);
		}

		for (int t = 1; t <= time_steps; ++t)
		{
			solve_tensor_nonlinear(t);

			{
				POLYFEM_SCOPED_TIMER("Update quantities");
				solve_data.nl_problem->update_quantities(t0 + (t + 1) * dt, sol);
				solve_data.alnl_problem->update_quantities(t0 + (t + 1) * dt, sol);
			}

			save_timestep(t0 + dt * t, t, t0, dt);

			if (args["differentiable"])
			{
				cache_transient_adjoint_quantities(t);
			}

			logger().info("{}/{}  t={}", t, time_steps, t0 + dt * t);
		}

		solve_data.nl_problem->save_raw(
			resolve_output_path(args["output"]["data"]["u_path"]),
			resolve_output_path(args["output"]["data"]["v_path"]),
			resolve_output_path(args["output"]["data"]["a_path"]));
	}

	void State::init_nonlinear_tensor_solve(const double t)
	{
		assert(!assembler.is_linear(formulation()) || is_contact_enabled()); // non-linear
		assert(!problem->is_scalar());                                       // tensor
		assert(!assembler.is_mixed(formulation()));

		const auto &cur_sol = (pre_sol.size() == sol.size()) ? pre_sol : sol;
		///////////////////////////////////////////////////////////////////////
		// Check for initial intersections
		if (is_contact_enabled())
		{
			POLYFEM_SCOPED_TIMER("Check for initial intersections");

			Eigen::MatrixXd displaced = boundary_nodes_pos + unflatten(cur_sol, mesh->dimension());

			if (ipc::has_intersections(collision_mesh, collision_mesh.vertices(displaced)))
			{
				OBJWriter::save(
					resolve_output_path("intersection.obj"), collision_mesh.vertices(displaced),
					collision_mesh.edges(), collision_mesh.faces());
				logger().error("Unable to solve, initial solution has intersections!");
				throw std::runtime_error("Unable to solve, initial solution has intersections!");
			}
		}

		///////////////////////////////////////////////////////////////////////
		// Initialize nonlinear problems
		assert(solve_data.rhs_assembler != nullptr);
		solve_data.nl_problem =
			std::make_shared<NLProblem>(*this, *solve_data.rhs_assembler, t, args["contact"]["dhat"]);

		const double al_weight = args["solver"]["augmented_lagrangian"]["initial_weight"];
		solve_data.alnl_problem =
			std::make_shared<ALNLProblem>(*this, *solve_data.rhs_assembler, t, args["contact"]["dhat"], al_weight);

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

			if (args["differentiable"])
			{
				if (initial_sol_update.size() > 0)
					sol = initial_sol_update;
				else
					initial_sol_update = sol;
				if (initial_vel_update.size() > 0)
					velocity = initial_vel_update;
				else
					initial_vel_update = velocity;
			}

			initial_velocity_cache = velocity;

			const double dt = args["time"]["dt"];
			solve_data.nl_problem->init_time_integrator(sol, velocity, acceleration, dt);
			solve_data.alnl_problem->init_time_integrator(sol, velocity, acceleration, dt);
		}

		///////////////////////////////////////////////////////////////////////

		solver_info = json::array();
	}

	void State::solve_tensor_nonlinear(const int t)
	{
		Eigen::VectorXd tmp_sol;

		assert(solve_data.nl_problem != nullptr);
		NLProblem &nl_problem = *(solve_data.nl_problem);
		assert(solve_data.alnl_problem != nullptr);
		ALNLProblem &alnl_problem = *(solve_data.alnl_problem);

		assert(sol.size() == rhs.size());

		double al_weight = args["solver"]["augmented_lagrangian"]["initial_weight"];
		const double max_al_weight = args["solver"]["augmented_lagrangian"]["max_weight"];

		assert(sol.size() == rhs.size());
		assert(tmp_sol.size() <= rhs.size());

		{
			POLYFEM_SCOPED_TIMER("Initializing lagging");
			nl_problem.init_lagging(sol);
			alnl_problem.init_lagging(sol);
		}

		if (pre_sol.rows() == sol.rows())
		{
			logger().debug("Use better initial guess...");
			sol = pre_sol;
		}

		nl_problem.full_to_reduced(sol, tmp_sol);

		const int friction_iterations = args["solver"]["contact"]["friction_iterations"];
		assert(friction_iterations >= 0);
		if (friction_iterations > 0)
			logger().debug("Lagging iteration 1");

		// Disable damping for the final lagged iteration
		if (friction_iterations <= 1)
		{
			nl_problem.lagged_damping_weight() = 0;
			alnl_problem.lagged_damping_weight() = 0;
		}

		// Save the subsolve sequence for debugging
		int subsolve_count = 0;
		save_subsolve(subsolve_count, t);

		///////////////////////////////////////////////////////////////////////

		nl_problem.line_search_begin(sol, tmp_sol);
		bool force_al = args["solver"]["augmented_lagrangian"]["force"];
		while (force_al || !std::isfinite(nl_problem.value(tmp_sol)) || !nl_problem.is_step_valid(sol, tmp_sol)
			   || !nl_problem.is_step_collision_free(sol, tmp_sol))
		{
			force_al = false;
			nl_problem.line_search_end(false);
			alnl_problem.set_weight(al_weight);
			logger().debug("Solving AL Problem with weight {}", al_weight);

			std::shared_ptr<cppoptlib::NonlinearSolver<ALNLProblem>> alnl_solver = make_nl_solver<ALNLProblem>();
			alnl_solver->setLineSearch(args["solver"]["nonlinear"]["line_search"]["method"]);
			alnl_problem.init(sol);
			tmp_sol = sol;
			alnl_solver->minimize(alnl_problem, tmp_sol);
			json alnl_solver_info;
			alnl_solver->getInfo(alnl_solver_info);

			solver_info.push_back(
				{{"type", "al"},
				 {"t", t}, // TODO: null if static?
				 {"weight", al_weight},
				 {"info", alnl_solver_info}});

			sol = tmp_sol;
			nl_problem.full_to_reduced(sol, tmp_sol);
			nl_problem.line_search_begin(sol, tmp_sol);

			al_weight *= 2;

			if (al_weight >= max_al_weight)
			{
				const std::string msg =
					fmt::format("Unable to solve AL problem, weight {} >= {}, stopping", al_weight, max_al_weight);
				logger().error(msg);
				throw std::runtime_error(msg);
				break;
			}

			save_subsolve(++subsolve_count, t);
		}
		nl_problem.line_search_end(false);

		///////////////////////////////////////////////////////////////////////

		std::shared_ptr<cppoptlib::NonlinearSolver<NLProblem>> nl_solver = make_nl_solver<NLProblem>();
		nl_solver->setLineSearch(args["solver"]["nonlinear"]["line_search"]["method"]);
		nl_problem.init(sol);
		nl_solver->minimize(nl_problem, tmp_sol);
		json nl_solver_info;
		nl_solver->getInfo(nl_solver_info);
		solver_info.push_back(
			{{"type", "rc"},
			 {"t", t}, // TODO: null if static?
			 {"info", nl_solver_info}});
		nl_problem.reduced_to_full(tmp_sol, sol);

		save_subsolve(++subsolve_count, t);

		///////////////////////////////////////////////////////////////////////

		// TODO: fix this
		nl_problem.lagged_damping_weight() = 0;

		if (!args["differentiable"])
		{
			// Lagging loop (start at 1 because we already did an iteration above)
			int lag_i;
			nl_problem.update_lagging(tmp_sol);
			bool lagging_converged = nl_problem.lagging_converged(tmp_sol);
			for (lag_i = 1; !lagging_converged && lag_i < friction_iterations; lag_i++)
			{
				logger().debug("Lagging iteration {:d}", lag_i + 1);
				nl_problem.init(sol);
				// Disable damping for the final lagged iteration
				if (lag_i == friction_iterations - 1)
					nl_problem.lagged_damping_weight() = 0;
				nl_solver->minimize(nl_problem, tmp_sol);

				nl_solver->getInfo(nl_solver_info);
				solver_info.push_back(
					{{"type", "rc"},
					{"t", t}, // TODO: null if static?
					{"lag_i", lag_i},
					{"info", nl_solver_info}});

				nl_problem.reduced_to_full(tmp_sol, sol);
				nl_problem.update_lagging(tmp_sol);
				lagging_converged = nl_problem.lagging_converged(tmp_sol);

				save_subsolve(++subsolve_count, t);
			}

			///////////////////////////////////////////////////////////////////////

			if (friction_iterations > 0)
			{
				logger().log(
					lagging_converged ? spdlog::level::info : spdlog::level::warn,
					"{} {:d} lagging iteration(s) (err={:g} tol={:g})",
					lagging_converged ? "Friction lagging converged using" : "Friction lagging maxed out at", lag_i,
					nl_problem.compute_lagging_error(tmp_sol),
					args["solver"]["contact"]["friction_convergence_tol"].get<double>());
			}
		}
	}

	////////////////////////////////////////////////////////////////////////
	// Template instantiations
	template std::shared_ptr<cppoptlib::NonlinearSolver<NLProblem>> State::make_nl_solver() const;
	template std::shared_ptr<cppoptlib::NonlinearSolver<ALNLProblem>> State::make_nl_solver() const;
} // namespace polyfem
