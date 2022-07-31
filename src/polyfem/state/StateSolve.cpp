#include <polyfem/State.hpp>

#include <polyfem/time_integrator/BDF.hpp>
#include <polyfem/solver/TransientNavierStokesSolver.hpp>
#include <polyfem/solver/OperatorSplittingSolver.hpp>
#include <polyfem/solver/NavierStokesSolver.hpp>

#include <polyfem/solver/SparseNewtonDescentSolver.hpp>
#include <polyfem/solver/LBFGSSolver.hpp>

#include <polyfem/solver/NLProblem.hpp>
#include <polyfem/solver/ALNLProblem.hpp>

#include <polysolve/LinearSolver.hpp>
#include <polysolve/FEMSolver.hpp>

#include <polyfem/utils/StringUtils.hpp>
#include <polyfem/utils/OBJ_IO.hpp>

#include <polyfem/autogen/auto_p_bases.hpp>
#include <polyfem/autogen/auto_q_bases.hpp>

#include <ipc/ipc.hpp>

#include <fstream>

namespace polyfem
{
	using namespace assembler;
	using namespace mesh;
	using namespace solver;
	using namespace time_integrator;
	using namespace utils;

	namespace
	{
		void import_matrix(const std::string &path, const json &import, Eigen::MatrixXd &mat)
		{
			if (import.contains("offset"))
			{
				const int offset = import["offset"];

				Eigen::MatrixXd tmp;
				read_matrix(path, tmp);
				mat.block(0, 0, offset, 1) = tmp.block(0, 0, offset, 1);
			}
			else
			{
				read_matrix(path, mat);
			}
		}
	} // namespace

	template <typename ProblemType>
	std::shared_ptr<cppoptlib::NonlinearSolver<ProblemType>> State::make_nl_solver() const
	{
		std::string name = args["solver"]["nonlinear"]["solver"];
		if (name == "newton" || name == "Newton")
		{
			return std::make_shared<cppoptlib::SparseNewtonDescentSolver<ProblemType>>(
				args["solver"]["nonlinear"], args["solver"]["linear"]["solver"], args["solver"]["linear"]["precond"]);
		}
		else if (name == "lbfgs" || name == "LBFGS" || name == "L-BFGS")
		{
			return std::make_shared<cppoptlib::LBFGSSolver<ProblemType>>(
				args["solver"]["nonlinear"]);
		}
		else
		{
			throw std::invalid_argument(fmt::format("invalid nonlinear solver type: {}", name));
		}
	}

	void State::init_solve(Eigen::VectorXd &c_sol)
	{
		POLYFEM_SCOPED_TIMER("Setup RHS");

		json rhs_solver_params = args["solver"]["linear"];
		if (!rhs_solver_params.contains("Pardiso"))
			rhs_solver_params["Pardiso"] = {};
		rhs_solver_params["Pardiso"]["mtype"] = -2; // matrix type for Pardiso (2 = SPD)

		const int size = problem->is_scalar() ? 1 : mesh->dimension();
		const auto &gbases = iso_parametric() ? bases : geom_bases;

		RhsAssembler rhs_assembler(assembler, *mesh, obstacle, input_dirichelt,
								   n_bases, size,
								   bases, iso_parametric() ? bases : geom_bases, ass_vals_cache,
								   formulation(), *problem,
								   args["space"]["advanced"]["bc_method"],
								   args["solver"]["linear"]["solver"], args["solver"]["linear"]["precond"], rhs_solver_params);

		solve_data.rhs_assembler = std::make_shared<RhsAssembler>(
			assembler, *mesh, obstacle, input_dirichelt, n_bases, size, bases, gbases, ass_vals_cache,
			formulation(), *problem, args["space"]["advanced"]["bc_method"],
			args["solver"]["linear"]["solver"], args["solver"]["linear"]["precond"],
			rhs_solver_params);
		RhsAssembler &rhs_assembler = *solve_data.rhs_assembler;

		const std::string u_path = resolve_input_path(args["input"]["data"]["u_path"]);
		if (!u_path.empty())
		{
			read_matrix(u_path, sol);
			// TODO fix import
			// import_matrix(u_path, args["input"]["data"]["u_path"], sol);
		}
		else
			rhs_assembler.initial_solution(sol);

		if (assembler.is_mixed(formulation()))
		{
			pressure.resize(0, 0);
			const int prev_size = sol.size();
			sol.conservativeResize(rhs.size(), sol.cols());
			// Zero initial pressure
			sol.block(prev_size, 0, n_pressure_bases, sol.cols()).setZero();
			sol(sol.size() - 1) = 0;
		}

		c_sol = sol;

		if (assembler.is_mixed(formulation()))
			sol_to_pressure();
	}

	void State::solve_linear()
	{
		assert(!problem->is_time_dependent());
		assert(assembler.is_linear(formulation()) && !args["contact"]["enabled"]);

		auto solver = polysolve::LinearSolver::create(
			args["solver"]["linear"]["solver"], args["solver"]["linear"]["precond"]);
		solver->setParameters(args["solver"]["linear"]);
		logger().info("{}...", solver->name());

		StiffnessMatrix A;
		Eigen::VectorXd b;
		json rhs_solver_params = args["solver"]["linear"];
		if (!rhs_solver_params.contains("Pardiso"))
			rhs_solver_params["Pardiso"] = {};
		rhs_solver_params["Pardiso"]["mtype"] = -2; // matrix type for Pardiso (2 = SPD)
		const int size = problem->is_scalar() ? 1 : mesh->dimension();
		RhsAssembler rhs_assembler(assembler, *mesh, obstacle, input_dirichelt,
								   n_bases, size,
								   bases, iso_parametric() ? bases : geom_bases, ass_vals_cache,
								   formulation(), *problem,
								   args["space"]["advanced"]["bc_method"],
								   args["solver"]["linear"]["solver"], args["solver"]["linear"]["precond"], rhs_solver_params);

		if (formulation() != "Bilaplacian")
			rhs_assembler.set_bc(local_boundary, boundary_nodes, n_boundary_samples(), local_neumann_boundary, rhs);
		else
			rhs_assembler.set_bc(local_boundary, boundary_nodes, n_boundary_samples(), std::vector<LocalBoundary>(), rhs);

		const int problem_dim = problem->is_scalar() ? 1 : mesh->dimension();
		const int precond_num = problem_dim * n_bases;

		A = stiffness;
		Eigen::VectorXd x;
		b = rhs;
		spectrum = dirichlet_solve(*solver, A, b, boundary_nodes, x, precond_num, args["output"]["data"]["stiffness_mat"], args["output"]["advanced"]["spectrum"], assembler.is_fluid(formulation()), use_avg_pressure);
		sol = x;
		solver->getInfo(solver_info);

		const auto error = (A * x - b).norm();
		if (error > 1e-4)
			logger().error("Solver error: {}", error);
		else
			logger().debug("Solver error: {}", error);

		if (assembler.is_mixed(formulation()))
		{
			sol_to_pressure();
		}
	}

	void State::solve_navier_stokes()
	{
		assert(!problem->is_time_dependent());
		assert(formulation() == "NavierStokes");

		NavierStokesSolver ns_solver(args["solver"]);
		Eigen::VectorXd x;
		json rhs_solver_params = args["solver"]["linear"];
		if (!rhs_solver_params.contains("Pardiso"))
			rhs_solver_params["Pardiso"] = {};
		rhs_solver_params["Pardiso"]["mtype"] = -2; // matrix type for Pardiso (2 = SPD)

		RhsAssembler rhs_assembler(assembler, *mesh, obstacle, input_dirichelt,
								   n_bases, mesh->dimension(),
								   bases, iso_parametric() ? bases : geom_bases, ass_vals_cache,
								   formulation(), *problem,
								   args["space"]["advanced"]["bc_method"],
								   args["solver"]["linear"]["solver"], args["solver"]["linear"]["precond"], rhs_solver_params);
		rhs_assembler.set_bc(local_boundary, boundary_nodes, n_boundary_samples(), local_neumann_boundary, rhs);
		ns_solver.minimize(*this, rhs, x);
		sol = x;
		sol_to_pressure();
	}

	void State::solve_transient_tensor_nonlinear(const int time_steps, const double t0, const double dt)
	{
		init_nonlinear_tensor_solve();

		save_timestep(t0, 0, t0, dt);

		for (int t = 1; t <= time_steps; ++t)
		{
			solve_tensor_nonlinear(t);

			{
				POLYFEM_SCOPED_TIMER("Update quantities");
				nl_problem.update_quantities(t0 + (t + 1) * dt, sol);
				alnl_problem.update_quantities(t0 + (t + 1) * dt, sol);
			}

			save_timestep(t0 + dt * t, t, t0, dt);

			logger().info("{}/{}  t={}", t, time_steps, t0 + dt * t);
		}

		solve_data.nl_problem->save_raw(
			resolve_output_path(args["output"]["data"]["u_path"]),
			resolve_output_path(args["output"]["data"]["v_path"]),
			resolve_output_path(args["output"]["data"]["a_path"]));
	}

	void State::init_nonlinear_tensor_solve()
	{
		assert(!assembler.is_linear(formulation()) || args["contact"]["enabled"]); // non-linear
		assert(!problem->is_scalar());                                             // tensor
		assert(!assembler.is_mixed(formulation()));

		///////////////////////////////////////////////////////////////////////
		// Check for initial intersections
		if (args["contact"]["enabled"])
		{
			POLYFEM_SCOPED_TIMER("Check for initial intersections");

			Eigen::MatrixXd displaced = boundary_nodes_pos + unflatten(sol, mesh->dimension());

			if (ipc::has_intersections(collision_mesh, collision_mesh.vertices(displaced)))
			{
				OBJWriter::save("intersection.obj", collision_mesh.vertices(displaced), collision_mesh.edges(), collision_mesh.faces());
				logger().error("Unable to solve, initial solution has intersections!");
				throw std::runtime_error("Unable to solve, initial solution has intersections!");
			}
		}

		///////////////////////////////////////////////////////////////////////
		// Initialize nonlinear problems
		solve_data.nl_problem = std::make_shared<NLProblem>(
			*this, solve_data.rhs_assembler, 1, args["contact"]["dhat"]);

		const double al_weight = args["solver"]["augmented_lagrangian"]["initial_weight"];
		solve_data.alnl_problem = std::make_shared<ALNLProblem>(
			*this, solve_data.rhs_assembler, 1, args["contact"]["dhat"], al_weight);

		///////////////////////////////////////////////////////////////////////
		// Initialize time integrator
		if (problem->is_time_dependent())
		{
			POLYFEM_SCOPED_TIMER("Initialize time integrator");

			// TODO import
			Eigen::MatrixXd velocity;
			const std::string v_path = resolve_input_path(args["input"]["data"]["v_path"]);
			if (!v_path.empty())
				import_matrix(v_path, args["import"], velocity);
			else
				rhs_assembler.initial_velocity(velocity);

			// TODO import
			Eigen::MatrixXd acceleration;
			const std::string a_path = resolve_input_path(args["input"]["data"]["a_path"]);
			if (!a_path.empty())
				import_matrix(a_path, args["import"], acceleration);
			else
				rhs_assembler.initial_acceleration(acceleration);

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
		NLProblem &nl_problem = *(solve_data.nl_problem);
		ALNLProblem &alnl_problem = *(solve_data.alnl_problem);

		assert(sol.size() == rhs.size());

		double al_weight = args["solver"]["augmented_lagrangian"]["initial_weight"];
		const double max_al_weight = args["solver"]["augmented_lagrangian"]["max_weight"];

		nl_problem.full_to_reduced(sol, tmp_sol);
		assert(sol.size() == rhs.size());
		assert(tmp_sol.size() <= rhs.size());

		{
			POLYFEM_SCOPED_TIMER("Initializing lagging");
			nl_problem.init_lagging(sol);
			alnl_problem.init_lagging(sol);
		}

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

		// TODO: maybe add linear solver here?

		solver_info = json::array();

		// Save the subsolve sequence for debugging
		int subsolve_count = 0;
		save_subsolve(subsolve_count, t);

		///////////////////////////////////////////////////////////////////////

		nl_problem.line_search_begin(sol, tmp_sol);
		bool force_al = args["solver"]["augmented_lagrangian"]["force"];
		while (force_al
			   || !std::isfinite(nl_problem.value(tmp_sol))
			   || !nl_problem.is_step_valid(sol, tmp_sol)
			   || !nl_problem.is_step_collision_free(sol, tmp_sol))
		{
			force_al = false;
			nl_problem.line_search_end();
			alnl_problem.set_weight(al_weight);
			logger().debug("Solving AL Problem with weight {}", al_weight);

			std::shared_ptr<cppoptlib::NonlinearSolver<ALNLProblem>> alnlsolver =
				make_nl_solver<ALNLProblem>();
			alnlsolver->setLineSearch(args["solver"]["nonlinear"]["line_search"]["method"]);
			alnl_problem.init(sol);
			tmp_sol = sol;
			alnlsolver->minimize(alnl_problem, tmp_sol);
			json alnl_solver_info;
			alnlsolver->getInfo(alnl_solver_info);

			solver_info.push_back({{"type", "al"},
								   {"t", t}, // TODO: null if static?
								   {"weight", al_weight},
								   {"info", alnl_solver_info}});

			sol = tmp_sol;
			nl_problem.full_to_reduced(sol, tmp_sol);
			nl_problem.line_search_begin(sol, tmp_sol);

			al_weight *= 2;

			if (al_weight >= max_al_weight)
			{
				const std::string msg = fmt::format(
					"Unable to solve AL problem, weight {} >= {}, stopping",
					al_weight, max_al_weight);
				logger().error(msg);
				throw std::runtime_error(msg);
				break;
			}

			save_subsolve(++subsolve_count, t);
		}
		nl_problem.line_search_end();

		///////////////////////////////////////////////////////////////////////

		std::shared_ptr<cppoptlib::NonlinearSolver<NLProblem>> nlsolver = make_nl_solver<NLProblem>();
		nlsolver->setLineSearch(args["solver"]["nonlinear"]["line_search"]["method"]);
		nl_problem.init(sol);
		nlsolver->minimize(nl_problem, tmp_sol);
		json nl_solver_info;
		nlsolver->getInfo(nl_solver_info);
		solver_info.push_back({{"type", "rc"},
							   {"t", t}, // TODO: null if static?
							   {"info", nl_solver_info}});
		nl_problem.reduced_to_full(tmp_sol, sol);

		save_subsolve(++subsolve_count, t);

		///////////////////////////////////////////////////////////////////////

		// TODO: fix this
		nl_problem.lagged_damping_weight() = 0;

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
			nlsolver->minimize(nl_problem, tmp_sol);

			nlsolver->getInfo(nl_solver_info);
			solver_info.push_back({{"type", "rc"},
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
				lagging_converged ? "Friction lagging converged using" : "Friction lagging maxed out at",
				lag_i, nl_problem.compute_lagging_error(tmp_sol),
				args["solver"]["contact"]["friction_convergence_tol"].get<double>());
		}
	}

	////////////////////////////////////////////////////////////////////////
	// Template instantiations
	template std::shared_ptr<cppoptlib::NonlinearSolver<NLProblem>> State::make_nl_solver() const;
	template std::shared_ptr<cppoptlib::NonlinearSolver<ALNLProblem>> State::make_nl_solver() const;
} // namespace polyfem
