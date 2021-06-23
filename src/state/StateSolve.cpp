#include <polyfem/State.hpp>

#include <polyfem/BDF.hpp>
#include <polyfem/TransientNavierStokesSolver.hpp>
#include <polyfem/NavierStokesSolver.hpp>

#include <polyfem/NLProblem.hpp>
#include <polyfem/ALNLProblem.hpp>

#include <polyfem/LbfgsSolver.hpp>
#include <polyfem/SparseNewtonDescentSolver.hpp>

#include <polysolve/LinearSolver.hpp>
#include <polysolve/FEMSolver.hpp>

#include <polyfem/StringUtils.hpp>

#include <fstream>

namespace polyfem
{
	void State::solve_transient_navier_stokes(const int time_steps, const double t0, const double dt, const RhsAssembler &rhs_assembler, Eigen::VectorXd &c_sol)
	{
		assert(formulation() == "NavierStokes" && problem->is_time_dependent());

		const auto &gbases = iso_parametric() ? bases : geom_bases;
		Eigen::MatrixXd current_rhs = rhs;

		StiffnessMatrix velocity_mass;
		assembler.assemble_mass_matrix(formulation(), mesh->is_volume(), n_bases, density, bases, gbases, ass_vals_cache, velocity_mass);

		StiffnessMatrix velocity_stiffness, mixed_stiffness, pressure_stiffness;

		Eigen::VectorXd prev_sol;

		int BDF_order = args["BDF_order"];
		// int aux_steps = BDF_order-1;
		BDF bdf(BDF_order);
		bdf.new_solution(c_sol);

		assembler.assemble_problem(formulation(), mesh->is_volume(), n_bases, bases, gbases, ass_vals_cache, velocity_stiffness);
		assembler.assemble_mixed_problem(formulation(), mesh->is_volume(), n_pressure_bases, n_bases, pressure_bases, bases, gbases, pressure_ass_vals_cache, ass_vals_cache, mixed_stiffness);
		assembler.assemble_pressure_problem(formulation(), mesh->is_volume(), n_pressure_bases, pressure_bases, gbases, pressure_ass_vals_cache, pressure_stiffness);

		TransientNavierStokesSolver ns_solver(solver_params(), build_json_params(), solver_type(), precond_type());
		const int n_larger = n_pressure_bases + (use_avg_pressure ? 1 : 0);

		for (int t = 1; t <= time_steps; ++t)
		{
			double time = t0 + t * dt;
			double current_dt = dt;

			logger().info("{}/{} steps, dt={}s t={}s", t, time_steps, current_dt, time);

			bdf.rhs(prev_sol);
			rhs_assembler.compute_energy_grad(local_boundary, boundary_nodes, density, args["n_boundary_samples"], local_neumann_boundary, rhs, time, current_rhs);
			rhs_assembler.set_bc(local_boundary, boundary_nodes, args["n_boundary_samples"], local_neumann_boundary, current_rhs, time);

			const int prev_size = current_rhs.size();
			if (prev_size != rhs.size())
			{
				current_rhs.conservativeResize(prev_size + n_larger, current_rhs.cols());
				current_rhs.block(prev_size, 0, n_larger, current_rhs.cols()).setZero();
			}

			ns_solver.minimize(*this, bdf.alpha(), current_dt, prev_sol,
							   velocity_stiffness, mixed_stiffness, pressure_stiffness,
							   velocity_mass, current_rhs, c_sol);
			bdf.new_solution(c_sol);
			sol = c_sol;
			sol_to_pressure();

			if (args["save_time_sequence"])
			{
				if (!solve_export_to_file)
					solution_frames.emplace_back();
				save_vtu(resolve_output_path(fmt::format("step_{:d}.vtu", t)), time);
				save_wire(resolve_output_path(fmt::format("step_{:d}.obj", t)));
			}
		}
	}

	void State::solve_transient_scalar(const int time_steps, const double t0, const double dt, const RhsAssembler &rhs_assembler, Eigen::VectorXd &x)
	{
		assert((problem->is_scalar() || assembler.is_mixed(formulation())) && problem->is_time_dependent());

		const json &params = solver_params();
		auto solver = polysolve::LinearSolver::create(args["solver_type"], args["precond_type"]);
		solver->setParameters(params);
		logger().info("{}...", solver->name());

		StiffnessMatrix A;
		Eigen::VectorXd b;
		Eigen::MatrixXd current_rhs = rhs;

		const int BDF_order = args["BDF_order"];
		// const int aux_steps = BDF_order-1;
		BDF bdf(BDF_order);
		bdf.new_solution(x);

		const int problem_dim = problem->is_scalar() ? 1 : mesh->dimension();
		const int precond_num = problem_dim * n_bases;

		for (int t = 1; t <= time_steps; ++t)
		{
			double time = t0 + t * dt;
			double current_dt = dt;

			logger().info("{}/{} {}s", t, time_steps, time);
			rhs_assembler.compute_energy_grad(local_boundary, boundary_nodes, density, args["n_boundary_samples"], local_neumann_boundary, rhs, time, current_rhs);
			rhs_assembler.set_bc(local_boundary, boundary_nodes, args["n_boundary_samples"], local_neumann_boundary, current_rhs, time);

			if (assembler.is_mixed(formulation()))
			{
				//divergence free
				int fluid_offset = use_avg_pressure ? (assembler.is_fluid(formulation()) ? 1 : 0) : 0;
				current_rhs.block(current_rhs.rows() - n_pressure_bases - use_avg_pressure, 0, n_pressure_bases + use_avg_pressure, current_rhs.cols()).setZero();
			}

			A = (bdf.alpha() / current_dt) * mass + stiffness;
			bdf.rhs(x);
			b = (mass * x) / current_dt;
			for (int i : boundary_nodes)
				b[i] = 0;
			b += current_rhs;

			spectrum = dirichlet_solve(*solver, A, b, boundary_nodes, x, precond_num, args["export"]["stiffness_mat"], t == time_steps && args["export"]["spectrum"], assembler.is_fluid(formulation()), use_avg_pressure);
			bdf.new_solution(x);
			sol = x;

			if (assembler.is_mixed(formulation()))
			{
				sol_to_pressure();
			}

			if (args["save_time_sequence"])
			{
				if (!solve_export_to_file)
					solution_frames.emplace_back();

				save_vtu(resolve_output_path(fmt::format("step_{:d}.vtu", t)), time);
				save_wire(resolve_output_path(fmt::format("step_{:d}.obj", t)));
			}
		}
	}

	void State::solve_transient_tensor_linear(const int time_steps, const double t0, const double dt, const RhsAssembler &rhs_assembler)
	{
		assert(!problem->is_scalar() && assembler.is_linear(formulation()) && !args["has_collision"] && problem->is_time_dependent());
		assert(!assembler.is_mixed(formulation()));

		const json &params = solver_params();
		auto solver = polysolve::LinearSolver::create(args["solver_type"], args["precond_type"]);
		solver->setParameters(params);
		logger().info("{}...", solver->name());

		const std::string v_path = resolve_path(args["import"]["v_path"], args["root_path"]);
		const std::string a_path = resolve_path(args["import"]["a_path"], args["root_path"]);

		Eigen::MatrixXd velocity, acceleration;

		if (!v_path.empty())
			read_matrix_binary(v_path, velocity);
		else
			rhs_assembler.initial_velocity(velocity);
		if (!a_path.empty())
			read_matrix_binary(a_path, acceleration);
		else
			rhs_assembler.initial_acceleration(acceleration);

		Eigen::MatrixXd current_rhs = rhs;

		const int problem_dim = problem->is_scalar() ? 1 : mesh->dimension();
		const int precond_num = problem_dim * n_bases;

		//Newmark
		const double gamma = 0.5;
		const double beta = 0.25;
		// makes the algorithm implicit and equivalent to the trapezoidal rule (unconditionally stable).

		Eigen::MatrixXd temp, b;
		StiffnessMatrix A;
		Eigen::VectorXd x, btmp;

		for (int t = 1; t <= time_steps; ++t)
		{
			const double dt2 = dt * dt;

			const Eigen::MatrixXd aOld = acceleration;
			const Eigen::MatrixXd vOld = velocity;
			const Eigen::MatrixXd uOld = sol;

			rhs_assembler.assemble(density, current_rhs, t0 + dt * t);
			current_rhs *= -1;

			temp = -(uOld + dt * vOld + ((1 / 2. - beta) * dt2) * aOld);
			b = stiffness * temp + current_rhs;

			rhs_assembler.set_acceleration_bc(local_boundary, boundary_nodes, args["n_boundary_samples"], local_neumann_boundary, b, t0 + dt * t);

			A = stiffness * beta * dt2 + mass;
			btmp = b;
			spectrum = dirichlet_solve(*solver, A, btmp, boundary_nodes, x, precond_num, args["export"]["stiffness_mat"], t == 1 && args["export"]["spectrum"], assembler.is_fluid(formulation()), use_avg_pressure);
			acceleration = x;

			sol += dt * vOld + dt2 * ((1 / 2.0 - beta) * aOld + beta * acceleration);
			velocity += dt * ((1 - gamma) * aOld + gamma * acceleration);

			rhs_assembler.set_bc(local_boundary, boundary_nodes, args["n_boundary_samples"], local_neumann_boundary, sol, t0 + dt * t);
			rhs_assembler.set_velocity_bc(local_boundary, boundary_nodes, args["n_boundary_samples"], local_neumann_boundary, velocity, t0 + dt * t);
			rhs_assembler.set_acceleration_bc(local_boundary, boundary_nodes, args["n_boundary_samples"], local_neumann_boundary, acceleration, t0 + dt * t);

			if (args["save_time_sequence"])
			{
				if (!solve_export_to_file)
					solution_frames.emplace_back();
				save_vtu(resolve_output_path(fmt::format("step_{:d}.vtu", t)), t0 + dt * t);
				save_wire(resolve_output_path(fmt::format("step_{:d}.obj", t)));
			}

			logger().info("{}/{} t={}", t, time_steps, t0 + dt * t);
		}

		{
			const std::string u_path = resolve_output_path(args["export"]["u_path"]);
			const std::string v_path = resolve_output_path(args["export"]["v_path"]);
			const std::string a_path = resolve_output_path(args["export"]["a_path"]);

			if (!u_path.empty())
				write_matrix_binary(u_path, sol);
			if (!v_path.empty())
				write_matrix_binary(v_path, velocity);
			if (!a_path.empty())
				write_matrix_binary(a_path, acceleration);
		}
	}

	void State::solve_transient_tensor_non_linear(const int time_steps, const double t0, const double dt, const RhsAssembler &rhs_assembler)
	{
		assert(!problem->is_scalar() && (!assembler.is_linear(formulation()) || args["has_collision"]) && problem->is_time_dependent());
		assert(!assembler.is_mixed(formulation()));

		// FD for debug
		// {
		// 	Eigen::MatrixXd velocity, acceleration;
		// 	boundary_nodes.clear();
		// 	local_boundary.clear();
		// 	// local_neumann_boundary.clear();
		// 	NLProblem nl_problem(*this, rhs_assembler, t0, args["dhat"], false);
		// 	Eigen::MatrixXd tmp_sol = rhs;

		// 	// tmp_sol.setRandom();
		// 	tmp_sol.setZero();
		// 	// tmp_sol /=10000.;

		// 	velocity.setZero();
		// 	VectorXd xxx = tmp_sol;
		// 	velocity = tmp_sol;
		// 	velocity.setZero();
		// 	acceleration = tmp_sol;
		// 	acceleration.setZero();
		// 	nl_problem.init_time_integrator(xxx, velocity, acceleration, dt);

		// 	Eigen::Matrix<double, Eigen::Dynamic, 1> actual_grad;
		// 	nl_problem.gradient(tmp_sol, actual_grad);

		// 	StiffnessMatrix hessian;
		// 	Eigen::MatrixXd expected_hessian;
		// 	nl_problem.hessian(tmp_sol, hessian);

		// 	Eigen::MatrixXd actual_hessian = Eigen::MatrixXd(hessian);
		// 	// std::cout << "hhh\n"<< actual_hessian<<std::endl;

		// 	for (int i = 0; i < actual_hessian.rows(); ++i)
		// 	{
		// 		double hhh = 1e-6;
		// 		VectorXd xp = tmp_sol;
		// 		xp(i) += hhh;
		// 		VectorXd xm = tmp_sol;
		// 		xm(i) -= hhh;

		// 		Eigen::Matrix<double, Eigen::Dynamic, 1> tmp_grad_p;
		// 		nl_problem.gradient(xp, tmp_grad_p);

		// 		Eigen::Matrix<double, Eigen::Dynamic, 1> tmp_grad_m;
		// 		nl_problem.gradient(xm, tmp_grad_m);

		// 		Eigen::Matrix<double, Eigen::Dynamic, 1> fd_h = (tmp_grad_p - tmp_grad_m) / (hhh * 2.);

		// 		const double vp = nl_problem.value(xp);
		// 		const double vm = nl_problem.value(xm);

		// 		const double fd = (vp - vm) / (hhh * 2.);
		// 		const double diff = std::abs(actual_grad(i) - fd);
		// 		if (diff > 1e-6)
		// 			std::cout << "diff grad " << i << ": " << actual_grad(i) << " vs " << fd << " error: " << diff << " rrr: " << actual_grad(i) / fd << std::endl;

		// 		for (int j = 0; j < actual_hessian.rows(); ++j)
		// 		{
		// 			const double diff = std::abs(actual_hessian(i, j) - fd_h(j));

		// 			if (diff > 1e-5)
		// 				std::cout << "diff H " << i << ", " << j << ": " << actual_hessian(i, j) << " vs " << fd_h(j) << " error: " << diff << " rrr: " << actual_hessian(i, j) / fd_h(j) << std::endl;
		// 		}
		// 	}

		// 	// std::cout<<"diff grad max "<<(actual_grad - expected_grad).array().abs().maxCoeff()<<std::endl;
		// 	// std::cout<<"diff \n"<<(actual_grad - expected_grad)<<std::endl;
		// 	exit(0);
		// }

		const std::string v_path = resolve_path(args["import"]["v_path"], args["root_path"]);
		const std::string a_path = resolve_path(args["import"]["a_path"], args["root_path"]);

		Eigen::MatrixXd velocity, acceleration;

		if (!v_path.empty())
			read_matrix_binary(v_path, velocity);
		else
			rhs_assembler.initial_velocity(velocity);
		if (!a_path.empty())
			read_matrix_binary(a_path, acceleration);
		else
			rhs_assembler.initial_acceleration(acceleration);

		const int full_size = n_bases * mesh->dimension();
		const int reduced_size = n_bases * mesh->dimension() - boundary_nodes.size();
		VectorXd tmp_sol;

		NLProblem nl_problem(*this, rhs_assembler, t0 + dt, args["dhat"], args["project_to_psd"]);
		nl_problem.init_time_integrator(sol, velocity, acceleration, dt);

		solver_info = json::array();

		// if (args["use_al"] || args["has_collision"])
		// {
		double al_weight = args["al_weight"];
		const double max_al_weight = args["max_al_weight"];
		ALNLProblem alnl_problem(*this, rhs_assembler, t0 + dt, args["dhat"], args["project_to_psd"], al_weight);
		alnl_problem.init_time_integrator(sol, velocity, acceleration, dt);

		for (int t = 1; t <= time_steps; ++t)
		{
			nl_problem.full_to_reduced(sol, tmp_sol);
			assert(sol.size() == rhs.size());
			assert(tmp_sol.size() < rhs.size());

			nl_problem.update_lagging(tmp_sol, /*start_of_timestep=*/true);
			alnl_problem.update_lagging(sol, /*start_of_timestep=*/true);

			if (args["friction_iterations"] > 0)
			{
				logger().debug("Lagging iteration 1");
			}

			nl_problem.line_search_begin(sol, tmp_sol);
			while (!std::isfinite(nl_problem.value(tmp_sol)) || !nl_problem.is_step_valid(sol, tmp_sol) || !nl_problem.is_step_collision_free(sol, tmp_sol))
			{
				nl_problem.line_search_end();
				alnl_problem.set_weight(al_weight);
				logger().debug("Solving AL Problem with weight {}", al_weight);

				cppoptlib::SparseNewtonDescentSolver<ALNLProblem> alnlsolver(solver_params(), solver_type(), precond_type());
				alnlsolver.setLineSearch(args["line_search"]);
				alnl_problem.init(sol);
				tmp_sol = sol;
				alnlsolver.minimize(alnl_problem, tmp_sol);
				json alnl_solver_info;
				alnlsolver.getInfo(alnl_solver_info);

				solver_info.push_back({{"type", "al"},
									   {"t", t},
									   {"weight", al_weight},
									   {"info", alnl_solver_info}});

				sol = tmp_sol;
				nl_problem.full_to_reduced(sol, tmp_sol);

				al_weight *= 2;

				if (al_weight >= max_al_weight)
				{
					logger().error("Unable to solve AL problem, weight {} >= {}, stopping", al_weight, max_al_weight);
					break;
				}
			}
			nl_problem.line_search_end();
			al_weight = args["al_weight"];
			logger().debug("Solving Problem");

			cppoptlib::SparseNewtonDescentSolver<NLProblem> nlsolver(solver_params(), solver_type(), precond_type());
			nlsolver.setLineSearch(args["line_search"]);
			nl_problem.init(sol);
			nlsolver.minimize(nl_problem, tmp_sol);
			json nl_solver_info;
			nlsolver.getInfo(nl_solver_info);
			nl_problem.reduced_to_full(tmp_sol, sol);

			// Lagging loop (start at 1 because we already did an iteration above)
			int lag_i;
			for (lag_i = 1; lag_i < args["friction_iterations"] && !nl_problem.lagging_converged(tmp_sol, /*do_lagging_update=*/true); lag_i++)
			{
				logger().debug("Lagging iteration {:d}", lag_i + 1);
				nl_problem.init(sol);
				nlsolver.minimize(nl_problem, tmp_sol);
				json nl_solver_info;
				nlsolver.getInfo(nl_solver_info);
				nl_problem.reduced_to_full(tmp_sol, sol);
			}

			if (args["friction_iterations"] > 0)
			{
				logger().info(
					lag_i >= args["friction_iterations"]
						? "Maxed out at {:d} lagging iteration{}"
						: "Convered using {:d} lagging iteration{}",
					lag_i, lag_i > 1 ? "s" : "");
			}

			nl_problem.update_quantities(t0 + (t + 1) * dt, sol);
			alnl_problem.update_quantities(t0 + (t + 1) * dt, sol);

			if (args["save_time_sequence"])
			{
				if (!solve_export_to_file)
					solution_frames.emplace_back();
				save_vtu(resolve_output_path(fmt::format("step_{:d}.vtu", t)), t0 + dt * t);
				save_wire(resolve_output_path(fmt::format("step_{:d}.obj", t)));
			}

			logger().info("{}/{}  t={}", t, time_steps, t0 + dt * t);

			solver_info.push_back({{"type", "rc"},
								   {"t", t},
								   {"info", nl_solver_info}});
		}
		// }
		// else
		// {
		// 	nl_problem.full_to_reduced(sol, tmp_sol);

		// 	for (int t = 1; t <= time_steps; ++t)
		// 	{
		// 		cppoptlib::SparseNewtonDescentSolver<NLProblem> nlsolver(solver_params(), solver_type(), precond_type());
		// 		nlsolver.setLineSearch(args["line_search"]);
		// 		nl_problem.init(sol);
		// 		nlsolver.minimize(nl_problem, tmp_sol);

		// 		if (nlsolver.error_code() == -10)
		// 		{
		// 			double substep_delta = 0.5;
		// 			double substep = substep_delta;
		// 			bool solved = false;

		// 			while (substep_delta > 1e-4 && !solved)
		// 			{
		// 				logger().debug("Substepping {}/{}, dt={}", (t - 1 + substep) * dt, t * dt, substep_delta);
		// 				nl_problem.substepping((t - 1 + substep) * dt);
		// 				nl_problem.full_to_reduced(sol, tmp_sol);
		// 				nlsolver.minimize(nl_problem, tmp_sol);

		// 				if (nlsolver.error_code() == -10)
		// 				{
		// 					substep -= substep_delta;
		// 					substep_delta /= 2;
		// 				}
		// 				else
		// 				{
		// 					logger().trace("Done {}/{}, dt={}", (t - 1 + substep) * dt, t * dt, substep_delta);
		// 					nl_problem.reduced_to_full(tmp_sol, sol);
		// 					substep_delta *= 2;
		// 				}

		// 				solved = substep >= 1;

		// 				substep += substep_delta;
		// 				if (substep >= 1)
		// 				{
		// 					substep_delta -= substep - 1;
		// 					substep = 1;
		// 				}
		// 			}
		// 		}

		// 		if (nlsolver.error_code() == -10)
		// 		{
		// 			logger().error("Unable to solve t={}", t * dt);
		// 			save_vtu("stop.vtu", dt * t);
		// 			break;
		// 		}

		// 		logger().debug("Step solved!");

		// 		nlsolver.getInfo(solver_info);
		// 		nl_problem.reduced_to_full(tmp_sol, sol);
		// 		if (assembler.is_mixed(formulation()))
		// 		{
		// 			sol_to_pressure();
		// 		}

		// 		// rhs_assembler.set_bc(local_boundary, boundary_nodes, args["n_boundary_samples"], local_neumann_boundary, sol, dt * t);

		// 		nl_problem.update_quantities((t + 1) * dt, sol);

		// 		if (args["save_time_sequence"])
		// 		{
		// 			if (!solve_export_to_file)
		// 				solution_frames.emplace_back();
		// 			save_vtu(fmt::format("step_{:d}.vtu", t), dt * t);
		// 			save_wire(fmt::format("step_{:d}.obj", t));
		// 		}

		// 		logger().info("{}/{}", t, time_steps);
		// 	}
		// }
		nl_problem.save_raw(
			resolve_output_path(args["export"]["u_path"]),
			resolve_output_path(args["export"]["v_path"]),
			resolve_output_path(args["export"]["a_path"]));
	}

	void State::solve_linear()
	{
		assert(!problem->is_time_dependent());
		assert(assembler.is_linear(formulation()) && !args["has_collision"]);
		const json &params = solver_params();
		auto solver = polysolve::LinearSolver::create(args["solver_type"], args["precond_type"]);
		solver->setParameters(params);
		StiffnessMatrix A;
		Eigen::VectorXd b;
		logger().info("{}...", solver->name());
		json rhs_solver_params = args["rhs_solver_params"];
		rhs_solver_params["mtype"] = -2; // matrix type for Pardiso (2 = SPD)
		const int size = problem->is_scalar() ? 1 : mesh->dimension();
		RhsAssembler rhs_assembler(assembler, *mesh,
								   n_bases, size,
								   bases, iso_parametric() ? bases : geom_bases, ass_vals_cache,
								   formulation(), *problem,
								   args["rhs_solver_type"], args["rhs_precond_type"], rhs_solver_params);

		if (formulation() != "Bilaplacian")
			rhs_assembler.set_bc(local_boundary, boundary_nodes, args["n_boundary_samples"], local_neumann_boundary, rhs);
		else
			rhs_assembler.set_bc(local_boundary, boundary_nodes, args["n_boundary_samples"], std::vector<LocalBoundary>(), rhs);

		const int problem_dim = problem->is_scalar() ? 1 : mesh->dimension();
		const int precond_num = problem_dim * n_bases;

		A = stiffness;
		Eigen::VectorXd x;
		b = rhs;
		spectrum = dirichlet_solve(*solver, A, b, boundary_nodes, x, precond_num, args["export"]["stiffness_mat"], args["export"]["spectrum"], assembler.is_fluid(formulation()), use_avg_pressure);
		sol = x;
		solver->getInfo(solver_info);

		logger().debug("Solver error: {}", (A * sol - b).norm());

		if (assembler.is_mixed(formulation()))
		{
			sol_to_pressure();
		}
	}

	void State::solve_navier_stokes()
	{
		assert(!problem->is_time_dependent());
		assert(formulation() == "NavierStokes");
		auto params = build_json_params();
		const double viscosity = params.count("viscosity") ? double(params["viscosity"]) : 1.;
		NavierStokesSolver ns_solver(viscosity, solver_params(), build_json_params(), solver_type(), precond_type());
		Eigen::VectorXd x;
		json rhs_solver_params = args["rhs_solver_params"];
		rhs_solver_params["mtype"] = -2; // matrix type for Pardiso (2 = SPD)

		RhsAssembler rhs_assembler(assembler, *mesh,
								   n_bases, mesh->dimension(),
								   bases, iso_parametric() ? bases : geom_bases, ass_vals_cache,
								   formulation(), *problem,
								   args["rhs_solver_type"], args["rhs_precond_type"], rhs_solver_params);
		rhs_assembler.set_bc(local_boundary, boundary_nodes, args["n_boundary_samples"], local_neumann_boundary, rhs);
		ns_solver.minimize(*this, rhs, x);
		sol = x;
		sol_to_pressure();
	}

	void State::solve_non_linear()
	{
		assert(!problem->is_time_dependent());
		assert(!assembler.is_linear(formulation()) || args["has_collision"]);

		const int full_size = n_bases * mesh->dimension();
		const int reduced_size = n_bases * mesh->dimension() - boundary_nodes.size();

		const int problem_dim = problem->is_scalar() ? 1 : mesh->dimension();
		const int precond_num = problem_dim * n_bases;

		const auto &gbases = iso_parametric() ? bases : geom_bases;

		json rhs_solver_params = args["rhs_solver_params"];
		rhs_solver_params["mtype"] = -2; // matrix type for Pardiso (2 = SPD)
		const int size = problem->is_scalar() ? 1 : mesh->dimension();
		RhsAssembler rhs_assembler(assembler, *mesh,
								   n_bases, size,
								   bases, iso_parametric() ? bases : geom_bases, ass_vals_cache,
								   formulation(), *problem,
								   args["rhs_solver_type"], args["rhs_precond_type"], rhs_solver_params);

		Eigen::VectorXd tmp_sol;

		sol.resizeLike(rhs);
		sol.setZero();

		// if (args["use_al"] || args["has_collision"])
		// {
		//FD
		{
			// 	ALNLProblem nl_problem(*this, rhs_assembler, 1, args["dhat"], false, 1e6);
			// 	tmp_sol = rhs;
			// 	tmp_sol.setRandom();
			// 	// tmp_sol.setOnes();
			// 	Eigen::Matrix<double, Eigen::Dynamic, 1> actual_grad;
			// 	nl_problem.gradient(tmp_sol, actual_grad);

			// 	StiffnessMatrix hessian;
			// 	// Eigen::MatrixXd expected_hessian;
			// 	nl_problem.hessian(tmp_sol, hessian);
			// 	// nl_problem.finiteGradient(tmp_sol, expected_grad, 0);

			// 	// Eigen::MatrixXd actual_hessian = Eigen::MatrixXd(hessian);
			// 	// 	// std::cout << "hhh\n"<< actual_hessian<<std::endl;

			// 	for (int i = 0; i < hessian.rows(); ++i)
			// 	{
			// 		double hhh = 1e-6;
			// 		VectorXd xp = tmp_sol;
			// 		xp(i) += hhh;
			// 		VectorXd xm = tmp_sol;
			// 		xm(i) -= hhh;

			// 		Eigen::Matrix<double, Eigen::Dynamic, 1> tmp_grad_p;
			// 		nl_problem.gradient(xp, tmp_grad_p);

			// 		Eigen::Matrix<double, Eigen::Dynamic, 1> tmp_grad_m;
			// 		nl_problem.gradient(xm, tmp_grad_m);

			// 		Eigen::Matrix<double, Eigen::Dynamic, 1> fd_h = (tmp_grad_p - tmp_grad_m) / (hhh * 2.);

			// 		const double vp = nl_problem.value(xp);
			// 		const double vm = nl_problem.value(xm);

			// 		const double fd = (vp - vm) / (hhh * 2.);
			// 		const double diff = std::abs(actual_grad(i) - fd);
			// 		if (diff > 1e-5)
			// 			std::cout << "diff grad " << i << ": " << actual_grad(i) << " vs " << fd << " error: " << diff << " rrr: " << actual_grad(i) / fd << std::endl;

			// 		for (int j = 0; j < hessian.rows(); ++j)
			// 		{
			// 			const double diff = std::abs(hessian.coeffRef(i, j) - fd_h(j));

			// 			if (diff > 1e-4)
			// 				std::cout << "diff H " << i << ", " << j << ": " << hessian.coeffRef(i, j) << " vs " << fd_h(j) << " error: " << diff << " rrr: " << hessian.coeffRef(i, j) / fd_h(j) << std::endl;
			// 		}
			// 	}

			// 	// 	// std::cout<<"diff grad max "<<(actual_grad - expected_grad).array().abs().maxCoeff()<<std::endl;
			// 	// 	// std::cout<<"diff \n"<<(actual_grad - expected_grad)<<std::endl;
			// 	exit(0);
		}

		ALNLProblem alnl_problem(*this, rhs_assembler, 1, args["dhat"], args["project_to_psd"], args["al_weight"]);
		NLProblem nl_problem(*this, rhs_assembler, 1, args["dhat"], args["project_to_psd"]);

		double al_weight = args["al_weight"];
		const double max_al_weight = args["max_al_weight"];
		nl_problem.full_to_reduced(sol, tmp_sol);

		//TODO: maybe add linear solver here?

		solver_info = json::array();

		int index = 0;
		nl_problem.line_search_begin(sol, tmp_sol);
		while (!std::isfinite(nl_problem.value(tmp_sol)) || !nl_problem.is_step_valid(sol, tmp_sol) || !nl_problem.is_step_collision_free(sol, tmp_sol))
		{
			nl_problem.line_search_end();
			alnl_problem.set_weight(al_weight);
			logger().debug("Solving AL Problem with weight {}", al_weight);

			cppoptlib::SparseNewtonDescentSolver<ALNLProblem> alnlsolver(solver_params(), solver_type(), precond_type());
			alnlsolver.setLineSearch(args["line_search"]);
			alnl_problem.init(sol);
			tmp_sol = sol;
			alnlsolver.minimize(alnl_problem, tmp_sol);
			json alnl_solver_info;
			alnlsolver.getInfo(alnl_solver_info);

			solver_info.push_back({{"type", "al"},
								   {"weight", al_weight},
								   {"info", alnl_solver_info}});

			sol = tmp_sol;
			nl_problem.full_to_reduced(sol, tmp_sol);

			al_weight *= 2;

			if (al_weight >= max_al_weight)
			{
				logger().error("Unable to solve AL problem, weight {} >= {}, stopping", al_weight, max_al_weight);
				break;
			}

			if (args["save_solve_sequence_debug"])
			{
				if (!solve_export_to_file)
					solution_frames.emplace_back();
				save_vtu(fmt::format("step_{:d}.vtu", index), 1);
				save_wire(fmt::format("step_{:d}.obj", index));
			}
			++index;
		}
		nl_problem.line_search_end();
		logger().debug("Solving Problem");
		cppoptlib::SparseNewtonDescentSolver<NLProblem> nlsolver(solver_params(), solver_type(), precond_type());
		nlsolver.setLineSearch(args["line_search"]);
		nl_problem.init(sol);
		nlsolver.minimize(nl_problem, tmp_sol);
		json nl_solver_info;
		nlsolver.getInfo(nl_solver_info);

		nl_problem.reduced_to_full(tmp_sol, sol);
		solver_info.push_back({{"type", "rc"},
							   {"info", nl_solver_info}});
		// }
		// else
		// {
		// 	int steps = args["nl_solver_rhs_steps"];
		// 	if (steps <= 0)
		// 	{
		// 		RowVectorNd min, max;
		// 		mesh->bounding_box(min, max);
		// 		steps = problem->n_incremental_load_steps((max - min).norm());
		// 	}
		// 	steps = std::max(steps, 1);

		// 	double step_t = 1.0 / steps;
		// 	double t = step_t;
		// 	double prev_t = 0;

		// 	StiffnessMatrix nlstiffness;
		// 	Eigen::VectorXd b;
		// 	Eigen::MatrixXd grad;
		// 	Eigen::MatrixXd prev_rhs;

		// 	prev_rhs.resizeLike(rhs);
		// 	prev_rhs.setZero();

		// 	b.resizeLike(sol);
		// 	b.setZero();

		// 	if (args["save_solve_sequence"])
		// 	{
		// 		if (!solve_export_to_file)
		// 			solution_frames.emplace_back();
		// 		save_vtu(fmt::format("step_{:d}.vtu", prev_t), 1);
		// 		save_wire(fmt::format("step_{:d}.obj", prev_t));
		// 	}

		// 	igl::Timer update_timer;
		// 	auto solver = polysolve::LinearSolver::create(args["solver_type"], args["precond_type"]);

		// 	while (t <= 1)
		// 	{
		// 		if (step_t < 1e-10)
		// 		{
		// 			logger().error("Step too small, giving up");
		// 			break;
		// 		}

		// 		logger().info("t: {} prev: {} step: {}", t, prev_t, step_t);

		// 		NLProblem nl_problem(*this, rhs_assembler, t, args["dhat"], args["project_to_psd"]);

		// 		logger().debug("Updating starting point...");
		// 		update_timer.start();

		// 		{
		// 			nl_problem.hessian_full(sol, nlstiffness);
		// 			nl_problem.gradient_no_rhs(sol, grad);
		// 			rhs_assembler.set_bc(local_boundary, boundary_nodes, args["n_boundary_samples"], local_neumann_boundary, grad, t);

		// 			b = grad;
		// 			for (int bId : boundary_nodes)
		// 				b(bId) = -(nl_problem.current_rhs()(bId) - prev_rhs(bId));
		// 			dirichlet_solve(*solver, nlstiffness, b, boundary_nodes, x, precond_num, args["export"]["stiffness_mat"], args["export"]["spectrum"]);
		// 			logger().trace("Checking step");
		// 			const bool valid = nl_problem.is_step_collision_free(sol, (sol - x).eval());
		// 			if (valid)
		// 				x = sol - x;
		// 			else
		// 				x = sol;
		// 			logger().trace("Done checking step, was {}valid", valid ? "" : "in");
		// 			// logger().debug("Solver error: {}", (nlstiffness * sol - b).norm());
		// 		}

		// 		nl_problem.full_to_reduced(x, tmp_sol);
		// 		update_timer.stop();
		// 		logger().debug("done!, took {}s", update_timer.getElapsedTime());

		// 		if (args["save_solve_sequence_debug"])
		// 		{
		// 			Eigen::MatrixXd xxx = sol;
		// 			sol = x;
		// 			if (assembler.is_mixed(formulation()))
		// 				sol_to_pressure();
		// 			if (!solve_export_to_file)
		// 				solution_frames.emplace_back();

		// 			save_vtu(fmt::format("step_s_{:d}.vtu", t), 1);
		// 			save_wire(fmt::format("step_s_{:d}.obj", t));

		// 			sol = xxx;
		// 		}

		// 		bool has_nan = false;
		// 		for (int k = 0; k < tmp_sol.size(); ++k)
		// 		{
		// 			if (std::isnan(tmp_sol[k]))
		// 			{
		// 				has_nan = true;
		// 				break;
		// 			}
		// 		}

		// 		if (has_nan)
		// 		{
		// 			do
		// 			{
		// 				step_t /= 2;
		// 				t = prev_t + step_t;
		// 			} while (t >= 1);
		// 			continue;
		// 		}

		// 		if (args["nl_solver"] == "newton")
		// 		{
		// 			cppoptlib::SparseNewtonDescentSolver<NLProblem> nlsolver(solver_params(), solver_type(), precond_type());
		// 			nlsolver.setLineSearch(args["line_search"]);
		// 			nl_problem.init(x);
		// 			nlsolver.minimize(nl_problem, tmp_sol);

		// 			if (nlsolver.error_code() == -10) //Nan
		// 			{
		// 				do
		// 				{
		// 					step_t /= 2;
		// 					t = prev_t + step_t;
		// 				} while (t >= 1);
		// 				continue;
		// 			}
		// 			else
		// 			{
		// 				prev_t = t;
		// 				step_t *= 2;
		// 			}

		// 			if (step_t > 1.0 / steps)
		// 				step_t = 1.0 / steps;

		// 			nlsolver.getInfo(solver_info);
		// 		}
		// 		else if (args["nl_solver"] == "lbfgs")
		// 		{
		// 			cppoptlib::LbfgsSolverL2<NLProblem> nlsolver;
		// 			nlsolver.setLineSearch(args["line_search"]);
		// 			nlsolver.setDebug(cppoptlib::DebugLevel::High);
		// 			nlsolver.minimize(nl_problem, tmp_sol);

		// 			prev_t = t;
		// 		}
		// 		else
		// 		{
		// 			throw std::invalid_argument("[State] invalid solver type for non-linear problem");
		// 		}

		// 		t = prev_t + step_t;
		// 		if ((prev_t < 1 && t > 1) || abs(t - 1) < 1e-10)
		// 			t = 1;

		// 		nl_problem.reduced_to_full(tmp_sol, sol);

		// 		// std::ofstream of("sol.txt");
		// 		// of<<sol<<std::endl;
		// 		// of.close();
		// 		prev_rhs = nl_problem.current_rhs();
		// 		if (args["save_solve_sequence"])
		// 		{
		// 			if (!solve_export_to_file)
		// 				solution_frames.emplace_back();
		// 			save_vtu(fmt::format("step_{:d}.vtu", prev_t), 1);
		// 			save_wire(fmt::format("step_{:d}.obj", prev_t));
		// 		}
		// 	}

		// 	if (assembler.is_mixed(formulation()))
		// 	{
		// 		sol_to_pressure();
		// 	}

		// 	// {
		// 	// 	boundary_nodes.clear();
		// 	// 	NLProblem nl_problem(*this, rhs_assembler, 1, args["dhat"]);
		// 	// 	tmp_sol = rhs;

		// 	// 	// tmp_sol.setRandom();
		// 	// 	tmp_sol.setOnes();
		// 	// 	Eigen::Matrix<double, Eigen::Dynamic, 1> actual_grad;
		// 	// 	nl_problem.gradient(tmp_sol, actual_grad);

		// 	// 	StiffnessMatrix hessian;
		// 	// 	// Eigen::MatrixXd expected_hessian;
		// 	// 	nl_problem.hessian(tmp_sol, hessian);
		// 	// 	// nl_problem.finiteGradient(tmp_sol, expected_grad, 0);

		// 	// 	Eigen::MatrixXd actual_hessian = Eigen::MatrixXd(hessian);
		// 	// 	// std::cout << "hhh\n"<< actual_hessian<<std::endl;

		// 	// 	for (int i = 0; i < actual_hessian.rows(); ++i)
		// 	// 	{
		// 	// 		double hhh = 1e-6;
		// 	// 		VectorXd xp = tmp_sol; xp(i) += hhh;
		// 	// 		VectorXd xm = tmp_sol; xm(i) -= hhh;

		// 	// 		Eigen::Matrix<double, Eigen::Dynamic, 1> tmp_grad_p;
		// 	// 		nl_problem.gradient(xp, tmp_grad_p);

		// 	// 		Eigen::Matrix<double, Eigen::Dynamic, 1> tmp_grad_m;
		// 	// 		nl_problem.gradient(xm, tmp_grad_m);

		// 	// 		Eigen::Matrix<double, Eigen::Dynamic, 1> fd_h = (tmp_grad_p - tmp_grad_m)/(hhh*2.);

		// 	// 		const double vp = nl_problem.value(xp);
		// 	// 		const double vm = nl_problem.value(xm);

		// 	// 		const double fd = (vp-vm)/(hhh*2.);
		// 	// 		const double  diff = std::abs(actual_grad(i) - fd);
		// 	// 		if(diff > 1e-5)
		// 	// 			std::cout<<"diff grad "<<i<<": "<<actual_grad(i)<<" vs "<<fd <<" error: " <<diff<<" rrr: "<<actual_grad(i)/fd<<std::endl;

		// 	// 		for(int j = 0; j < actual_hessian.rows(); ++j)
		// 	// 		{
		// 	// 			const double diff = std::abs(actual_hessian(i,j) - fd_h(j));

		// 	// 			if(diff > 1e-4)
		// 	// 				std::cout<<"diff H "<<i<<", "<<j<<": "<<actual_hessian(i,j)<<" vs "<<fd_h(j)<<" error: " <<diff<<" rrr: "<<actual_hessian(i,j)/fd_h(j)<<std::endl;

		// 	// 		}
		// 	// 	}

		// 	// 	// std::cout<<"diff grad max "<<(actual_grad - expected_grad).array().abs().maxCoeff()<<std::endl;
		// 	// 	// std::cout<<"diff \n"<<(actual_grad - expected_grad)<<std::endl;
		// 	// 	exit(0);
		// 	// }

		// 	// NLProblem::reduced_to_full_aux(full_size, reduced_size, tmp_sol, rhs, sol);
		// }
	}

} // namespace polyfem
