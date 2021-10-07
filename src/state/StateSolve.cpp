#include <polyfem/State.hpp>

#include <polyfem/BDF.hpp>
#include <polyfem/TransientNavierStokesSolver.hpp>
#include <polyfem/OperatorSplittingSolver.hpp>
#include <polyfem/NavierStokesSolver.hpp>

#include <polyfem/SparseNewtonDescentSolver.hpp>
#include <polyfem/LBFGSSolver.hpp>

#include <polyfem/NLProblem.hpp>
#include <polyfem/ALNLProblem.hpp>

#include <polysolve/LinearSolver.hpp>
#include <polysolve/FEMSolver.hpp>

#include <polyfem/StringUtils.hpp>

#include <polyfem/auto_p_bases.hpp>
#include <polyfem/auto_q_bases.hpp>

#include <ipc/ipc.hpp>

#include <fstream>

namespace polyfem
{
	namespace
	{
		void import_matrix(const std::string &path, const json &import, Eigen::MatrixXd &mat)
		{
			if (import.contains("offset"))
			{
				const int offset = import["offset"];

				Eigen::MatrixXd tmp;
				read_matrix_binary(path, tmp);
				mat.block(0, 0, offset, 1) = tmp.block(0, 0, offset, 1);
			}
			else
			{
				read_matrix_binary(path, mat);
			}
		}
	} // namespace

	template <typename ProblemType>
	std::shared_ptr<cppoptlib::NonlinearSolver<ProblemType>> State::make_nl_solver() const
	{
		std::string name = args["nl_solver"];
		if (name == "newton" || name == "Newton")
		{
			return std::make_shared<cppoptlib::SparseNewtonDescentSolver<ProblemType>>(
				solver_params(), solver_type(), precond_type());
		}
		else if (name == "lbfgs" || name == "LBFGS" || name == "L-BFGS")
		{
			return std::make_shared<cppoptlib::LBFGSSolver<ProblemType>>(
				solver_params());
		}
		else
		{
			throw std::invalid_argument(fmt::format("invalid nonlinear solver type: {}", name));
		}
	}

	void State::init_transient(Eigen::VectorXd &c_sol)
	{
		igl::Timer td_timer;
		td_timer.start();
		logger().trace("Setup rhs...");

		const auto &gbases = iso_parametric() ? bases : geom_bases;
		json rhs_solver_params = args["rhs_solver_params"];
		rhs_solver_params["mtype"] = -2; // matrix type for Pardiso (2 = SPD)

		step_data.rhs_assembler = std::make_shared<RhsAssembler>(
			assembler, *mesh, obstacle,
			n_bases, problem->is_scalar() ? 1 : mesh->dimension(),
			bases, gbases, ass_vals_cache,
			formulation(), *problem,
			args["bc_method"],
			args["rhs_solver_type"], args["rhs_precond_type"], rhs_solver_params);
		RhsAssembler &rhs_assembler = *step_data.rhs_assembler;

		const std::string u_path = resolve_path(args["import"]["u_path"], args["root_path"]);
		if (!u_path.empty())
			read_matrix_binary(u_path, sol);
		else
			rhs_assembler.initial_solution(sol);

		if (assembler.is_mixed(formulation()))
		{
			pressure.resize(0, 0);
			const int prev_size = sol.size();
			sol.conservativeResize(rhs.size(), sol.cols());
			//Zero initial pressure
			sol.block(prev_size, 0, n_pressure_bases, sol.cols()).setZero();
			sol(sol.size() - 1) = 0;
		}

		c_sol = sol;

		if (assembler.is_mixed(formulation()))
			sol_to_pressure();

		td_timer.stop();
		logger().trace("done, took {}s", td_timer.getElapsedTime());

		if (args["save_time_sequence"])
		{
			td_timer.start();
			logger().trace("Saving VTU...");

			if (!solve_export_to_file)
				solution_frames.emplace_back();
			save_vtu(resolve_output_path("step_0.vtu"), 0);
			save_wire(resolve_output_path("step_0.obj"));

			td_timer.stop();
			logger().trace("done, took {}s", td_timer.getElapsedTime());
		}
	}

	void State::solve_transient_navier_stokes_split(const int time_steps, const double dt, const RhsAssembler &rhs_assembler)
	{
		assert(formulation() == "OperatorSplitting" && problem->is_time_dependent());
		const json &params = solver_params();
		Eigen::MatrixXd local_pts;
		auto &gbases = iso_parametric() ? bases : geom_bases;
		if (mesh->dimension() == 2)
		{
			if (gbases[0].bases.size() == 3)
				autogen::p_nodes_2d(args["discr_order"], local_pts);
			else
				autogen::q_nodes_2d(args["discr_order"], local_pts);
		}
		else
		{
			if (gbases[0].bases.size() == 4)
				autogen::p_nodes_3d(args["discr_order"], local_pts);
			else
				autogen::q_nodes_3d(args["discr_order"], local_pts);
		}
		std::vector<int> bnd_nodes;
		bnd_nodes.reserve(boundary_nodes.size() / mesh->dimension());
		for (auto it = boundary_nodes.begin(); it != boundary_nodes.end(); it++)
		{
			if (!(*it % mesh->dimension()))
				continue;
			bnd_nodes.push_back(*it / mesh->dimension());
		}

		const int dim = mesh->dimension();
		const int n_el = int(bases.size());       // number of elements
		const int shape = gbases[0].bases.size(); // number of geometry vertices in an element
		const double viscosity_ = build_json_params()["viscosity"];

		logger().info("Matrices assembly...");
		StiffnessMatrix stiffness_viscosity, mixed_stiffness, velocity_mass;
		// coefficient matrix of viscosity
		assembler.assemble_problem("Laplacian", mesh->is_volume(), n_bases, bases, gbases, ass_vals_cache, stiffness_viscosity);
		assembler.assemble_mass_matrix("Laplacian", mesh->is_volume(), n_bases, density, bases, gbases, ass_vals_cache, mass);

		// coefficient matrix of pressure projection
		assembler.assemble_problem("Laplacian", mesh->is_volume(), n_pressure_bases, pressure_bases, gbases, pressure_ass_vals_cache, stiffness);

		// matrix used to calculate divergence of velocity
		assembler.assemble_mixed_problem("Stokes", mesh->is_volume(), n_pressure_bases, n_bases, pressure_bases, bases, gbases, pressure_ass_vals_cache, ass_vals_cache, mixed_stiffness);
		assembler.assemble_mass_matrix("Stokes", mesh->is_volume(), n_bases, density, bases, gbases, ass_vals_cache, velocity_mass);
		mixed_stiffness = mixed_stiffness.transpose();
		logger().info("Matrices assembly ends!");

		OperatorSplittingSolver ss(*mesh, shape, n_el, local_boundary, boundary_nodes, pressure_boundary_nodes, bnd_nodes, mass, stiffness_viscosity, stiffness, velocity_mass, dt, viscosity_, args["solver_type"], args["precond_type"], params, args["export"]["stiffness_mat"]);

		/* initialize solution */
		pressure = Eigen::MatrixXd::Zero(n_pressure_bases, 1);

		for (int t = 1; t <= time_steps; t++)
		{
			double time = t * dt;
			logger().info("{}/{} steps, t={}s", t, time_steps, time);

			/* advection */
			logger().info("Advection...");
			if (args["particle"])
				ss.advection_FLIP(*mesh, gbases, bases, sol, dt, local_pts);
			else
				ss.advection(*mesh, gbases, bases, sol, dt, local_pts);
			logger().info("Advection finished!");

			/* apply boundary condition */
			rhs_assembler.set_bc(local_boundary, boundary_nodes, args["n_boundary_samples"], local_neumann_boundary, sol, time);

			/* viscosity */
			logger().info("Solving diffusion...");
			if (viscosity_ > 0)
				ss.solve_diffusion_1st(mass, bnd_nodes, sol);
			logger().info("Diffusion solved!");

			/* external force */
			ss.external_force(*mesh, assembler, gbases, bases, dt, sol, local_pts, problem, time);

			/* incompressibility */
			logger().info("Pressure projection...");
			ss.solve_pressure(mixed_stiffness, pressure_boundary_nodes, sol, pressure);

			ss.projection(n_bases, gbases, bases, pressure_bases, local_pts, pressure, sol);
			// ss.projection(velocity_mass, mixed_stiffness, boundary_nodes, sol, pressure);
			logger().info("Pressure projection finished!");

			pressure = pressure / dt;

			/* apply boundary condition */
			rhs_assembler.set_bc(local_boundary, boundary_nodes, args["n_boundary_samples"], local_neumann_boundary, sol, time);

			/* export to vtu */
			if (args["save_time_sequence"] && !(t % args["skip_frame"].get<int>()))
			{
				if (!solve_export_to_file)
					solution_frames.emplace_back();
				save_vtu(resolve_output_path(fmt::format("step_{:d}.vtu", t)), time);
				save_wire(resolve_output_path(fmt::format("step_{:d}.obj", t)));

				save_pvd(
					resolve_output_path(args["export"]["time_sequence"]),
					[](int i) { return fmt::format("step_{:d}.vtu", i); },
					t, /*t0=*/0, dt, args["skip_frame"].get<int>());
			}
		}

		const bool export_surface = args["export"]["surface"];

		if (export_surface)
		{
			save_pvd(
				resolve_output_path("sim_surf.pvd"),
				[](int i) { return fmt::format("step_{:d}_surf.vtu", i); },
				time_steps, /*t0=*/0, dt);
		}
	}

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

			if (args["save_time_sequence"] && !(t % args["skip_frame"].get<int>()))
			{
				if (!solve_export_to_file)
					solution_frames.emplace_back();
				save_vtu(resolve_output_path(fmt::format("step_{:d}.vtu", t)), time);
				save_wire(resolve_output_path(fmt::format("step_{:d}.obj", t)));

				save_pvd(
					resolve_output_path(args["export"]["time_sequence"]),
					[](int i) { return fmt::format("step_{:d}.vtu", i); },
					t, t0, dt, args["skip_frame"].get<int>());
			}
		}

		const bool export_surface = args["export"]["surface"];

		if (export_surface)
		{
			save_pvd(
				resolve_output_path("sim_surf.pvd"),
				[](int i) { return fmt::format("step_{:d}_surf.vtu", i); },
				time_steps, t0, dt);
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

			if (args["save_time_sequence"] && !(t % args["skip_frame"].get<int>()))
			{
				if (!solve_export_to_file)
					solution_frames.emplace_back();

				save_vtu(resolve_output_path(fmt::format("step_{:d}.vtu", t)), time);
				save_wire(resolve_output_path(fmt::format("step_{:d}.obj", t)));

				save_pvd(
					resolve_output_path(args["export"]["time_sequence"]),
					[](int i) { return fmt::format("step_{:d}.vtu", i); },
					t, t0, dt, args["skip_frame"].get<int>());
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
			import_matrix(v_path, args["import"], velocity);
		else
			rhs_assembler.initial_velocity(velocity);
		if (!a_path.empty())
			import_matrix(a_path, args["import"], acceleration);
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

			if (args["save_time_sequence"] && !(t % args["skip_frame"].get<int>()))
			{
				if (!solve_export_to_file)
					solution_frames.emplace_back();
				save_vtu(resolve_output_path(fmt::format("step_{:d}.vtu", t)), t0 + dt * t);
				save_wire(resolve_output_path(fmt::format("step_{:d}.obj", t)));

				save_pvd(
					resolve_output_path(args["export"]["time_sequence"]),
					[](int i) { return fmt::format("step_{:d}.vtu", i); },
					t, t0, dt, args["skip_frame"].get<int>());
			}

			logger().info("{}/{} t={}", t, time_steps, t0 + dt * t);
		}

		{
			const std::string u_out_path = resolve_output_path(args["export"]["u_path"]);
			const std::string v_out_path = resolve_output_path(args["export"]["v_path"]);
			const std::string a_out_path = resolve_output_path(args["export"]["a_path"]);

			if (!u_out_path.empty())
				write_matrix_binary(u_out_path, sol);
			if (!v_out_path.empty())
				write_matrix_binary(v_out_path, velocity);
			if (!a_out_path.empty())
				write_matrix_binary(a_out_path, acceleration);
		}

		const bool export_surface = args["export"]["surface"];

		if (export_surface)
		{
			save_pvd(
				resolve_output_path("sim_surf.pvd"),
				[](int i) { return fmt::format("step_{:d}_surf.vtu", i); },
				time_steps, t0, dt);
		}
	}

	void State::solve_transient_tensor_non_linear(const int time_steps, const double t0, const double dt, const RhsAssembler &rhs_assembler)
	{
		solve_transient_tensor_non_linear_init(t0, dt, rhs_assembler);

		for (int t = 1; t <= time_steps; ++t)
		{
			solve_transient_tensor_non_linear_step(t0, dt, t, solver_info);
			logger().info("{}/{}  t={}", t, time_steps, t0 + dt * t);

			if (args["save_time_sequence"] && !(t % args["skip_frame"].get<int>()))
			{
				save_pvd(
					resolve_output_path(args["export"]["time_sequence"]),
					[](int i) { return fmt::format("step_{:d}.vtu", i); },
					t, t0, dt, args["skip_frame"].get<int>());
			}
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

		// 		if (args["save_time_sequence"] && !(t % args["skip_frame"].get<int>()))
		// 		{
		// 			if (!solve_export_to_file)
		// 				solution_frames.emplace_back();
		// 			save_vtu(fmt::format("step_{:d}.vtu", t), dt * t);
		// 			save_wire(fmt::format("step_{:d}.obj", t));
		// 		}

		// 		logger().info("{}/{}", t, time_steps);
		// 	}
		// }
		const NLProblem &nl_problem = *step_data.nl_problem;

		nl_problem.save_raw(
			resolve_output_path(args["export"]["u_path"]),
			resolve_output_path(args["export"]["v_path"]),
			resolve_output_path(args["export"]["a_path"]));

		const bool export_surface = args["export"]["surface"];
		const bool contact_forces = args["export"]["contact_forces"] && !problem->is_scalar();

		if (export_surface)
		{
			save_pvd(
				resolve_output_path("sim_surf.pvd"),
				[](int i) { return fmt::format("step_{:d}_surf.vtu", i); },
				time_steps, t0, dt);

			if (contact_forces)
			{
				save_pvd(
					resolve_output_path("sim_surf_contact.pvd"),
					[](int i) { return fmt::format("step_{:d}_surf_contact.vtu", i); },
					time_steps, t0, dt);
			}
		}
	}

	void State::solve_transient_tensor_non_linear_init(const double t0, const double dt, const RhsAssembler &rhs_assembler)
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

		igl::Timer timer;
		timer.start();
		logger().trace("Reading matrices...");
		const std::string v_path = resolve_path(args["import"]["v_path"], args["root_path"]);
		const std::string a_path = resolve_path(args["import"]["a_path"], args["root_path"]);

		Eigen::MatrixXd velocity, acceleration;

		if (!v_path.empty())
			import_matrix(v_path, args["import"], velocity);
		else
			rhs_assembler.initial_velocity(velocity);
		if (!a_path.empty())
			import_matrix(a_path, args["import"], acceleration);
		else
			rhs_assembler.initial_acceleration(acceleration);

		timer.stop();
		logger().trace("done, took {}s", timer.getElapsedTime());

		if (args["has_collision"])
		{
			timer.start();
			logger().trace("Checking collisions...");
			const int problem_dim = mesh->dimension();
			Eigen::MatrixXd tmp = boundary_nodes_pos;
			assert(tmp.rows() * problem_dim == sol.size());
			for (int i = 0; i < sol.size(); i += problem_dim)
			{
				for (int d = 0; d < problem_dim; ++d)
				{
					tmp(i / problem_dim, d) += sol(i + d);
				}
			}

			if (ipc::has_intersections(tmp, boundary_edges, boundary_triangles))
			{
				logger().error("Unable to solve, initial solution has intersections!");
				throw "Unable to solve, initial solution has intersections!";
			}

			timer.stop();
			logger().trace("done, took {}s", timer.getElapsedTime());
		}

		timer.start();
		logger().trace("Init time integrators...");

		const int full_size = n_bases * mesh->dimension();
		const int reduced_size = n_bases * mesh->dimension() - boundary_nodes.size();
		VectorXd tmp_sol;

		step_data.nl_problem = std::make_shared<NLProblem>(*this, rhs_assembler, t0 + dt, args["dhat"], args["project_to_psd"]);
		NLProblem &nl_problem = *step_data.nl_problem;
		nl_problem.init_time_integrator(sol, velocity, acceleration, dt);

		solver_info = json::array();

		// if (args["use_al"] || args["has_collision"])
		// {
		double al_weight = args["al_weight"];
		step_data.alnl_problem = std::make_shared<ALNLProblem>(*this, rhs_assembler, t0 + dt, args["dhat"], args["project_to_psd"], al_weight);
		ALNLProblem &alnl_problem = *step_data.alnl_problem;
		alnl_problem.init_time_integrator(sol, velocity, acceleration, dt);

		timer.stop();
		logger().trace("done, took {}s", timer.getElapsedTime());
	}

	void State::solve_transient_tensor_non_linear_step(const double t0, const double dt, const int t, json &solver_info)
	{
		VectorXd tmp_sol;
		NLProblem &nl_problem = *step_data.nl_problem;
		ALNLProblem &alnl_problem = *step_data.alnl_problem;
		igl::Timer timer;

		double al_weight = args["al_weight"];
		const double max_al_weight = args["max_al_weight"];

		nl_problem.full_to_reduced(sol, tmp_sol);
		assert(sol.size() == rhs.size());
		assert(tmp_sol.size() <= rhs.size());

		timer.start();
		logger().trace("Updating lagging...");

		nl_problem.update_lagging(sol, /*start_of_timestep=*/true);
		alnl_problem.update_lagging(sol, /*start_of_timestep=*/true);

		timer.stop();
		logger().trace("done, took {}s", timer.getElapsedTime());

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

			std::shared_ptr<cppoptlib::NonlinearSolver<ALNLProblem>> alnlsolver = make_nl_solver<ALNLProblem>();
			alnlsolver->setLineSearch(args["line_search"]);
			alnl_problem.init(sol);
			tmp_sol = sol;
			alnlsolver->minimize(alnl_problem, tmp_sol);
			json alnl_solver_info;
			alnlsolver->getInfo(alnl_solver_info);

			solver_info.push_back({{"type", "al"},
								   {"t", t},
								   {"weight", al_weight},
								   {"info", alnl_solver_info}});

			sol = tmp_sol;
			nl_problem.full_to_reduced(sol, tmp_sol);
			nl_problem.line_search_begin(sol, tmp_sol);

			al_weight *= 2;

			if (al_weight >= max_al_weight)
			{
				logger().error("Unable to solve AL problem, weight {} >= {}, stopping", al_weight, max_al_weight);
				break;
			}
		}
		nl_problem.line_search_end();
		logger().debug("Solving Problem");

		std::shared_ptr<cppoptlib::NonlinearSolver<NLProblem>> nlsolver = make_nl_solver<NLProblem>();
		nlsolver->setLineSearch(args["line_search"]);
		nl_problem.init(sol);
		nlsolver->minimize(nl_problem, tmp_sol);
		json nl_solver_info;
		nlsolver->getInfo(nl_solver_info);
		solver_info.push_back({{"type", "rc"},
							   {"t", t},
							   {"info", nl_solver_info}});
		nl_problem.reduced_to_full(tmp_sol, sol);

		// Lagging loop (start at 1 because we already did an iteration above)
		int lag_i;
		bool lagging_converged = nl_problem.lagging_converged(tmp_sol, /*do_lagging_update=*/true);
		for (lag_i = 1; !lagging_converged && lag_i < args["friction_iterations"]; lag_i++)
		{
			logger().debug("Lagging iteration {:d}", lag_i + 1);
			nl_problem.init(sol);
			nlsolver->minimize(nl_problem, tmp_sol);

			nlsolver->getInfo(nl_solver_info);
			solver_info.push_back({{"type", "rc"},
								   {"t", t},
								   {"lag_i", lag_i},
								   {"info", nl_solver_info}});

			nl_problem.reduced_to_full(tmp_sol, sol);
			lagging_converged = nl_problem.lagging_converged(tmp_sol, /*do_lagging_update=*/true);
		}

		if (args["friction_iterations"] > 0)
		{
			logger().log(
				lagging_converged ? spdlog::level::info : spdlog::level::warn,
				"{} {:d} lagging iteration(s) (err={:g} tol={:g})",
				lagging_converged ? "Friction lagging converged using" : "Friction lagging maxed out at",
				lag_i, nl_problem.compute_lagging_error(tmp_sol, /*do_lagging_update=*/false),
				args["friction_convergence_tol"].get<double>());
		}

		timer.start();
		logger().trace("Update quantities...");

		nl_problem.update_quantities(t0 + (t + 1) * dt, sol);
		alnl_problem.update_quantities(t0 + (t + 1) * dt, sol);
		timer.stop();
		logger().trace("done, took {}s", timer.getElapsedTime());

		if (args["save_time_sequence"] && !(t % args["skip_frame"].get<int>()))
		{
			timer.start();
			logger().trace("Saving VTU...");

			if (!solve_export_to_file)
				solution_frames.emplace_back();
			save_vtu(resolve_output_path(fmt::format("step_{:d}.vtu", t)), t0 + dt * t);
			save_wire(resolve_output_path(fmt::format("step_{:d}.obj", t)));

			timer.stop();
			logger().trace("done, took {}s", timer.getElapsedTime());
		}
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
		RhsAssembler rhs_assembler(assembler, *mesh, obstacle,
								   n_bases, size,
								   bases, iso_parametric() ? bases : geom_bases, ass_vals_cache,
								   formulation(), *problem,
								   args["bc_method"],
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

		RhsAssembler rhs_assembler(assembler, *mesh, obstacle,
								   n_bases, mesh->dimension(),
								   bases, iso_parametric() ? bases : geom_bases, ass_vals_cache,
								   formulation(), *problem,
								   args["bc_method"],
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
		RhsAssembler rhs_assembler(assembler, *mesh, obstacle,
								   n_bases, size,
								   bases, iso_parametric() ? bases : geom_bases, ass_vals_cache,
								   formulation(), *problem,
								   args["bc_method"],
								   args["rhs_solver_type"], args["rhs_precond_type"], rhs_solver_params);

		Eigen::VectorXd tmp_sol;

		sol.resizeLike(rhs);
		sol.setZero();

		const std::string u_path = resolve_path(args["import"]["u_path"], args["root_path"]);
		if (!u_path.empty())
			import_matrix(u_path, args["import"], sol);

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

		nl_problem.update_lagging(sol, /*start_of_timestep=*/true);
		alnl_problem.update_lagging(sol, /*start_of_timestep=*/true);

		//TODO: maybe add linear solver here?

		solver_info = json::array();

		int index = 0;
		nl_problem.line_search_begin(sol, tmp_sol);
		while (!std::isfinite(nl_problem.value(tmp_sol)) || !nl_problem.is_step_valid(sol, tmp_sol) || !nl_problem.is_step_collision_free(sol, tmp_sol))
		{
			nl_problem.line_search_end();
			alnl_problem.set_weight(al_weight);
			logger().debug("Solving AL Problem with weight {}", al_weight);

			std::shared_ptr<cppoptlib::NonlinearSolver<ALNLProblem>> alnlsolver = make_nl_solver<ALNLProblem>();
			alnlsolver->setLineSearch(args["line_search"]);
			alnl_problem.init(sol);
			tmp_sol = sol;
			alnlsolver->minimize(alnl_problem, tmp_sol);
			json alnl_solver_info;
			alnlsolver->getInfo(alnl_solver_info);

			solver_info.push_back({{"type", "al"},
								   {"weight", al_weight},
								   {"info", alnl_solver_info}});

			sol = tmp_sol;
			nl_problem.full_to_reduced(sol, tmp_sol);
			nl_problem.line_search_begin(sol, tmp_sol);

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
		std::shared_ptr<cppoptlib::NonlinearSolver<NLProblem>> nlsolver = make_nl_solver<NLProblem>();
		nlsolver->setLineSearch(args["line_search"]);
		nl_problem.init(sol);
		nlsolver->minimize(nl_problem, tmp_sol);
		json nl_solver_info;
		nlsolver->getInfo(nl_solver_info);

		nl_problem.reduced_to_full(tmp_sol, sol);
		solver_info.push_back({{"type", "rc"},
							   {"info", nl_solver_info}});

		{
			const std::string u_path = resolve_path(args["export"]["u_path"], args["root_path"]);
			if (!u_path.empty())
				write_matrix_binary(u_path, sol);
		}
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

	////////////////////////////////////////////////////////////////////////
	// Template instantiations
	template std::shared_ptr<cppoptlib::NonlinearSolver<NLProblem>> State::make_nl_solver() const;
	template std::shared_ptr<cppoptlib::NonlinearSolver<ALNLProblem>> State::make_nl_solver() const;
} // namespace polyfem
