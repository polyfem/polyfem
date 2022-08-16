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

#include <polyfem/autogen/auto_p_bases.hpp>
#include <polyfem/autogen/auto_q_bases.hpp>

#include <ipc/ipc.hpp>

#include <igl/write_triangle_mesh.h>

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

	void State::init_transient(Eigen::VectorXd &c_sol)
	{
		igl::Timer td_timer;
		td_timer.start();
		logger().trace("Setup rhs...");

		const auto &gbases = iso_parametric() ? bases : geom_bases;
		json rhs_solver_params = args["solver"]["linear"];
		if (!rhs_solver_params.contains("Pardiso"))
			rhs_solver_params["Pardiso"] = {};
		rhs_solver_params["Pardiso"]["mtype"] = -2; // matrix type for Pardiso (2 = SPD)

		step_data.rhs_assembler = std::make_shared<RhsAssembler>(
			assembler, *mesh, obstacle, input_dirichelt,
			n_bases, problem->is_scalar() ? 1 : mesh->dimension(),
			bases, gbases, ass_vals_cache,
			formulation(), *problem,
			args["space"]["advanced"]["bc_method"],
			args["solver"]["linear"]["solver"], args["solver"]["linear"]["precond"], rhs_solver_params);
		RhsAssembler &rhs_assembler = *step_data.rhs_assembler;

		const std::string u_path = resolve_input_path(args["input"]["data"]["u_path"]);
		if (!u_path.empty())
			read_matrix(u_path, sol);
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

		td_timer.stop();
		logger().trace("done, took {}s", td_timer.getElapsedTime());

		save_timestep(0, 0, 0, 0);
	}

	void State::solve_transient_navier_stokes_split(const int time_steps, const double dt, const RhsAssembler &rhs_assembler)
	{
		assert(formulation() == "OperatorSplitting" && problem->is_time_dependent());
		Eigen::MatrixXd local_pts;
		auto &gbases = iso_parametric() ? bases : geom_bases;
		if (mesh->dimension() == 2)
		{
			if (gbases[0].bases.size() == 3)
				autogen::p_nodes_2d(args["space"]["discr_order"], local_pts);
			else
				autogen::q_nodes_2d(args["space"]["discr_order"], local_pts);
		}
		else
		{
			if (gbases[0].bases.size() == 4)
				autogen::p_nodes_3d(args["space"]["discr_order"], local_pts);
			else
				autogen::q_nodes_3d(args["space"]["discr_order"], local_pts);
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
		//TODO fix me
		const double viscosity_ = -1; //build_json_params()["viscosity"];

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

		OperatorSplittingSolver ss(*mesh, shape, n_el, local_boundary, boundary_nodes, pressure_boundary_nodes, bnd_nodes, mass, stiffness_viscosity, stiffness, velocity_mass, dt, viscosity_, args["solver"]["linear"]["solver"], args["solver"]["linear"]["precond"], args["solver"]["linear"], args["output"]["data"]["stiffness_mat"]);

		/* initialize solution */
		pressure = Eigen::MatrixXd::Zero(n_pressure_bases, 1);

		const int n_b_samples = n_boundary_samples();
		for (int t = 1; t <= time_steps; t++)
		{
			double time = t * dt;
			logger().info("{}/{} steps, t={}s", t, time_steps, time);

			/* advection */
			logger().info("Advection...");
			if (args["space"]["advanced"]["particle"])
				ss.advection_FLIP(*mesh, gbases, bases, sol, dt, local_pts);
			else
				ss.advection(*mesh, gbases, bases, sol, dt, local_pts);
			logger().info("Advection finished!");

			/* apply boundary condition */
			rhs_assembler.set_bc(local_boundary, boundary_nodes, n_b_samples, local_neumann_boundary, sol, time);

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
			rhs_assembler.set_bc(local_boundary, boundary_nodes, n_b_samples, local_neumann_boundary, sol, time);

			/* export to vtu */
			save_timestep(time, t, 0, dt);
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

		BDF time_integrator;
		time_integrator.set_parameters(args["time"]["BDF"]);
		time_integrator.init(c_sol, Eigen::VectorXd::Zero(c_sol.size()), Eigen::VectorXd::Zero(c_sol.size()), dt);

		assembler.assemble_problem(formulation(), mesh->is_volume(), n_bases, bases, gbases, ass_vals_cache, velocity_stiffness);
		assembler.assemble_mixed_problem(formulation(), mesh->is_volume(), n_pressure_bases, n_bases, pressure_bases, bases, gbases, pressure_ass_vals_cache, ass_vals_cache, mixed_stiffness);
		assembler.assemble_pressure_problem(formulation(), mesh->is_volume(), n_pressure_bases, pressure_bases, gbases, pressure_ass_vals_cache, pressure_stiffness);

		TransientNavierStokesSolver ns_solver(args["solver"]);
		const int n_larger = n_pressure_bases + (use_avg_pressure ? 1 : 0);

		const int n_b_samples = n_boundary_samples();

		for (int t = 1; t <= time_steps; ++t)
		{
			double time = t0 + t * dt;
			double current_dt = dt;

			logger().info("{}/{} steps, dt={}s t={}s", t, time_steps, current_dt, time);

			prev_sol = time_integrator.weighted_sum_x_prevs();
			rhs_assembler.compute_energy_grad(local_boundary, boundary_nodes, density, n_b_samples, local_neumann_boundary, rhs, time, current_rhs);
			rhs_assembler.set_bc(local_boundary, boundary_nodes, n_b_samples, local_neumann_boundary, current_rhs, time);

			const int prev_size = current_rhs.size();
			if (prev_size != rhs.size())
			{
				current_rhs.conservativeResize(prev_size + n_larger, current_rhs.cols());
				current_rhs.block(prev_size, 0, n_larger, current_rhs.cols()).setZero();
			}

			ns_solver.minimize(*this, sqrt(time_integrator.acceleration_scaling()), prev_sol,
							   velocity_stiffness, mixed_stiffness, pressure_stiffness,
							   velocity_mass, current_rhs, c_sol);
			time_integrator.update_quantities(c_sol);
			sol = c_sol;
			sol_to_pressure();

			save_timestep(time, t, t0, dt);
		}
	}

	void State::solve_transient_scalar(const int time_steps, const double t0, const double dt, const RhsAssembler &rhs_assembler, Eigen::VectorXd &x)
	{
		assert((problem->is_scalar() || assembler.is_mixed(formulation())) && problem->is_time_dependent());

		auto solver = polysolve::LinearSolver::create(args["solver"]["linear"]["solver"], args["solver"]["linear"]["precond"]);
		solver->setParameters(args["solver"]["linear"]);
		logger().info("{}...", solver->name());

		StiffnessMatrix A;
		Eigen::VectorXd b;
		Eigen::MatrixXd current_rhs = rhs;

		BDF time_integrator;
		time_integrator.set_parameters(args["time"]["BDF"]);
		time_integrator.init(x, Eigen::VectorXd::Zero(x.size()), Eigen::VectorXd::Zero(x.size()), dt);

		const int problem_dim = problem->is_scalar() ? 1 : mesh->dimension();
		const int precond_num = problem_dim * n_bases;

		const int n_b_samples = n_boundary_samples();

		for (int t = 1; t <= time_steps; ++t)
		{
			double time = t0 + t * dt;
			double current_dt = dt;

			logger().info("{}/{} {}s", t, time_steps, time);
			rhs_assembler.compute_energy_grad(local_boundary, boundary_nodes, density, n_b_samples, local_neumann_boundary, rhs, time, current_rhs);
			rhs_assembler.set_bc(local_boundary, boundary_nodes, n_b_samples, local_neumann_boundary, current_rhs, time);

			if (assembler.is_mixed(formulation()))
			{
				// divergence free
				int fluid_offset = use_avg_pressure ? (assembler.is_fluid(formulation()) ? 1 : 0) : 0;
				current_rhs.block(current_rhs.rows() - n_pressure_bases - use_avg_pressure, 0, n_pressure_bases + use_avg_pressure, current_rhs.cols()).setZero();
			}

			A = mass / time_integrator.beta_dt() + stiffness;
			x = time_integrator.weighted_sum_x_prevs();
			b = (mass * x) / time_integrator.beta_dt();
			for (int i : boundary_nodes)
				b[i] = 0;
			b += current_rhs;

			spectrum = dirichlet_solve(*solver, A, b, boundary_nodes, x, precond_num, args["output"]["data"]["stiffness_mat"], t == time_steps && args["output"]["advanced"]["spectrum"], assembler.is_fluid(formulation()), use_avg_pressure);
			time_integrator.update_quantities(x);
			sol = x;

			const auto error = (A * x - b).norm();
			if (error > 1e-4)
				logger().error("Solver error: {}", error);
			else
				logger().debug("Solver error: {}", error);

			if (assembler.is_mixed(formulation()))
			{
				sol_to_pressure();
			}

			save_timestep(time, t, t0, dt);
		}
	}

	void State::solve_transient_tensor_linear(const int time_steps, const double t0, const double dt, const RhsAssembler &rhs_assembler)
	{
		assert(!problem->is_scalar() && assembler.is_linear(formulation()) && !args["contact"]["enabled"] && problem->is_time_dependent());
		assert(!assembler.is_mixed(formulation()));

		auto solver = polysolve::LinearSolver::create(args["solver"]["linear"]["solver"], args["solver"]["linear"]["precond"]);
		solver->setParameters(args["solver"]["linear"]);
		logger().info("{}...", solver->name());

		const std::string v_path = resolve_input_path(args["input"]["data"]["v_path"]);
		const std::string a_path = resolve_input_path(args["input"]["data"]["a_path"]);

		Eigen::MatrixXd velocity, acceleration;

		//TODO offset
		if (!v_path.empty())
			import_matrix(v_path, args["import"], velocity);
		else
			rhs_assembler.initial_velocity(velocity);
		//TODO offset
		if (!a_path.empty())
			import_matrix(a_path, args["import"], acceleration);
		else
			rhs_assembler.initial_acceleration(acceleration);

		Eigen::MatrixXd current_rhs = rhs;

		const int problem_dim = problem->is_scalar() ? 1 : mesh->dimension();
		const int precond_num = problem_dim * n_bases;

		Eigen::MatrixXd temp, b;
		StiffnessMatrix A;
		Eigen::VectorXd x, btmp;

		auto time_integrator = ImplicitTimeIntegrator::construct_time_integrator(args["time"]["integrator"]);
		time_integrator->set_parameters(args["time"]["BDF"]);
		time_integrator->set_parameters(args["time"]["newmark"]);
		time_integrator->init(sol, velocity, acceleration, dt);

		const int n_b_samples = n_boundary_samples();

		for (int t = 1; t <= time_steps; ++t)
		{
			const double time = t0 + dt * t;

			rhs_assembler.assemble(density, current_rhs, time);
			current_rhs *= -1;
			rhs_assembler.set_bc(std::vector<LocalBoundary>(), std::vector<int>(), n_b_samples, local_neumann_boundary, current_rhs, time);

			current_rhs *= time_integrator->acceleration_scaling();
			current_rhs += mass * time_integrator->x_tilde();
			rhs_assembler.set_bc(local_boundary, boundary_nodes, n_b_samples, std::vector<LocalBoundary>(), current_rhs, time);

			b = current_rhs;
			A = stiffness * time_integrator->acceleration_scaling() + mass;
			btmp = b;
			spectrum = dirichlet_solve(*solver, A, btmp, boundary_nodes, x, precond_num, args["output"]["data"]["stiffness_mat"], t == 1 && args["output"]["advanced"]["spectrum"], assembler.is_fluid(formulation()), use_avg_pressure);
			time_integrator->update_quantities(x);
			sol = x;

			const auto error = (A * x - b).norm();
			if (error > 1e-4)
				logger().error("Solver error: {}", error);
			else
				logger().debug("Solver error: {}", error);

			save_timestep(time, t, t0, dt);
			logger().info("{}/{} t={}", t, time_steps, time);
		}

		{
			const std::string u_out_path = resolve_output_path(args["output"]["data"]["u_path"]);
			const std::string v_out_path = resolve_output_path(args["output"]["data"]["v_path"]);
			const std::string a_out_path = resolve_output_path(args["output"]["data"]["a_path"]);

			if (!u_out_path.empty())
				write_matrix(u_out_path, sol);
			if (!v_out_path.empty())
				write_matrix(v_out_path, velocity);
			if (!a_out_path.empty())
				write_matrix(a_out_path, acceleration);
		}
	}

	void State::solve_transient_tensor_non_linear(const int time_steps, const double t0, const double dt, const RhsAssembler &rhs_assembler)
	{
		solve_transient_tensor_non_linear_init(t0, dt, rhs_assembler);

		for (int t = 1; t <= time_steps; ++t)
		{
			solve_transient_tensor_non_linear_step(t0, dt, t, solver_info);
			logger().info("{}/{}  t={}", t, time_steps, t0 + dt * t);
		}

		const NLProblem &nl_problem = *step_data.nl_problem;

		nl_problem.save_raw(
			resolve_output_path(args["output"]["data"]["u_path"]),
			resolve_output_path(args["output"]["data"]["v_path"]),
			resolve_output_path(args["output"]["data"]["a_path"]));
	}

	void State::solve_transient_tensor_non_linear_init(const double t0, const double dt, const RhsAssembler &rhs_assembler)
	{
		assert(!problem->is_scalar() && (!assembler.is_linear(formulation()) || args["contact"]["enabled"]) && problem->is_time_dependent());
		assert(!assembler.is_mixed(formulation()));

		// FD for debug
		// {
		// 	Eigen::MatrixXd velocity, acceleration;
		// 	boundary_nodes.clear();
		// 	local_boundary.clear();
		// 	// local_neumann_boundary.clear();
		// 	NLProblem nl_problem(*this, rhs_assembler, t0, args["contact"]["dhat"], false);
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
		const std::string v_path = resolve_input_path(args["input"]["data"]["v_path"]);
		const std::string a_path = resolve_input_path(args["input"]["data"]["a_path"]);

		Eigen::MatrixXd velocity, acceleration;

		//TODO import
		if (!v_path.empty())
			import_matrix(v_path, args["import"], velocity);
		else
			rhs_assembler.initial_velocity(velocity);
		//TODO import
		if (!a_path.empty())
			import_matrix(a_path, args["import"], acceleration);
		else
			rhs_assembler.initial_acceleration(acceleration);

		timer.stop();
		logger().trace("done, took {}s", timer.getElapsedTime());

		if (args["contact"]["enabled"])
		{
			timer.start();
			logger().trace("Checking collisions...");
			Eigen::MatrixXd displaced = boundary_nodes_pos + unflatten(sol, mesh->dimension());

			if (ipc::has_intersections(collision_mesh, collision_mesh.vertices(displaced)))
			{
				const std::string msg = "Unable to solve, initial solution has intersections!";
				logger().error(msg);
				igl::write_triangle_mesh(resolve_output_path("intersection.obj"), collision_mesh.vertices(displaced), collision_mesh.faces());
				throw std::runtime_error(msg);
			}

			timer.stop();
			logger().trace("done, took {}s", timer.getElapsedTime());
		}

		timer.start();
		logger().trace("Init time integrators...");

		const int full_size = n_bases * mesh->dimension();
		const int reduced_size = n_bases * mesh->dimension() - boundary_nodes.size();
		VectorXd tmp_sol;

		step_data.nl_problem = std::make_shared<NLProblem>(*this, rhs_assembler, t0 + dt, args["contact"]["dhat"]);
		NLProblem &nl_problem = *step_data.nl_problem;
		nl_problem.init_time_integrator(sol, velocity, acceleration, dt);

		solver_info = json::array();

		double al_weight = args["solver"]["augmented_lagrangian"]["initial_weight"];
		step_data.alnl_problem = std::make_shared<ALNLProblem>(*this, rhs_assembler, t0 + dt, args["contact"]["dhat"], al_weight);
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

		if (args["solver"]["contact"]["friction_iterations"] > 0)
		{
			logger().debug("Lagging iteration 1");
		}

		nl_problem.line_search_begin(sol, tmp_sol);
		bool force_al = args["solver"]["augmented_lagrangian"]["force"];
		while (force_al || !std::isfinite(nl_problem.value(tmp_sol)) || !nl_problem.is_step_valid(sol, tmp_sol) || !nl_problem.is_step_collision_free(sol, tmp_sol))
		{
			force_al = false;
			nl_problem.line_search_end();
			alnl_problem.set_weight(al_weight);
			logger().debug("Solving AL Problem with weight {}", al_weight);

			std::shared_ptr<cppoptlib::NonlinearSolver<ALNLProblem>> alnlsolver = make_nl_solver<ALNLProblem>();
			alnlsolver->setLineSearch(args["solver"]["nonlinear"]["line_search"]["method"]);
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
				const std::string msg = fmt::format("Unable to solve AL problem, weight {} >= {}, stopping", al_weight, max_al_weight);
				logger().error(msg);
				throw std::runtime_error(msg);
				break;
			}
		}
		nl_problem.line_search_end();
		logger().debug("Solving Problem");

		std::shared_ptr<cppoptlib::NonlinearSolver<NLProblem>> nlsolver = make_nl_solver<NLProblem>();
		nlsolver->setLineSearch(args["solver"]["nonlinear"]["line_search"]["method"]);
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
		nl_problem.update_lagging(tmp_sol);
		bool lagging_converged = nl_problem.lagging_converged(tmp_sol);
		for (lag_i = 1; !lagging_converged && lag_i < args["solver"]["contact"]["friction_iterations"]; lag_i++)
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
			nl_problem.update_lagging(tmp_sol);
			lagging_converged = nl_problem.lagging_converged(tmp_sol);
		}

		if (args["solver"]["contact"]["friction_iterations"] > 0)
		{
			logger().log(
				lagging_converged ? spdlog::level::info : spdlog::level::warn,
				"{} {:d} lagging iteration(s) (err={:g} tol={:g})",
				lagging_converged ? "Friction lagging converged using" : "Friction lagging maxed out at",
				lag_i, nl_problem.compute_lagging_error(tmp_sol),
				args["solver"]["contact"]["friction_convergence_tol"].get<double>());
		}

		{
			POLYFEM_SCOPED_TIMER("Update quantities");
			nl_problem.update_quantities(t0 + (t + 1) * dt, sol);
			alnl_problem.update_quantities(t0 + (t + 1) * dt, sol);
		}

		save_timestep(t0 + dt * t, t, t0, dt);
	}

	void State::solve_linear()
	{
		assert(!problem->is_time_dependent());
		assert(assembler.is_linear(formulation()) && !args["contact"]["enabled"]);
		auto solver = polysolve::LinearSolver::create(args["solver"]["linear"]["solver"], args["solver"]["linear"]["precond"]);
		solver->setParameters(args["solver"]["linear"]);
		StiffnessMatrix A;
		Eigen::VectorXd b;
		logger().info("{}...", solver->name());
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

	void State::solve_non_linear()
	{
		assert(!problem->is_time_dependent());
		assert(!assembler.is_linear(formulation()) || args["contact"]["enabled"]);

		const int full_size = n_bases * mesh->dimension();
		const int reduced_size = n_bases * mesh->dimension() - boundary_nodes.size();

		const int problem_dim = problem->is_scalar() ? 1 : mesh->dimension();
		const int precond_num = problem_dim * n_bases;

		const auto &gbases = iso_parametric() ? bases : geom_bases;

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

		Eigen::VectorXd tmp_sol;

		if (sol.size() != rhs.size())
		{
			sol.resizeLike(rhs);
			sol.setZero();
		}

		const std::string u_path = resolve_input_path(args["input"]["data"]["u_path"]);
		//TODO fix import
		if (!u_path.empty())
			import_matrix(u_path, args["input"]["data"]["u_path"], sol);

		// if (args["use_al"] || args["contact"]["enabled"])
		// {
		// FD
		{
			// 	ALNLProblem nl_problem(*this, rhs_assembler, 1, args["contact"]["dhat"], 1e6);
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

		step_data.nl_problem = std::make_shared<NLProblem>(*this, rhs_assembler, 1, args["contact"]["dhat"]);
		NLProblem &nl_problem = *(step_data.nl_problem);
		step_data.alnl_problem = std::make_shared<ALNLProblem>(*this, rhs_assembler, 1, args["contact"]["dhat"], args["solver"]["augmented_lagrangian"]["initial_weight"]);
		ALNLProblem &alnl_problem = *(step_data.alnl_problem);

		double al_weight = args["solver"]["augmented_lagrangian"]["initial_weight"];
		const double max_al_weight = args["solver"]["augmented_lagrangian"]["max_weight"];
		nl_problem.full_to_reduced(sol, tmp_sol);

		nl_problem.init_lagging(sol);
		alnl_problem.init_lagging(sol);

		const int friction_iterations = args["solver"]["contact"]["friction_iterations"];
		assert(friction_iterations >= 0);
		if (friction_iterations > 0)
		{
			logger().debug("Lagging iteration 1");
		}

		// Disable damping for the final lagged iteration
		if (friction_iterations <= 1)
		{
			nl_problem.lagged_damping_weight() = 0;
			alnl_problem.lagged_damping_weight() = 0;
		}

		// TODO: maybe add linear solver here?

		solver_info = json::array();

		int index = 0;

		if (args["output"]["advanced"]["save_solve_sequence_debug"])
		{
			if (!solve_export_to_file)
				solution_frames.emplace_back();
			save_vtu(resolve_output_path(fmt::format("step_{:d}.vtu", index)), 1);
		}

		nl_problem.line_search_begin(sol, tmp_sol);
		bool force_al = args["solver"]["augmented_lagrangian"]["force"];
		while (force_al || !std::isfinite(nl_problem.value(tmp_sol)) || !nl_problem.is_step_valid(sol, tmp_sol) || !nl_problem.is_step_collision_free(sol, tmp_sol))
		{
			force_al = false;
			nl_problem.line_search_end();
			alnl_problem.set_weight(al_weight);
			logger().debug("Solving AL Problem with weight {}", al_weight);

			std::shared_ptr<cppoptlib::NonlinearSolver<ALNLProblem>> alnlsolver = make_nl_solver<ALNLProblem>();
			alnlsolver->setLineSearch(args["solver"]["nonlinear"]["line_search"]["method"]);
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

			++index;
			if (args["output"]["advanced"]["save_solve_sequence_debug"])
			{
				if (!solve_export_to_file)
					solution_frames.emplace_back();
				save_vtu(resolve_output_path(fmt::format("step_{:d}.vtu", index)), 1);
			}
		}
		nl_problem.line_search_end();
		logger().debug("Solving Problem");
		std::shared_ptr<cppoptlib::NonlinearSolver<NLProblem>> nlsolver = make_nl_solver<NLProblem>();
		nlsolver->setLineSearch(args["solver"]["nonlinear"]["line_search"]["method"]);
		nl_problem.init(sol);
		nlsolver->minimize(nl_problem, tmp_sol);
		json nl_solver_info;
		nlsolver->getInfo(nl_solver_info);
		solver_info.push_back({{"type", "rc"},
							   {"info", nl_solver_info}});
		nl_problem.reduced_to_full(tmp_sol, sol);

		++index;
		if (args["output"]["advanced"]["save_solve_sequence_debug"])
		{
			if (!solve_export_to_file)
				solution_frames.emplace_back();
			save_vtu(resolve_output_path(fmt::format("step_{:d}.vtu", index)), 1);
		}

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
								   {"lag_i", lag_i},
								   {"info", nl_solver_info}});

			nl_problem.reduced_to_full(tmp_sol, sol);
			nl_problem.update_lagging(tmp_sol);
			lagging_converged = nl_problem.lagging_converged(tmp_sol);

			++index;
			if (args["output"]["advanced"]["save_solve_sequence_debug"])
			{
				if (!solve_export_to_file)
					solution_frames.emplace_back();
				save_vtu(resolve_output_path(fmt::format("step_{:d}.vtu", index)), 1);
			}
		}

		if (friction_iterations > 0)
		{
			logger().log(
				lagging_converged ? spdlog::level::info : spdlog::level::warn,
				"{} {:d} lagging iteration(s) (err={:g} tol={:g})",
				lagging_converged ? "Friction lagging converged using" : "Friction lagging maxed out at",
				lag_i, nl_problem.compute_lagging_error(tmp_sol),
				args["solver"]["contact"]["friction_convergence_tol"].get<double>());
		}

		{
			const std::string u_path = resolve_output_path(args["output"]["data"]["u_path"]);
			if (!u_path.empty())
				write_matrix(u_path, sol);
		}
	}

	////////////////////////////////////////////////////////////////////////
	// Template instantiations
	template std::shared_ptr<cppoptlib::NonlinearSolver<NLProblem>> State::make_nl_solver() const;
	template std::shared_ptr<cppoptlib::NonlinearSolver<ALNLProblem>> State::make_nl_solver() const;
} // namespace polyfem
