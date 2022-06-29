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

		auto solver = polysolve::LinearSolver::create(
			args["solver"]["linear"]["solver"], args["solver"]["linear"]["precond"]);
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

} // namespace polyfem
