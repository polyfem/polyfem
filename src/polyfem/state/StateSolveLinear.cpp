#include <polyfem/State.hpp>

#include <polyfem/time_integrator/ImplicitTimeIntegrator.hpp>
#include <polyfem/time_integrator/BDF.hpp>

#include <polysolve/LinearSolver.hpp>
#include <polysolve/FEMSolver.hpp>

namespace polyfem
{
	using namespace mesh;
	using namespace time_integrator;
	using namespace utils;

	void State::solve_linear()
	{
		assert(!problem->is_time_dependent());
		assert(assembler.is_linear(formulation()) && !args["contact"]["enabled"]);

		auto solver =
			polysolve::LinearSolver::create(args["solver"]["linear"]["solver"], args["solver"]["linear"]["precond"]);
		solver->setParameters(args["solver"]["linear"]);
		logger().info("{}...", solver->name());

		assert(solve_data.rhs_assembler != nullptr);
		solve_data.rhs_assembler->set_bc(
			local_boundary, boundary_nodes, n_boundary_samples(),
			(formulation() != "Bilaplacian") ? local_neumann_boundary : std::vector<LocalBoundary>(), rhs);

		const int problem_dim = problem->is_scalar() ? 1 : mesh->dimension();
		const int precond_num = problem_dim * n_bases;

		StiffnessMatrix A = stiffness;
		Eigen::VectorXd b = rhs;

		Eigen::VectorXd x;
		spectrum = dirichlet_solve(
			*solver, A, b, boundary_nodes, x, precond_num, args["output"]["data"]["stiffness_mat"],
			args["output"]["advanced"]["spectrum"], assembler.is_fluid(formulation()), use_avg_pressure);
		sol = x;

		solver->getInfo(solver_info);

		const auto error = (A * x - b).norm();
		if (error > 1e-4)
			logger().error("Solver error: {}", error);
		else
			logger().debug("Solver error: {}", error);

		if (assembler.is_mixed(formulation()))
			sol_to_pressure();
	}

	void State::solve_transient_scalar(const int time_steps, const double t0, const double dt)
	{
		assert((problem->is_scalar() || assembler.is_mixed(formulation())) && problem->is_time_dependent());

		auto solver =
			polysolve::LinearSolver::create(args["solver"]["linear"]["solver"], args["solver"]["linear"]["precond"]);
		solver->setParameters(args["solver"]["linear"]);
		logger().info("{}...", solver->name());

		StiffnessMatrix A;
		Eigen::VectorXd b;
		Eigen::MatrixXd current_rhs = rhs;

		BDF time_integrator;
		time_integrator.set_parameters(args["time"]["BDF"]);
		time_integrator.init(sol, Eigen::VectorXd::Zero(sol.size()), Eigen::VectorXd::Zero(sol.size()), dt);

		const int problem_dim = problem->is_scalar() ? 1 : mesh->dimension();
		const int precond_num = problem_dim * n_bases;

		const int n_b_samples = n_boundary_samples();

		for (int t = 1; t <= time_steps; ++t)
		{
			double time = t0 + t * dt;
			double current_dt = dt;

			logger().info("{}/{} {}s", t, time_steps, time);
			solve_data.rhs_assembler->compute_energy_grad(
				local_boundary, boundary_nodes, density, n_b_samples, local_neumann_boundary, rhs, time, current_rhs);
			solve_data.rhs_assembler->set_bc(
				local_boundary, boundary_nodes, n_b_samples, local_neumann_boundary, current_rhs, time);

			if (assembler.is_mixed(formulation()))
			{
				// divergence free
				int fluid_offset = use_avg_pressure ? (assembler.is_fluid(formulation()) ? 1 : 0) : 0;
				current_rhs
					.block(
						current_rhs.rows() - n_pressure_bases - use_avg_pressure, 0,
						n_pressure_bases + use_avg_pressure, current_rhs.cols())
					.setZero();
			}

			A = mass / time_integrator.beta_dt() + stiffness;
			sol = time_integrator.weighted_sum_x_prevs();
			b = (mass * sol) / time_integrator.beta_dt();
			for (int i : boundary_nodes)
				b[i] = 0;
			b += current_rhs;

			Eigen::VectorXd x;
			spectrum = dirichlet_solve(
				*solver, A, b, boundary_nodes, x, precond_num, args["output"]["data"]["stiffness_mat"],
				t == time_steps && args["output"]["advanced"]["spectrum"], assembler.is_fluid(formulation()),
				use_avg_pressure);
			sol = x;
			time_integrator.update_quantities(sol);

			const auto error = (A * sol - b).norm();
			if (error > 1e-4)
				logger().error("Solver error: {}", error);
			else
				logger().debug("Solver error: {}", error);

			if (assembler.is_mixed(formulation()))
				sol_to_pressure();

			save_timestep(time, t, t0, dt);
		}
	}

	void State::solve_transient_tensor_linear(const int time_steps, const double t0, const double dt)
	{
		assert(
			!problem->is_scalar() && assembler.is_linear(formulation()) && !args["contact"]["enabled"]
			&& problem->is_time_dependent());
		assert(!assembler.is_mixed(formulation()));
		assert(solve_data.rhs_assembler != nullptr);

		auto solver =
			polysolve::LinearSolver::create(args["solver"]["linear"]["solver"], args["solver"]["linear"]["precond"]);
		solver->setParameters(args["solver"]["linear"]);
		logger().info("{}...", solver->name());

		const std::string v_path = resolve_input_path(args["input"]["data"]["v_path"]);
		const std::string a_path = resolve_input_path(args["input"]["data"]["a_path"]);

		Eigen::MatrixXd velocity, acceleration;

		// TODO offset
		if (!v_path.empty())
			import_matrix(v_path, args["import"], velocity);
		else
			solve_data.rhs_assembler->initial_velocity(velocity);
		// TODO offset
		if (!a_path.empty())
			import_matrix(a_path, args["import"], acceleration);
		else
			solve_data.rhs_assembler->initial_acceleration(acceleration);

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

			solve_data.rhs_assembler->assemble(density, current_rhs, time);
			current_rhs *= -1;
			solve_data.rhs_assembler->set_bc(
				std::vector<LocalBoundary>(), std::vector<int>(), n_b_samples, local_neumann_boundary, current_rhs,
				time);

			current_rhs *= time_integrator->acceleration_scaling();
			current_rhs += mass * time_integrator->x_tilde();
			solve_data.rhs_assembler->set_bc(
				local_boundary, boundary_nodes, n_b_samples, std::vector<LocalBoundary>(), current_rhs, time);

			A = stiffness * time_integrator->acceleration_scaling() + mass;
			b = current_rhs;
			btmp = b;
			spectrum = dirichlet_solve(
				*solver, A, btmp, boundary_nodes, x, precond_num, args["output"]["data"]["stiffness_mat"],
				t == 1 && args["output"]["advanced"]["spectrum"], assembler.is_fluid(formulation()), use_avg_pressure);
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

		std::vector<std::pair<std::string, Eigen::MatrixXd &>> out_data{
			{"u", sol}, {"v", velocity}, {"a", acceleration}};
		for (auto &[name, val] : out_data)
		{
			const std::string out_path = resolve_output_path(args["output"]["data"][fmt::format("{}_path", name)]);
			if (!out_path.empty())
				write_matrix(out_path, val);
		}
	}
} // namespace polyfem
