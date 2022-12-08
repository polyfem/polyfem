#include <polyfem/State.hpp>

#include <polyfem/time_integrator/ImplicitTimeIntegrator.hpp>
#include <polyfem/time_integrator/BDF.hpp>

#include <polyfem/solver/forms/BodyForm.hpp>
#include <polyfem/solver/forms/ElasticForm.hpp>
#include <polyfem/solver/forms/InertiaForm.hpp>

#include <polyfem/utils/Timer.hpp>

#include <polysolve/FEMSolver.hpp>

namespace polyfem
{
	using namespace mesh;
	using namespace time_integrator;
	using namespace utils;
	using namespace solver;
	using namespace io;

	void State::solve_linear(
		const std::unique_ptr<polysolve::LinearSolver> &solver,
		StiffnessMatrix &A,
		Eigen::VectorXd &b,
		const bool compute_spectrum,
		Eigen::MatrixXd &sol, Eigen::MatrixXd &pressure)
	{
		assert(assembler.is_linear(formulation()) && !is_contact_enabled());
		assert(solve_data.rhs_assembler != nullptr);

		const int problem_dim = problem->is_scalar() ? 1 : mesh->dimension();
		const int full_size = A.rows();
		int precond_num = problem_dim * n_bases;

		apply_lagrange_multipliers(A);
		b.conservativeResizeLike(Eigen::VectorXd::Zero(A.rows()));

		std::vector<int> boundary_nodes_tmp = boundary_nodes;
		full_to_periodic(boundary_nodes_tmp);
		if (need_periodic_reduction())
		{
			precond_num = full_to_periodic(A);
 			Eigen::MatrixXd tmp = b;
 			full_to_periodic(tmp, true);
 			b = tmp;
		}

		Eigen::VectorXd x;
		if (args["optimization"]["enabled"])
		{
			auto A_tmp = A;
			prefactorize(*solver, A, boundary_nodes_tmp, precond_num, args["output"]["data"]["stiffness_mat"]);
			dirichlet_solve_prefactorized(*solver, A_tmp, b, boundary_nodes_tmp, x);
		}
		else
		{
			stats.spectrum = dirichlet_solve(
				*solver, A, b, boundary_nodes_tmp, x, precond_num, args["output"]["data"]["stiffness_mat"], compute_spectrum,
				assembler.is_fluid(formulation()), use_avg_pressure);
		}

		solver->getInfo(stats.solver_info);

		const auto error = (A * x - b).norm();
		if (error > 1e-4)
			logger().error("Solver error: {}", error);
		else
			logger().debug("Solver error: {}", error);

		x.conservativeResize(x.size() - n_lagrange_multipliers());
 		if (need_periodic_reduction())
 			sol = periodic_to_full(full_size, x);
 		else
 			sol = x; // Explicit copy because sol is a MatrixXd (with one column)

		if (assembler.is_mixed(formulation()))
			sol_to_pressure(sol, pressure);
	}

	void State::solve_linear(Eigen::MatrixXd &sol, Eigen::MatrixXd &pressure)
	{
		assert(!problem->is_time_dependent());
		assert(assembler.is_linear(formulation()) && !is_contact_enabled());

		// --------------------------------------------------------------------
		if (lin_solver_cached)
			lin_solver_cached.reset();
		
		lin_solver_cached =
			polysolve::LinearSolver::create(args["solver"]["linear"]["solver"], args["solver"]["linear"]["precond"]);
		lin_solver_cached->setParameters(args["solver"]["linear"]);
		logger().info("{}...", lin_solver_cached->name());

		// --------------------------------------------------------------------

		solve_data.rhs_assembler->set_bc(
			local_boundary, boundary_nodes, n_boundary_samples(),
			(formulation() != "Bilaplacian") ? local_neumann_boundary : std::vector<LocalBoundary>(), rhs);

		StiffnessMatrix A = stiffness;
		Eigen::VectorXd b = rhs;

		// --------------------------------------------------------------------

		solve_linear(lin_solver_cached, A, b, args["output"]["advanced"]["spectrum"], sol, pressure);
	}

	void State::init_linear_solve(Eigen::MatrixXd &sol, const double t)
	{
		assert(assembler.is_linear(formulation()) && !is_contact_enabled()); // linear

		if (assembler.is_mixed(formulation()))
			return;

		const int ndof = n_bases * mesh->dimension();

		solve_data.elastic_form = std::make_shared<ElasticForm>(
			n_bases, bases, geom_bases(),
			assembler, ass_vals_cache,
			formulation(),
			problem->is_time_dependent() ? args["time"]["dt"].get<double>() : 0.0,
			mesh->is_volume());

		solve_data.body_form = std::make_shared<BodyForm>(
			ndof, n_pressure_bases,
			boundary_nodes, local_boundary, local_neumann_boundary, n_boundary_samples(),
			rhs, *solve_data.rhs_assembler,
			assembler.density(),
			/*apply_DBC=*/true, /*is_formulation_mixed=*/false, problem->is_time_dependent());
		solve_data.body_form->update_quantities(t, sol);

		solve_data.inertia_form = nullptr;
		solve_data.damping_form = nullptr;
		if (problem->is_time_dependent())
		{
			solve_data.time_integrator = time_integrator::ImplicitTimeIntegrator::construct_time_integrator(args["time"]["integrator"]);
			solve_data.inertia_form = std::make_shared<InertiaForm>(mass, *solve_data.time_integrator);
		}

		solve_data.contact_form = nullptr;
		solve_data.friction_form = nullptr;

		///////////////////////////////////////////////////////////////////////
		// Initialize time integrator
		if (problem->is_time_dependent() && assembler.is_tensor(formulation()))
		{
			POLYFEM_SCOPED_TIMER("Initialize time integrator");

			Eigen::MatrixXd velocity, acceleration;
			initial_velocity(velocity);
			assert(velocity.size() == sol.size());
			initial_velocity(acceleration);
			assert(acceleration.size() == sol.size());

			if (args["optimization"]["enabled"])
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
			solve_data.time_integrator->init(sol, velocity, acceleration, dt);
		}
		solve_data.update_dt();
	}
	
	void State::solve_transient_linear(const int time_steps, const double t0, const double dt, Eigen::MatrixXd &sol, Eigen::MatrixXd &pressure)
	{
		assert(problem->is_time_dependent());
		assert(assembler.is_linear(formulation()) && !is_contact_enabled());
		assert(solve_data.rhs_assembler != nullptr);

		const bool is_scalar_or_mixed = problem->is_scalar() || assembler.is_mixed(formulation());

		// --------------------------------------------------------------------

		auto solver =
			polysolve::LinearSolver::create(args["solver"]["linear"]["solver"], args["solver"]["linear"]["precond"]);
		solver->setParameters(args["solver"]["linear"]);
		logger().info("{}...", solver->name());

		// --------------------------------------------------------------------

		std::shared_ptr<ImplicitTimeIntegrator> time_integrator;
		if (is_scalar_or_mixed)
		{
			time_integrator = std::make_shared<BDF>();
			time_integrator->set_parameters(args["time"]);
			time_integrator->init(sol, Eigen::VectorXd::Zero(sol.size()), Eigen::VectorXd::Zero(sol.size()), dt);
		}
		else
		{
			Eigen::MatrixXd velocity, acceleration;
			initial_velocity(velocity);
			initial_acceleration(acceleration);

			time_integrator = ImplicitTimeIntegrator::construct_time_integrator(args["time"]["integrator"]);
			time_integrator->init(sol, velocity, acceleration, dt);
		}

		// --------------------------------------------------------------------

		const int n_b_samples = n_boundary_samples();

		if (args["optimization"]["enabled"])
		{
			log_and_throw_error("Transient linear problems are not differentiable yet!");
			cache_transient_adjoint_quantities(0, sol, Eigen::MatrixXd::Zero(mesh->dimension(), mesh->dimension()));
		}

		Eigen::MatrixXd current_rhs = rhs;

		// --------------------------------------------------------------------

		for (int t = 1; t <= time_steps; ++t)
		{
			const double time = t0 + t * dt;

			StiffnessMatrix A;
			Eigen::VectorXd b;
			bool compute_spectrum = args["output"]["advanced"]["spectrum"];

			if (is_scalar_or_mixed)
			{
				solve_data.rhs_assembler->compute_energy_grad(
					local_boundary, boundary_nodes, assembler.density(), n_b_samples, local_neumann_boundary, rhs, time,
					current_rhs);

				solve_data.rhs_assembler->set_bc(
					local_boundary, boundary_nodes, n_b_samples, local_neumann_boundary, current_rhs, sol, time);

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

				std::shared_ptr<BDF> bdf = std::dynamic_pointer_cast<BDF>(time_integrator);
				A = mass / bdf->beta_dt() + stiffness;
				b = (mass * bdf->weighted_sum_x_prevs()) / bdf->beta_dt();
				for (int i : boundary_nodes)
					b[i] = 0;
				b += current_rhs;

				compute_spectrum &= t == time_steps;
			}
			else
			{
				solve_data.rhs_assembler->assemble(assembler.density(), current_rhs, time);

				current_rhs *= -1;

				solve_data.rhs_assembler->set_bc(
					std::vector<LocalBoundary>(), std::vector<int>(), n_b_samples, local_neumann_boundary, current_rhs, sol, time);

				current_rhs *= time_integrator->acceleration_scaling();
				current_rhs += mass * time_integrator->x_tilde();

				solve_data.rhs_assembler->set_bc(
					local_boundary, boundary_nodes, n_b_samples, std::vector<LocalBoundary>(), current_rhs, sol, time);

				A = stiffness * time_integrator->acceleration_scaling() + mass;
				b = current_rhs;

				compute_spectrum &= t == 1;
			}

			solve_linear(solver, A, b, compute_spectrum, sol, pressure);

			if (args["optimization"]["enabled"])
			{
				log_and_throw_error("Transient linear problems are not differentiable yet!");
				cache_transient_adjoint_quantities(t, sol, Eigen::MatrixXd::Zero(mesh->dimension(), mesh->dimension()));
			}

			time_integrator->update_quantities(sol);

			save_timestep(time, t, t0, dt, sol, pressure);
			logger().info("{}/{}  t={}", t, time_steps, time);
		}

		time_integrator->save_raw(
			resolve_output_path(args["output"]["data"]["u_path"]),
			resolve_output_path(args["output"]["data"]["v_path"]),
			resolve_output_path(args["output"]["data"]["a_path"]));
	}
} // namespace polyfem
