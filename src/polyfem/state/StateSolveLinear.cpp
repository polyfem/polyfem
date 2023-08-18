#include <polyfem/State.hpp>

#include <polyfem/assembler/Mass.hpp>
#include <polyfem/assembler/AssemblerUtils.hpp>

#include <polyfem/time_integrator/ImplicitTimeIntegrator.hpp>
#include <polyfem/time_integrator/BDF.hpp>

#include <polyfem/solver/forms/BodyForm.hpp>
#include <polyfem/solver/forms/ElasticForm.hpp>
#include <polyfem/solver/forms/InertiaForm.hpp>
#include <polysolve/FEMSolver.hpp>

#include <polyfem/utils/Timer.hpp>

#include <unsupported/Eigen/SparseExtra>

namespace polyfem
{
	using namespace mesh;
	using namespace time_integrator;
	using namespace utils;
	using namespace solver;
	using namespace io;

	void State::build_stiffness_mat(StiffnessMatrix &stiffness)
	{
		igl::Timer timer;
		timer.start();
		logger().info("Assembling stiffness mat...");
		assert(assembler->is_linear());

		if (mixed_assembler != nullptr)
		{

			StiffnessMatrix velocity_stiffness, mixed_stiffness, pressure_stiffness;
			assembler->assemble(mesh->is_volume(), n_bases, bases, geom_bases(), ass_vals_cache, velocity_stiffness);
			mixed_assembler->assemble(mesh->is_volume(), n_pressure_bases, n_bases, pressure_bases, bases, geom_bases(), pressure_ass_vals_cache, ass_vals_cache, mixed_stiffness);
			pressure_assembler->assemble(mesh->is_volume(), n_pressure_bases, pressure_bases, geom_bases(), pressure_ass_vals_cache, pressure_stiffness);

			const int problem_dim = problem->is_scalar() ? 1 : mesh->dimension();

			assembler::AssemblerUtils::merge_mixed_matrices(n_bases, n_pressure_bases, problem_dim, use_avg_pressure ? assembler->is_fluid() : false,
															velocity_stiffness, mixed_stiffness, pressure_stiffness,
															stiffness);
		}
		else
		{
			assembler->assemble(mesh->is_volume(), n_bases, bases, geom_bases(), ass_vals_cache, stiffness);
		}

		timer.stop();
		timings.assembling_stiffness_mat_time = timer.getElapsedTime();
		logger().info(" took {}s", timings.assembling_stiffness_mat_time);

		stats.nn_zero = stiffness.nonZeros();
		stats.num_dofs = stiffness.rows();
		stats.mat_size = (long long)stiffness.rows() * (long long)stiffness.cols();
		logger().info("sparsity: {}/{}", stats.nn_zero, stats.mat_size);

		const std::string full_mat_path = args["output"]["data"]["full_mat"];
		if (!full_mat_path.empty())
		{
			Eigen::saveMarket(stiffness, full_mat_path);
		}
	}

	void State::solve_linear(
		const std::unique_ptr<polysolve::LinearSolver> &solver,
		StiffnessMatrix &A,
		Eigen::VectorXd &b,
		const bool compute_spectrum,
		Eigen::MatrixXd &sol, Eigen::MatrixXd &pressure)
	{
		assert(assembler->is_linear() && !is_contact_enabled());
		assert(solve_data.rhs_assembler != nullptr);

		const int problem_dim = problem->is_scalar() ? 1 : mesh->dimension();
		const int precond_num = problem_dim * n_bases;

		Eigen::VectorXd x;
		if (optimization_enabled)
		{
			auto A_tmp = A;
			prefactorize(*solver, A, boundary_nodes, precond_num, args["output"]["data"]["stiffness_mat"]);
			dirichlet_solve_prefactorized(*solver, A_tmp, b, boundary_nodes, x);
		}
		else
		{
			stats.spectrum = dirichlet_solve(
				*solver, A, b, boundary_nodes, x, precond_num, args["output"]["data"]["stiffness_mat"], compute_spectrum,
				assembler->is_fluid(), use_avg_pressure);
		}
		sol = x; // Explicit copy because sol is a MatrixXd (with one column)

		solver->getInfo(stats.solver_info);

		const auto error = (A * x - b).norm();
		if (error > 1e-4)
			logger().error("Solver error: {}", error);
		else
			logger().debug("Solver error: {}", error);

		if (mixed_assembler != nullptr)
			sol_to_pressure(sol, pressure);
	}

	void State::solve_linear(Eigen::MatrixXd &sol, Eigen::MatrixXd &pressure)
	{
		assert(!problem->is_time_dependent());
		assert(assembler->is_linear() && !is_contact_enabled());

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
			(assembler->name() != "Bilaplacian") ? local_neumann_boundary : std::vector<LocalBoundary>(), rhs);

		StiffnessMatrix A;
		build_stiffness_mat(A);

		Eigen::VectorXd b = rhs;

		// --------------------------------------------------------------------

		solve_linear(lin_solver_cached, A, b, args["output"]["advanced"]["spectrum"], sol, pressure);
	}

	void State::init_linear_solve(Eigen::MatrixXd &sol, const double t)
	{
		assert(assembler->is_linear() && !is_contact_enabled()); // linear

		if (mixed_assembler != nullptr)
			return;

		const int ndof = n_bases * mesh->dimension();

		solve_data.elastic_form = std::make_shared<ElasticForm>(
			n_bases, bases, geom_bases(),
			*assembler, ass_vals_cache,
			problem->is_time_dependent() ? args["time"]["dt"].get<double>() : 0.0,
			mesh->is_volume());

		solve_data.body_form = std::make_shared<BodyForm>(
			ndof, n_pressure_bases,
			boundary_nodes, local_boundary, local_neumann_boundary, n_boundary_samples(),
			rhs, *solve_data.rhs_assembler,
			mass_matrix_assembler->density(),
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
		if (problem->is_time_dependent() && assembler->is_tensor())
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
	}

	void State::solve_transient_linear(const int time_steps, const double t0, const double dt, Eigen::MatrixXd &sol, Eigen::MatrixXd &pressure)
	{
		assert(problem->is_time_dependent());
		assert(assembler->is_linear() && !is_contact_enabled());
		assert(solve_data.rhs_assembler != nullptr);

		const bool is_scalar_or_mixed = problem->is_scalar() || mixed_assembler != nullptr;

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

		if (optimization_enabled)
		{
			log_and_throw_error("Transient linear problems are not differentiable yet!");
			cache_transient_adjoint_quantities(0, sol, Eigen::MatrixXd::Zero(mesh->dimension(), mesh->dimension()));
		}

		Eigen::MatrixXd current_rhs = rhs;

		StiffnessMatrix stiffness;
		build_stiffness_mat(stiffness);

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
					local_boundary, boundary_nodes, mass_matrix_assembler->density(), n_b_samples, local_neumann_boundary, rhs, time,
					current_rhs);

				solve_data.rhs_assembler->set_bc(
					local_boundary, boundary_nodes, n_b_samples, local_neumann_boundary, current_rhs, sol, time);

				if (mixed_assembler != nullptr)
				{
					// divergence free
					int fluid_offset = use_avg_pressure ? (assembler->is_fluid() ? 1 : 0) : 0;
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
				solve_data.rhs_assembler->assemble(mass_matrix_assembler->density(), current_rhs, time);

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

			if (optimization_enabled)
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
