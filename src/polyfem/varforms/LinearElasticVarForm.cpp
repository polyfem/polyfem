#include "LinearElasticVarForm.hpp"

#include <polyfem/solver/forms/BodyForm.hpp>
#include <polyfem/solver/forms/ElasticForm.hpp>
#include <polyfem/solver/forms/InertiaForm.hpp>

#include <polyfem/time_integrator/ImplicitTimeIntegrator.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/Timer.hpp>

#include <unsupported/Eigen/SparseExtra>

#include <polysolve/linear/FEMSolver.hpp>

namespace polyfem::varform
{
	namespace
	{
		bool write_matrix_market(const json &args, const StiffnessMatrix &stiffness)
		{
			const std::string full_mat_path = args["output"]["data"]["full_mat"];
			if (full_mat_path.empty())
				return false;

			Eigen::saveMarket(stiffness, full_mat_path);
			return true;
		}
	} // namespace

	std::vector<io::OutputField> LinearElasticVarForm::output_fields(
		const io::OutputSample &sample,
		const Eigen::MatrixXd &solution,
		const io::OutputFieldOptions &options) const
	{
		const std::vector<std::pair<std::string, std::shared_ptr<solver::Form>>> named_forms{
			{"elastic", elastic_form},
			{"inertia", inertia_form},
			{"body", body_form}};
		return elastic_output_fields(
			sample, solution, options, nullptr, time_integrator.get(), named_forms, elastic_form.get());
	}

	void LinearElasticVarForm::build_stiffness_mat(StiffnessMatrix &stiffness)
	{
		igl::Timer timer;
		timer.start();
		logger().info("Assembling stiffness mat...");
		assert(assembler->is_linear());

		assembler->assemble(mesh_->is_volume(), disp_space.n_bases, *disp_space.bases, geom_bases(), displacement_caches.values, 0, stiffness);

		timer.stop();
		timings.assembling_stiffness_mat_time = timer.getElapsedTime();
		logger().info(" took {}s", timings.assembling_stiffness_mat_time);

		stats.nn_zero = stiffness.nonZeros();
		stats.num_dofs = stiffness.rows();
		stats.mat_size = (long long)stiffness.rows() * (long long)stiffness.cols();
		logger().info("sparsity: {}/{}", stats.nn_zero, stats.mat_size);

		write_matrix_market(args, stiffness);
	}

	void LinearElasticVarForm::solve_linear_system(
		const std::unique_ptr<polysolve::linear::Solver> &solver,
		StiffnessMatrix &A,
		Eigen::VectorXd &b,
		const bool compute_spectrum,
		Eigen::MatrixXd &sol)
	{
		assert(assembler->is_linear());
		assert(rhs_assembler != nullptr);

		const int problem_dim = problem->is_scalar() ? 1 : mesh_->dimension();
		const int precond_num = problem_dim * disp_space.n_bases;

		Eigen::VectorXd x;
		stats.spectrum = dirichlet_solve(
			*solver,
			A,
			b,
			boundary.boundary_nodes,
			x,
			precond_num,
			args["output"]["data"]["stiffness_mat"],
			compute_spectrum,
			assembler->is_fluid(),
			/*use_avg_pressure=*/true);

		sol = x;
		solver->get_info(stats.solver_info);

		const auto error = (A * x - b).norm();
		if (error > 1e-4)
			logger().error("Solver error: {}", error);
		else
			logger().debug("Solver error: {}", error);
	}

	void LinearElasticVarForm::init_linear_solve(Eigen::MatrixXd &sol, const double t)
	{
		assert(sol.cols() == 1);
		assert(assembler->is_linear());

		const int ndof = disp_space.n_bases * mesh_->dimension();

		elastic_form = std::make_shared<solver::ElasticForm>(
			disp_space.n_bases, *disp_space.bases, geom_bases(),
			*assembler, displacement_caches.values,
			t, problem->is_time_dependent() ? args["time"]["dt"].get<double>() : 0.0,
			mesh_->is_volume());

		body_form = std::make_shared<solver::BodyForm>(
			ndof, 0,
			boundary.boundary_nodes, boundary.local_boundary, boundary.local_neumann_boundary, n_boundary_samples(),
			rhs, *rhs_assembler,
			mass_matrix_assembler->density(),
			/*is_formulation_mixed=*/false, problem->is_time_dependent());
		body_form->update_quantities(t, sol);

		if (problem->is_time_dependent())
		{
			time_integrator = time_integrator::ImplicitTimeIntegrator::construct_time_integrator(args["time"]["integrator"]);
			inertia_form = std::make_shared<solver::InertiaForm>(mass, *time_integrator);

			POLYFEM_SCOPED_TIMER("Initialize time integrator");

			Eigen::MatrixXd solution, velocity, acceleration;
			initial_solution(solution);
			solution.col(0) = sol;
			assert(solution.rows() == sol.size());
			initial_velocity(velocity);
			assert(velocity.rows() == sol.size());
			initial_acceleration(acceleration);
			assert(acceleration.rows() == sol.size());

			time_integrator->init(solution, velocity, acceleration, dt);

			elastic_form->set_weight(time_integrator->acceleration_scaling());
			body_form->set_weight(time_integrator->acceleration_scaling());
		}
		else
		{
			time_integrator = nullptr;
		}
	}

	void LinearElasticVarForm::solve_static_linear(Eigen::MatrixXd &sol)
	{
		auto solver = polysolve::linear::Solver::create(args["solver"]["linear"], logger());
		logger().info("{}...", solver->name());

		rhs_assembler->set_bc(
			boundary.local_boundary, boundary.boundary_nodes, n_boundary_samples(),
			boundary.local_neumann_boundary, rhs);

		StiffnessMatrix A;
		build_stiffness_mat(A);

		Eigen::VectorXd b = rhs;
		solve_linear_system(solver, A, b, args["output"]["advanced"]["spectrum"], sol);
	}

	void LinearElasticVarForm::solve_transient_linear(Eigen::MatrixXd &sol)
	{
		assert(problem->is_time_dependent());
		assert(rhs_assembler != nullptr);
		assert(time_integrator != nullptr);

		auto solver = polysolve::linear::Solver::create(args["solver"]["linear"], logger());
		logger().info("{}...", solver->name());

		save_timestep(t0, 0, t0, dt, sol);

		Eigen::MatrixXd current_rhs = rhs;

		StiffnessMatrix stiffness;
		build_stiffness_mat(stiffness);

		for (int t = 1; t <= time_steps; ++t)
		{
			const double time = t0 + t * dt;

			rhs_assembler->assemble(mass_matrix_assembler->density(), current_rhs, time);
			current_rhs *= -1;

			rhs_assembler->set_bc(
				std::vector<mesh::LocalBoundary>(), std::vector<int>(), n_boundary_samples(),
				boundary.local_neumann_boundary, current_rhs, sol, time);

			current_rhs *= time_integrator->acceleration_scaling();
			current_rhs += mass * time_integrator->x_tilde();

			rhs_assembler->set_bc(
				boundary.local_boundary, boundary.boundary_nodes, n_boundary_samples(),
				std::vector<mesh::LocalBoundary>(), current_rhs, sol, time);

			StiffnessMatrix A = stiffness * time_integrator->acceleration_scaling() + mass;
			Eigen::VectorXd b = current_rhs;

			solve_linear_system(solver, A, b, args["output"]["advanced"]["spectrum"].get<bool>() && t == 1, sol);

			time_integrator->update_quantities(sol);
			save_timestep(time, t, t0, dt, sol);
			save_elastic_step_state(t0, dt, t, time_integrator.get());

			logger().info("{}/{}  t={}", t, time_steps, time);
			notify_time_step(t);
		}
	}

	void LinearElasticVarForm::solve_problem(Eigen::MatrixXd &sol)
	{
		stats.spectrum.setZero();

		igl::Timer timer;
		timer.start();
		logger().info("Solving {}", assembler->name());

		{
			POLYFEM_SCOPED_TIMER("Setup RHS");

			if (sol.size() <= 0)
				initial_solution(sol);

			if (sol.cols() > 1)
				sol.conservativeResize(Eigen::NoChange, 1);
		}

		init_linear_solve(sol, problem->is_time_dependent() ? t0 + dt : 1.0);

		if (problem->is_time_dependent())
			solve_transient_linear(sol);
		else
			solve_static_linear(sol);

		timer.stop();
		timings.solving_time = timer.getElapsedTime();
		logger().info(" took {}s", timings.solving_time);
	}

} // namespace polyfem::varform
