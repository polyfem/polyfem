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

	void LinearElasticVarForm::reset()
	{
		ElasticVarForm::reset();
		solve_data_.elastic_form = nullptr;
		solve_data_.body_form = nullptr;
		solve_data_.inertia_form = nullptr;
		solve_data_.time_integrator = nullptr;
	}

	std::vector<io::OutputField> LinearElasticVarForm::output_fields(
		const io::OutputSample &sample,
		const Eigen::MatrixXd &solution,
		const io::OutputFieldOptions &options) const
	{
		const std::vector<std::pair<std::string, std::shared_ptr<solver::Form>>> named_forms{
			{"elastic", solve_data_.elastic_form},
			{"inertia", solve_data_.inertia_form},
			{"body", solve_data_.body_form}};
		return elastic_output_fields(
			sample, solution, options, nullptr, solve_data_.time_integrator.get(), named_forms, solve_data_.elastic_form.get());
	}

	void LinearElasticVarForm::build_stiffness_mat(StiffnessMatrix &stiffness)
	{
		igl::Timer timer;
		timer.start();
		logger().info("Assembling stiffness mat...");
		assert(primary_assembler_->is_linear());

		primary_assembler_->assemble(mesh_->is_volume(), space_.n_bases, space_.basis_list(), space_.geometry_basis_list(), ass_vals_cache_, 0, stiffness);

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
		assert(primary_assembler_->is_linear());
		assert(rhs_assembler_ != nullptr);

		const int problem_dim = problem->is_scalar() ? 1 : mesh_->dimension();
		const int precond_num = problem_dim * space_.n_bases;

		Eigen::VectorXd x;
		stats.spectrum = dirichlet_solve(
			*solver,
			A,
			b,
			boundary_.boundary_nodes,
			x,
			precond_num,
			args["output"]["data"]["stiffness_mat"],
			compute_spectrum,
			primary_assembler_->is_fluid(),
			/*use_avg_pressure=*/true);

		sol = x;
		solver->get_info(stats.solver_info);

		const auto error = (A * x - b).norm();
		if (error > 1e-4)
			logger().error("Solver error: {}", error);
		else
			logger().debug("Solver error: {}", error);
	}

	void LinearElasticVarForm::init_linear_solve(Eigen::MatrixXd &sol, const double t, const InitialConditionOverride *initial_condition_override)
	{
		assert(sol.cols() == 1);
		assert(primary_assembler_->is_linear());

		const int ndof = space_.n_bases * mesh_->dimension();

		solve_data_.elastic_form = std::make_shared<solver::ElasticForm>(
			space_.n_bases, *space_.bases, space_.geometry_basis_list(),
			*primary_assembler_, ass_vals_cache_,
			t, problem->is_time_dependent() ? args["time"]["dt"].get<double>() : 0.0,
			mesh_->is_volume(),
			args["solver"]["advanced"]["jacobian_threshold"],
			args["solver"]["advanced"]["check_inversion"],
			args["solver"]["advanced"]["conservative_max_iter"]);

		solve_data_.body_form = std::make_shared<solver::BodyForm>(
			ndof, 0,
			boundary_.boundary_nodes, boundary_.local_boundary, boundary_.local_neumann_boundary, elastic_boundary_samples(),
			rhs_, *rhs_assembler_,
			mass_assembler_->density(),
			/*is_formulation_mixed=*/false, problem->is_time_dependent());
		solve_data_.body_form->update_quantities(t, sol);

		if (problem->is_time_dependent())
		{
			solve_data_.time_integrator = time_integrator::ImplicitTimeIntegrator::construct_time_integrator(args["time"]["integrator"]);
			solve_data_.inertia_form = std::make_shared<solver::InertiaForm>(mass_, *solve_data_.time_integrator);

			POLYFEM_SCOPED_TIMER("Initialize time integrator");

			Eigen::MatrixXd solution, velocity, acceleration;
			initial_solution(solution, initial_condition_override);
			solution.col(0) = sol;
			assert(solution.rows() == sol.size());
			initial_velocity(velocity, initial_condition_override);
			assert(velocity.rows() == sol.size());
			initial_acceleration(acceleration, initial_condition_override);
			assert(acceleration.rows() == sol.size());
			if (solution.cols() != velocity.cols() || solution.cols() != acceleration.cols())
			{
				log_and_throw_error(
					"Incompatible initial-condition history for transient solve: "
					"solution has {} columns, velocity has {}, acceleration has {}.",
					solution.cols(), velocity.cols(), acceleration.cols());
			}

			solve_data_.time_integrator->init(solution, velocity, acceleration, dt);

			solve_data_.elastic_form->set_weight(solve_data_.time_integrator->acceleration_scaling());
			solve_data_.body_form->set_weight(solve_data_.time_integrator->acceleration_scaling());
		}
		else
		{
			solve_data_.time_integrator = nullptr;
		}
	}

	void LinearElasticVarForm::solve_static_linear(Eigen::MatrixXd &sol, const ForwardStepCallback &post_step)
	{
		auto solver = polysolve::linear::Solver::create(args["solver"]["linear"], logger());
		logger().info("{}...", solver->name());

		rhs_assembler_->set_bc(
			boundary_.local_boundary, boundary_.boundary_nodes, elastic_boundary_samples(),
			boundary_.local_neumann_boundary, rhs_);

		StiffnessMatrix A;
		build_stiffness_mat(A);

		Eigen::VectorXd b = rhs_;
		solve_linear_system(solver, A, b, args["output"]["advanced"]["spectrum"], sol);
		if (post_step)
			post_step(0, sol);
	}

	void LinearElasticVarForm::solve_transient_linear(Eigen::MatrixXd &sol, const ForwardStepCallback &post_step)
	{
		assert(problem->is_time_dependent());
		assert(rhs_assembler_ != nullptr);
		assert(solve_data_.time_integrator != nullptr && "Transient linear elasticity requires an initialized time integrator");

		auto solver = polysolve::linear::Solver::create(args["solver"]["linear"], logger());
		logger().info("{}...", solver->name());

		save_timestep(t0, 0, t0, dt, sol);
		if (post_step)
			post_step(0, sol);

		Eigen::MatrixXd current_rhs = rhs_;

		StiffnessMatrix stiffness;
		build_stiffness_mat(stiffness);

		for (int t = 1; t <= time_steps; ++t)
		{
			const double time = t0 + t * dt;

			rhs_assembler_->assemble(mass_assembler_->density(), current_rhs, time);
			current_rhs *= -1;

			rhs_assembler_->set_bc(
				std::vector<mesh::LocalBoundary>(), std::vector<int>(), elastic_boundary_samples(),
				boundary_.local_neumann_boundary, current_rhs, sol, time);

			current_rhs *= solve_data_.time_integrator->acceleration_scaling();
			current_rhs += mass_ * solve_data_.time_integrator->x_tilde();

			rhs_assembler_->set_bc(
				boundary_.local_boundary, boundary_.boundary_nodes, elastic_boundary_samples(),
				std::vector<mesh::LocalBoundary>(), current_rhs, sol, time);

			StiffnessMatrix A = stiffness * solve_data_.time_integrator->acceleration_scaling() + mass_;
			Eigen::VectorXd b = current_rhs;

			solve_linear_system(solver, A, b, args["output"]["advanced"]["spectrum"].get<bool>() && t == 1, sol);
			if (post_step)
				post_step(t, sol);

			solve_data_.time_integrator->update_quantities(sol);
			save_timestep(time, t, t0, dt, sol);
			save_elastic_step_state(t0, dt, t, solve_data_.time_integrator.get());

			logger().info("{}/{}  t={}", t, time_steps, time);
			notify_time_step(t, time_steps, t0, dt);
		}
	}

	void LinearElasticVarForm::solve_problem(
		Eigen::MatrixXd &sol,
		const InitialConditionOverride *initial_condition_override,
		const ForwardStepCallback &post_step)
	{
		assert(
			(problem->is_time_dependent() || !initial_condition_override
			 || (initial_condition_override->velocity.size() == 0
				 && initial_condition_override->acceleration.size() == 0))
			&& "Static elasticity does not accept initial velocity or acceleration overrides");

		stats.spectrum.setZero();

		igl::Timer timer;
		timer.start();
		logger().info("Solving {}", primary_assembler_->name());

		{
			POLYFEM_SCOPED_TIMER("Setup RHS");

			if (initial_condition_override && initial_condition_override->solution.size() != 0)
				initial_solution(sol, initial_condition_override);
			else if (sol.size() <= 0)
				initial_solution(sol, initial_condition_override);

			if (!problem->is_time_dependent())
			{
				if (initial_condition_override && initial_condition_override->solution.size() != 0)
					assert(sol.cols() == 1 && "Static initial solution override must have exactly one column");
				else if (sol.cols() != 1)
					log_and_throw_error("Static elasticity requires exactly one initial solution column.");
			}
			if (sol.cols() > 1)
				sol.conservativeResize(Eigen::NoChange, 1);
		}

		init_linear_solve(sol, problem->is_time_dependent() ? t0 + dt : 1.0, initial_condition_override);

		if (problem->is_time_dependent())
			solve_transient_linear(sol, post_step);
		else
			solve_static_linear(sol, post_step);

		timer.stop();
		timings.solving_time = timer.getElapsedTime();
		logger().info(" took {}s", timings.solving_time);
	}

} // namespace polyfem::varform
