#include "ALSolver.hpp"

#include <polyfem/utils/Logger.hpp>

namespace polyfem::solver
{

	ALSolver::ALSolver(
		const std::vector<std::shared_ptr<AugmentedLagrangianForm>> &alagr_form,
		double initial_al_weight,
		const double scaling,
		const double max_al_weight,
		const double eta_tol,
		const std::function<void(const Eigen::VectorXd &)> &update_barrier_stiffness,
		const std::function<bool(const Eigen::VectorXd &, bool)> &update_al_weight)
		: alagr_forms{alagr_form},
		  initial_al_weight(initial_al_weight),
		  scaling(scaling),
		  max_al_weight(max_al_weight),
		  eta_tol(eta_tol),
		  update_barrier_stiffness(update_barrier_stiffness),
	      update_al_weight(update_al_weight)
	{
	}

	void ALSolver::solve_al(NLProblem &nl_problem, Eigen::MatrixXd &sol,
							const json &nl_solver_params,
							const json &linear_solver,
							const double characteristic_length)
	{
		assert(sol.size() == nl_problem.full_size());

		Eigen::VectorXd tmp_sol = nl_problem.full_to_reduced(sol);
		assert(tmp_sol.size() == nl_problem.reduced_size());

		// --------------------------------------------------------------------
		double al_weight;
		double last_al_weight = 1e-15;
		bool dummy = true; // dummy variable to pass to update_al_weight is overridden by the lambda in statesolvenonlinear
							//todo: make sure that this is what is expected for L2 call to ALSolver
		bool is_adaptive = 	update_al_weight(sol, true);
		for (auto &f : alagr_forms)
		{
			al_weight = f->lagrangian_weight();
			last_al_weight = f->get_last_al_weight();
		}
		if (last_al_weight > al_weight)
			al_weight = last_al_weight;
		for (auto &f : alagr_forms)
			f->set_initial_weight(al_weight);


		int al_steps = 0;

		double initial_error = 0;
		for (const auto &f : alagr_forms)
			initial_error += f->compute_error(sol);

		//const int n_il_steps = std::max(1, int(initial_error / 1e-2));
		const int n_il_steps = 1;
		for (int t = 1; t <= n_il_steps; ++t)
		{
			const double il_factor = t / double(n_il_steps);
			const Eigen::VectorXd initial_sol = sol;

			for (auto &f : alagr_forms)
				f->set_incr_load(il_factor);

			nl_problem.update_constraint_values();

			nl_problem.use_reduced_size();

			nl_problem.line_search_begin(sol, tmp_sol);

			logger().info("AL IL {}/{} (factor={}) with weight {}", t, n_il_steps, il_factor, al_weight);

			double current_error = 0;
			for (const auto &f : alagr_forms)
				current_error += f->compute_error(sol);

			logger().debug("Initial error = {}", current_error);
			bool first = true;
			update_barrier_stiffness(sol);
			while (first
				   || current_error > 1e-2
				   || !std::isfinite(nl_problem.value(tmp_sol))
				   || !nl_problem.is_step_valid(sol, tmp_sol)
				   || !nl_problem.is_step_collision_free(sol, tmp_sol))
			{
				first = false;
				nl_problem.line_search_end();

				nl_problem.use_full_size();
				//logger().debug("Solving AL Problem with weight {}", al_weight);

				nl_problem.init(sol);

				tmp_sol = sol;

				bool increase_al_weight = true;
				double prev_delta_x_norm = 1e10;

				try
				{
					auto iteration_cb = [&](const polysolve::nonlinear::Criteria &crit) -> bool
					{
						if (std::abs(crit.xDelta) < prev_delta_x_norm)
						{
							prev_delta_x_norm = std::abs(crit.xDelta);
						}

						else if(crit.iterations > 3 &&
							 prev_delta_x_norm*2 < std::abs(crit.xDelta)
							 )
						{
							logger().warn("Current xDelta criteria {}", std::abs(crit.xDelta));
							logger().warn("Caught jump in xDelta norm, trying again");
							increase_al_weight = false;
							update_al_weight(tmp_sol, dummy);
							update_barrier_stiffness(tmp_sol);
							return true;
						}
						if (crit.iterations > 3 &&
													(std::abs(crit.gradNorm) < 1e-3 ||
													 std::abs(crit.xDeltaDotGrad) < 1e-10))
						{
							logger().warn("Converged after {} iterations", crit.iterations);
							increase_al_weight = false;
							return true;
						}

						if (crit.alpha < 1e-3 && crit.iterations > 3 &&
							 prev_delta_x_norm*10 < std::abs(crit.xDelta))
						{
							al_weight /= scaling;
							increase_al_weight = false;
							logger().debug("Decreasing AL weight to {}", al_weight);
							for (const auto &f : alagr_forms)
							{
								f->set_al_weight(al_weight);
							}
							update_barrier_stiffness(tmp_sol);
							return true;
						}


						return false;
					};

					const auto scale = nl_problem.normalize_forms();
					auto nl_solver = polysolve::nonlinear::Solver::create(
						nl_solver_params, linear_solver, characteristic_length / scale, logger());
					nl_solver->set_iteration_callback(iteration_cb);
					nl_solver->minimize(nl_problem, tmp_sol);
					nl_problem.finish();
				}
				catch (const std::runtime_error &e)
				{
					std::string err_msg = e.what();
					logger().debug("Failed to solve; {}", err_msg);
					// if the nonlinear solve fails due to invalid energy at the current solution, changing the weights would not help
					// if (err_msg.find("f(x) is nan or inf; stopping") != std::string::npos)
					// log_and_throw_error("Failed to solve with AL; f(x) is nan or inf");
				}

				sol = tmp_sol;

				const auto prev_error = current_error;

				current_error = 0;
				for (const auto &f : alagr_forms)
					current_error += f->compute_error(sol);
				// logger().debug("Current error = {}", current_error);

				nl_problem.use_reduced_size();
				tmp_sol = nl_problem.full_to_reduced(sol);
				nl_problem.line_search_begin(sol, tmp_sol);

				logger().debug("Current error = {}, prev error = {}", current_error, prev_error);

				const double ratio_error = std::log10(std::max(std::abs(prev_error), std::abs(current_error)) / std::min(std::abs(prev_error), std::abs(current_error)));

				const double tolerance = 1 - eta_tol;
				const double ratio_tolerance = std::log10(std::max(std::abs(1.0), std::abs(tolerance)) /
									   std::min(std::abs(1.0), std::abs(tolerance)));

				bool no_movement = false;
				double dbc_movement = 0;
				for (const auto &f : alagr_forms)
					dbc_movement = f->get_dbcdist();
				if (current_error == initial_error && dbc_movement == 0)
					no_movement = true;

				if ( (is_adaptive && increase_al_weight && prev_error!= 0 && ratio_error<ratio_tolerance && al_weight < max_al_weight && !no_movement) || (is_adaptive && increase_al_weight && prev_error<current_error && al_weight < max_al_weight && !no_movement))
				{
					al_weight *= scaling;


					logger().debug("Increasing weight to {}", al_weight);
					for (const auto &f : alagr_forms)
					{
						f->set_al_weight(al_weight);
						f->set_last_al_weight(al_weight);
					}

				}
				else
				{
					for (auto &f : alagr_forms)
						f->update_lagrangian(sol, al_weight);
				}

				++al_steps;
			}
			nl_problem.line_search_end();
			post_subsolve(al_weight);
		}
	}

	void ALSolver::solve_reduced(NLProblem &nl_problem, Eigen::MatrixXd &sol,
								 const json &nl_solver_params,
								 const json &linear_solver,
								 const double characteristic_length)
	{
		assert(sol.size() == nl_problem.full_size());

		Eigen::VectorXd tmp_sol = nl_problem.full_to_reduced(sol);
		nl_problem.use_reduced_size();
		nl_problem.line_search_begin(sol, tmp_sol);

		if (!std::isfinite(nl_problem.value(tmp_sol))
			|| !nl_problem.is_step_valid(sol, tmp_sol)
			|| !nl_problem.is_step_collision_free(sol, tmp_sol))
			log_and_throw_error("Failed to apply constraints conditions; solve with augmented lagrangian first!");
		nl_problem.line_search_end();
		// --------------------------------------------------------------------
		// Perform one final solve with the DBC projected out

		logger().debug("Successfully applied constraints conditions; solving in reduced space");

		nl_problem.init(sol);

		try
		{
			const auto scale = nl_problem.normalize_forms();
			auto nl_solver = polysolve::nonlinear::Solver::create(
				nl_solver_params, linear_solver, characteristic_length / scale, logger());
			nl_solver->minimize(nl_problem, tmp_sol);
			nl_problem.finish();
		}
		catch (const std::runtime_error &e)
		{
			sol = nl_problem.reduced_to_full(tmp_sol);
			throw e;
		}
		sol = nl_problem.reduced_to_full(tmp_sol);

		post_subsolve(0);
	}


} // namespace polyfem::solver