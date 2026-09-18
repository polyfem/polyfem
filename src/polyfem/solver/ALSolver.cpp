#include "ALSolver.hpp"

#include <polyfem/utils/Logger.hpp>

namespace polyfem::solver
{
	ALSolver::ALSolver(
		const std::vector<std::shared_ptr<AugmentedLagrangianForm>> &alagr_form,
		const double initial_al_weight,
		const double scaling,
		const double max_al_weight,
		const double eta_tol,
		const std::function<void(const Eigen::VectorXd &)> &update_barrier_stiffness)
		: alagr_forms{alagr_form},
		  initial_al_weight(initial_al_weight),
		  scaling(scaling),
		  max_al_weight(max_al_weight),
		  eta_tol(eta_tol),
		  update_barrier_stiffness(update_barrier_stiffness)
	{
	}

	void ALSolver::solve_al(NLProblem &nl_problem, Eigen::MatrixXd &sol,
							const json &nl_solver_params,
							const json &linear_solver,
							const double characteristic_length,
							std::shared_ptr<polysolve::nonlinear::Solver> nl_solverin)
	{
		assert(sol.size() == nl_problem.full_size());

		const Eigen::VectorXd initial_sol = sol;
		Eigen::VectorXd tmp_sol = nl_problem.full_to_reduced(sol);
		assert(tmp_sol.size() == nl_problem.reduced_size());

		// --------------------------------------------------------------------

		double al_weight = initial_al_weight;
		int al_steps = 0;

		double initial_error = 0;
		for (const auto &f : alagr_forms)
			initial_error += f->compute_error(sol);

		nl_problem.use_reduced_size();
		nl_problem.line_search_begin(sol, tmp_sol);

		for (auto &f : alagr_forms)
			f->set_initial_weight(al_weight);

		double current_error = 0;
		for (const auto &f : alagr_forms)
			current_error += f->compute_error(sol);

		logger().debug("Initial error = {}", current_error);

		while (!std::isfinite(nl_problem.value(tmp_sol))
			   || !nl_problem.is_step_valid(sol, tmp_sol)
			   || !nl_problem.is_step_collision_free(sol, tmp_sol))
		{
			nl_problem.line_search_end();

			nl_problem.use_full_size();
			logger().debug("Solving AL Problem with weight {}", al_weight);

			nl_problem.init(sol);
			update_barrier_stiffness(sol);
			tmp_sol = sol;

			std::shared_ptr<polysolve::nonlinear::Solver> nl_solver;
			try
			{
				const auto scale = nl_problem.normalize_forms();
				nl_solver = nl_solverin == nullptr ? polysolve::nonlinear::Solver::create(
														 nl_solver_params, linear_solver, characteristic_length * scale, logger())
												   : nl_solverin;
				nl_solver->minimize(nl_problem, tmp_sol);
				nl_problem.finish();
			}
			catch (const std::runtime_error &e)
			{
				std::string err_msg = e.what();
				// if the nonlinear solve fails due to invalid energy at the current solution, changing the weights would not help
				if (err_msg.find("f(x) is nan or inf; stopping") != std::string::npos)
					log_and_throw_error("Failed to solve with AL; f(x) is nan or inf");
				if (err_msg.find("Reached iteration limit") != std::string::npos)
					log_and_throw_error("Reached iteration limit in AL");
			}
			record_solver_info(nl_solver);

			sol = tmp_sol;

			current_error = 0;
			for (const auto &f : alagr_forms)
				current_error += f->compute_error(sol);
			logger().debug("Current error = {}", current_error);
			const double eta = 1 - sqrt(current_error / initial_error);

			logger().debug("Current eta = {}", eta);

			if (eta < 0)
			{
				logger().debug("Higher error than initial, increase weight and revert to previous solution");
				sol = initial_sol;
			}

			nl_problem.use_reduced_size();
			tmp_sol = nl_problem.full_to_reduced(sol);
			nl_problem.line_search_begin(sol, tmp_sol);

			if (eta < eta_tol && al_weight < max_al_weight)
				al_weight *= scaling;

			for (auto &f : alagr_forms)
				f->update_lagrangian(sol, al_weight);

			post_subsolve(al_weight);
			++al_steps;
		}
		nl_problem.line_search_end();
	}

	void ALSolver::solve_reduced(NLProblem &nl_problem, Eigen::MatrixXd &sol,
								 const json &nl_solver_params,
								 const json &linear_solver,
								 const double characteristic_length,
								 std::shared_ptr<polysolve::nonlinear::Solver> nl_solverin)
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
		update_barrier_stiffness(sol);
		std::shared_ptr<polysolve::nonlinear::Solver> nl_solver;
		try
		{
			const auto scale = nl_problem.normalize_forms();
			nl_solver = nl_solverin == nullptr ? polysolve::nonlinear::Solver::create(
													 nl_solver_params, linear_solver, characteristic_length * scale, logger())
											   : nl_solverin;
			nl_solver->minimize(nl_problem, tmp_sol);
			nl_problem.finish();
		}
		catch (const std::runtime_error &e)
		{
			sol = nl_problem.reduced_to_full(tmp_sol);
			throw e;
		}
		sol = nl_problem.reduced_to_full(tmp_sol);

		record_solver_info(nl_solver);
		post_subsolve(0);
	}

	void ALSolver::record_solver_info(const std::shared_ptr<polysolve::nonlinear::Solver> &nl_solver)
	{
		// Null only when normalizing the forms or creating the solver threw, i.e. no solver ran.
		if (nl_solver == nullptr)
		{
			nl_solver_info = json();
			return;
		}
		nl_solver_info = nl_solver->info();
		// minimize() refreshes info()["status"] only when it returns; when it throws (line search
		// failed, ...) and solve_al carries on, info() still holds a stale "NotStarted" or "Continue",
		// while status() holds the reason it stopped.
		nl_solver_info["status"] = nl_solver->status();
	}

} // namespace polyfem::solver