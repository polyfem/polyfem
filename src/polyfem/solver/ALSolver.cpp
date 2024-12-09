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

	void ALSolver::solve_al(std::shared_ptr<NLSolver> nl_solver, NLProblem &nl_problem, Eigen::MatrixXd &sol)
	{
		assert(sol.size() == nl_problem.full_size());

		const Eigen::VectorXd initial_sol = sol;
		Eigen::VectorXd tmp_sol = nl_problem.full_to_reduced(sol);
		assert(tmp_sol.size() == nl_problem.reduced_size());

		// --------------------------------------------------------------------

		double al_weight = initial_al_weight;
		int al_steps = 0;
		const int iters = nl_solver->stop_criteria().iterations;

		double initial_error = 0;
		for (const auto &f : alagr_forms)
			initial_error += f->compute_error(sol);

		nl_problem.line_search_begin(sol, tmp_sol);

		for (auto &f : alagr_forms)
			f->set_initial_weight(al_weight);

		while (!std::isfinite(nl_problem.value(tmp_sol))
			   || !nl_problem.is_step_valid(sol, tmp_sol)
			   || !nl_problem.is_step_collision_free(sol, tmp_sol))
		{
			nl_problem.line_search_end();

			set_al_weight(nl_problem, sol, al_weight);
			logger().debug("Solving AL Problem with weight {}", al_weight);

			nl_problem.init(sol);
			update_barrier_stiffness(sol);
			tmp_sol = sol;

			try
			{
				nl_solver->minimize(nl_problem, tmp_sol);
				nl_problem.finish();
			}
			catch (const std::runtime_error &e)
			{
				std::string err_msg = e.what();
				// if the nonlinear solve fails due to invalid energy at the current solution, changing the weights would not help
				if (err_msg.find("f(x) is nan or inf; stopping") != std::string::npos)
					log_and_throw_error("Failed to solve with AL; f(x) is nan or inf");
			}

			sol = tmp_sol;
			set_al_weight(nl_problem, sol, -1);

			double current_error = 0;
			for (const auto &f : alagr_forms)
				f->compute_error(sol);
			const double eta = 1 - sqrt(current_error / initial_error);

			logger().debug("Current eta = {}", eta);

			if (eta < 0)
			{
				logger().debug("Higher error than initial, increase weight and revert to previous solution");
				sol = initial_sol;
			}

			tmp_sol = nl_problem.full_to_reduced(sol);
			nl_problem.line_search_begin(sol, tmp_sol);

			if (eta < eta_tol && al_weight < max_al_weight)
				al_weight *= scaling;
			else
			{
				for (auto &f : alagr_forms)
					f->update_lagrangian(sol, al_weight);
			}

			post_subsolve(al_weight);
			++al_steps;
		}
		nl_problem.line_search_end();
		nl_solver->stop_criteria().iterations = iters;
	}

	void ALSolver::solve_reduced(std::shared_ptr<NLSolver> nl_solver, NLProblem &nl_problem, Eigen::MatrixXd &sol)
	{
		assert(sol.size() == nl_problem.full_size());

		Eigen::VectorXd tmp_sol = nl_problem.full_to_reduced(sol);
		nl_problem.line_search_begin(sol, tmp_sol);

		if (!std::isfinite(nl_problem.value(tmp_sol))
			|| !nl_problem.is_step_valid(sol, tmp_sol)
			|| !nl_problem.is_step_collision_free(sol, tmp_sol))
			log_and_throw_error("Failed to apply constraints conditions; solve with augmented lagrangian first!");

		// --------------------------------------------------------------------
		// Perform one final solve with the DBC projected out

		logger().debug("Successfully applied constraints conditions; solving in reduced space");

		nl_problem.init(sol);
		update_barrier_stiffness(sol);
		try
		{
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

	void ALSolver::set_al_weight(NLProblem &nl_problem, const Eigen::VectorXd &x, const double weight)
	{
		if (alagr_forms.empty())
			return;
		if (weight > 0)
		{
			for (auto &f : alagr_forms)
				f->enable();

			nl_problem.use_full_size();
		}
		else
		{
			for (auto &f : alagr_forms)
				f->disable();

			nl_problem.use_reduced_size();
		}
	}

} // namespace polyfem::solver