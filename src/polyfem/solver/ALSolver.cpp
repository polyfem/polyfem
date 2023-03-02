#include "ALSolver.hpp"

#include <polyfem/utils/Logger.hpp>

namespace polyfem::solver
{
	ALSolver::ALSolver(
		std::shared_ptr<NLSolver> nl_solver,
		std::shared_ptr<ALForm> al_form,
		const double initial_al_weight,
		const double scaling,
		const int max_al_steps,
		const std::function<void(const Eigen::VectorXd &)> &update_barrier_stiffness)
		: nl_solver(nl_solver),
		  al_form(al_form),
		  initial_al_weight(initial_al_weight),
		  scaling(scaling),
		  max_al_steps(max_al_steps),
		  update_barrier_stiffness(update_barrier_stiffness)
	{
	}

	void ALSolver::solve(NLProblem &nl_problem, Eigen::MatrixXd &sol, bool force_al)
	{
		assert(sol.size() == nl_problem.full_size());

		Eigen::VectorXd tmp_sol = nl_problem.full_to_reduced(sol);
		assert(tmp_sol.size() == nl_problem.reduced_size());

		std::vector<double> initial_weight;
		for (const auto &f : nl_problem.forms())
		{
			initial_weight.push_back(f->weight());
		}

		// --------------------------------------------------------------------

		double al_weight = initial_al_weight;
		int al_steps = 0;

		nl_problem.line_search_begin(sol, tmp_sol);
		while (force_al
			   || !std::isfinite(nl_problem.value(tmp_sol))
			   || !nl_problem.is_step_valid(sol, tmp_sol)
			   || !nl_problem.is_step_collision_free(sol, tmp_sol))
		{
			force_al = false;
			nl_problem.line_search_end();

			set_al_weight(nl_problem, sol, al_weight, initial_weight);
			logger().debug("Solving AL Problem with weight {}", al_weight);

			nl_problem.init(sol);
			update_barrier_stiffness(sol);
			tmp_sol = sol;
			nl_solver->minimize(nl_problem, tmp_sol);

			sol = tmp_sol;
			set_al_weight(nl_problem, sol, -1, initial_weight);
			tmp_sol = nl_problem.full_to_reduced(sol);
			nl_problem.line_search_begin(sol, tmp_sol);

			al_weight /= scaling;

			if (al_steps >= max_al_steps)
			{
				log_and_throw_error(fmt::format("Unable to solve AL problem, out of iterations {} (current weight = {}), stopping", max_al_steps, al_weight));
				break;
			}

			post_subsolve(al_weight);
			++al_steps;
		}
		nl_problem.line_search_end();

		// --------------------------------------------------------------------
		// Perform one final solve with the DBC projected out

		logger().debug("Successfully applied boundary conditions; solving in reduced space:");

		nl_problem.init(sol);
		update_barrier_stiffness(sol);
		try
		{
			nl_solver->minimize(nl_problem, tmp_sol);
		}
		catch (const std::runtime_error &e)
		{
			sol = nl_problem.reduced_to_full(tmp_sol);
			throw e;
		}
		sol = nl_problem.reduced_to_full(tmp_sol);

		post_subsolve(0);
	}

	void ALSolver::set_al_weight(NLProblem &nl_problem, const Eigen::VectorXd &x, const double weight, const std::vector<double> &initial_weight)
	{
		if (al_form == nullptr)
			return;
		if (weight > 0)
		{
			for (int i = 0; i < nl_problem.forms().size(); ++i)
			{
				nl_problem.forms()[i]->set_weight(initial_weight[i] * weight);
			}
			al_form->enable();
			al_form->set_weight(1 - weight);
			nl_problem.use_full_size();
			nl_problem.set_apply_DBC(x, false);
		}
		else
		{
			for (int i = 0; i < nl_problem.forms().size(); ++i)
			{
				nl_problem.forms()[i]->set_weight(initial_weight[i]);
			}

			al_form->disable();
			nl_problem.use_reduced_size();
			nl_problem.set_apply_DBC(x, true);
		}
	}

} // namespace polyfem::solver