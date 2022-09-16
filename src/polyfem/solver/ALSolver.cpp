#include "ALSolver.hpp"

#include <polyfem/utils/Logger.hpp>

namespace polyfem::solver
{

	ALSolver::ALSolver(
		std::shared_ptr<cppoptlib::NonlinearSolver<NLProblem>> nl_solver,
		std::shared_ptr<ALForm> al_form,
		const double initial_al_weight,
		const double max_al_weight,
		const std::function<void(const Eigen::VectorXd &)> &updated_barrier_stiffness)
		: nl_solver(nl_solver),
		  al_form(al_form),
		  initial_al_weight(initial_al_weight),
		  max_al_weight(max_al_weight),
		  updated_barrier_stiffness(updated_barrier_stiffness)
	{
	}

	void ALSolver::solve(
		NLProblem &nl_problem,
		Eigen::MatrixXd &sol,
		bool force_al)
	{
		assert(sol.size() == nl_problem.full_size());

		Eigen::VectorXd tmp_sol = nl_problem.full_to_reduced(sol);
		assert(tmp_sol.size() == nl_problem.reduced_size());

		// --------------------------------------------------------------------

		double al_weight = initial_al_weight;

		nl_problem.line_search_begin(sol, tmp_sol);
		while (force_al
			   || !std::isfinite(nl_problem.value(tmp_sol))
			   || !nl_problem.is_step_valid(sol, tmp_sol)
			   || !nl_problem.is_step_collision_free(sol, tmp_sol))
		{
			force_al = false;
			nl_problem.line_search_end();

			set_al_weight(nl_problem, sol, al_weight);
			logger().debug("Solving AL Problem with weight {}", al_weight);

			nl_problem.init(sol);
			updated_barrier_stiffness(sol);
			tmp_sol = sol;
			nl_solver->minimize(nl_problem, tmp_sol);

			sol = tmp_sol;
			set_al_weight(nl_problem, sol, -1);
			tmp_sol = nl_problem.full_to_reduced(sol);
			nl_problem.line_search_begin(sol, tmp_sol);

			al_weight *= 2;

			if (al_weight >= max_al_weight)
			{
				log_and_throw_error(fmt::format("Unable to solve AL problem, weight {} >= {}, stopping", al_weight, max_al_weight));
				break;
			}

			post_subsolve(al_weight);
		}
		nl_problem.line_search_end();

		// --------------------------------------------------------------------
		// Perform one final solve with the DBC projected out

		nl_problem.init(sol);
		updated_barrier_stiffness(sol);
		nl_solver->minimize(nl_problem, tmp_sol);
		sol = nl_problem.reduced_to_full(tmp_sol);

		post_subsolve(0);
	}

	void ALSolver::set_al_weight(NLProblem &nl_problem, const Eigen::VectorXd &x, const double weight)
	{
		if (al_form == nullptr)
			return;
		if (weight > 0)
		{
			al_form->enable();
			al_form->set_weight(weight);
			nl_problem.use_full_size();
			nl_problem.set_apply_DBC(x, false);
		}
		else
		{
			al_form->disable();
			nl_problem.use_reduced_size();
			nl_problem.set_apply_DBC(x, true);
		}
	}

} // namespace polyfem::solver