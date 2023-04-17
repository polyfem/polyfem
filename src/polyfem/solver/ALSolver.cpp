#include "ALSolver.hpp"

#include <polyfem/utils/Logger.hpp>

namespace polyfem::solver
{
	ALSolver::ALSolver(
		std::shared_ptr<NLSolver> nl_solver,
		std::shared_ptr<BCLagrangianForm> lagr_form,
		std::shared_ptr<BCPenaltyForm> pen_form,
		const double initial_al_weight,
		const double scaling,
		const double max_al_weight,
		const double eta_tol,
		const int max_solver_iter,
		const std::function<void(const Eigen::VectorXd &)> &update_barrier_stiffness)
		: nl_solver(nl_solver),
		  lagr_form(lagr_form),
		  pen_form(pen_form),
		  initial_al_weight(initial_al_weight),
		  scaling(scaling),
		  max_al_weight(max_al_weight),
		  eta_tol(eta_tol),
		  max_solver_iter(max_solver_iter),
		  update_barrier_stiffness(update_barrier_stiffness)
	{
	}

	void ALSolver::solve(NLProblem &nl_problem, Eigen::MatrixXd &sol, bool force_al)
	{
		assert(sol.size() == nl_problem.full_size());

		Eigen::VectorXd tmp_sol = nl_problem.full_to_reduced(sol);
		assert(tmp_sol.size() == nl_problem.reduced_size());

		// --------------------------------------------------------------------

		double al_weight = initial_al_weight;
		int al_steps = 0;
		const int iters = nl_solver->max_iterations();
		nl_solver->max_iterations() = max_solver_iter;

		const StiffnessMatrix &mask = pen_form->mask();
		const double initial_error = (pen_form->target() - sol).transpose() * mask * (pen_form->target() - sol);

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
			update_barrier_stiffness(sol);
			tmp_sol = sol;

			try
			{
				nl_solver->minimize(nl_problem, tmp_sol);
			}
			catch (const std::runtime_error &e)
			{
			}

			sol = tmp_sol;
			set_al_weight(nl_problem, sol, -1);
			tmp_sol = nl_problem.full_to_reduced(sol);
			nl_problem.line_search_begin(sol, tmp_sol);

			const double current_error = (pen_form->target() - sol).transpose() * mask * (pen_form->target() - sol);
			const double eta = 1 - sqrt(current_error / initial_error);

			logger().debug("Current eta = {}", eta);

			if (eta < eta_tol && al_weight < max_al_weight)
				al_weight *= scaling;
			else
				lagr_form->update_lagrangian(sol, al_weight);

			post_subsolve(al_weight);
			++al_steps;
		}
		nl_problem.line_search_end();
		nl_solver->max_iterations() = iters;

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

	void ALSolver::set_al_weight(NLProblem &nl_problem, const Eigen::VectorXd &x, const double weight)
	{
		if (pen_form == nullptr || lagr_form == nullptr)
			return;
		if (weight > 0)
		{
			pen_form->enable();
			lagr_form->enable();
			pen_form->set_weight(weight);
			nl_problem.use_full_size();
			nl_problem.set_apply_DBC(x, false);
		}
		else
		{
			pen_form->disable();
			lagr_form->disable();
			nl_problem.use_reduced_size();
			nl_problem.set_apply_DBC(x, true);
		}
	}

} // namespace polyfem::solver