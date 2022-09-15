#include "ALSolver.hpp"

#include <polyfem/utils/Logger.hpp>

namespace polyfem::solver
{

	ALSolver::ALSolver(
		std::shared_ptr<cppoptlib::NonlinearSolver<NLProblem>> nl_solver,
		std::shared_ptr<NLProblem> nl_problem,
		std::shared_ptr<ALForm> al_form,
		const double initial_al_weight,
		const double max_al_weight,
		const std::function<void(const Eigen::VectorXd &)> &updated_barrier_stiffness)
		: nl_solver(nl_solver),
		  nl_problem(nl_problem),
		  al_form(al_form),
		  initial_al_weight(initial_al_weight),
		  max_al_weight(max_al_weight),
		  updated_barrier_stiffness(updated_barrier_stiffness)
	{
	}

	void ALSolver::solve(
		NLProblem &nl_problem,
		Eigen::MatrixXd &sol,
		json &solver_info,
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

			set_al_weight(al_weight);
			logger().debug("Solving AL Problem with weight {}", al_weight);

			// std::shared_ptr<cppoptlib::NonlinearSolver<NLProblem>> alnl_solver = make_nl_solver();
			// alnl_solver->set_line_search(line_search_method);

			nl_problem.init(sol);
			updated_barrier_stiffness(sol);
			tmp_sol = sol;
			alnl_solver->minimize(nl_problem, tmp_sol);
			json alnl_solver_info;
			alnl_solver->get_info(alnl_solver_info);

			solver_info.push_back(
				{{"type", "al"},
				 {"t", t}, // TODO: null if static?
				 {"weight", al_weight},
				 {"info", alnl_solver_info}});

			sol = tmp_sol;
			set_al_weight(-1);
			tmp_sol = nl_problem.full_to_reduced(sol);
			nl_problem.line_search_begin(sol, tmp_sol);

			al_weight *= 2;

			if (al_weight >= max_al_weight)
			{
				log_and_throw_error(fmt::format("Unable to solve AL problem, weight {} >= {}, stopping", al_weight, max_al_weight));
				break;
			}

			post_solve();
		}
		nl_problem.line_search_end();

		///////////////////////////////////////////////////////////////////////
		// Perform one final solve with the DBC projected out

		std::shared_ptr<cppoptlib::NonlinearSolver<NLProblem>> nl_solver = make_nl_solver();
		nl_solver->set_line_search(line_search_method);
		nl_problem.init(sol);
		updated_barrier_stiffness(sol);
		nl_solver->minimize(nl_problem, tmp_sol);
		json nl_solver_info;
		nl_solver->get_info(nl_solver_info);
		solver_info.push_back(
			{{"type", "rc"},
			 {"t", t}, // TODO: null if static?
			 {"info", nl_solver_info}});
		sol = nl_problem.reduced_to_full(tmp_sol);

		post_solve();
	}

} // namespace polyfem::solver