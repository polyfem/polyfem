#include "ALSolver.hpp"

#include <polyfem/utils/Logger.hpp>

namespace polyfem::solver
{

	void solve_al_nl_problem(
		NLProblem &nl_problem,
		const int t,
		const double initial_al_weight,
		const double max_al_weight,
		const std::function<bool(const Eigen::VectorXd &, const Eigen::VectorXd &)> is_step_collision_free,
		const std::function<void(const double)> set_al_weight,
		const std::function<std::shared_ptr<cppoptlib::NonlinearSolver<NLProblem>>()> make_nl_solver,
		const std::string &line_search_method,
		const std::function<void(const Eigen::VectorXd &)> updated_barrier_stiffness,
		Eigen::MatrixXd &sol,
		json &solver_info,
		const std::function<void(void)> post_solve,
		bool force_al)
	{
		// assert(sol.size() == rhs.size());

		Eigen::VectorXd tmp_sol;
		nl_problem.full_to_reduced(sol, tmp_sol);
		// assert(sol.size() == rhs.size());
		// assert(tmp_sol.size() <= rhs.size());

		///////////////////////////////////////////////////////////////////////

		double al_weight = initial_al_weight;

		nl_problem.line_search_begin(sol, tmp_sol);
		Eigen::VectorXd tmp_sol_full;
		nl_problem.reduced_to_full(tmp_sol, tmp_sol_full);
		while (force_al
			   || !std::isfinite(nl_problem.value(tmp_sol))
			   || !nl_problem.is_step_valid(sol, tmp_sol)
			   || !is_step_collision_free(sol, tmp_sol_full))
		{
			force_al = false;
			nl_problem.line_search_end();
			set_al_weight(al_weight);
			logger().debug("Solving AL Problem with weight {}", al_weight);

			std::shared_ptr<cppoptlib::NonlinearSolver<NLProblem>> alnl_solver = make_nl_solver();
			alnl_solver->setLineSearch(line_search_method);
			nl_problem.init(sol);
			updated_barrier_stiffness(sol);
			tmp_sol = sol;
			alnl_solver->minimize(nl_problem, tmp_sol);
			json alnl_solver_info;
			alnl_solver->getInfo(alnl_solver_info);

			solver_info.push_back(
				{{"type", "al"},
				 {"t", t}, // TODO: null if static?
				 {"weight", al_weight},
				 {"info", alnl_solver_info}});

			sol = tmp_sol;
			set_al_weight(-1);
			nl_problem.full_to_reduced(sol, tmp_sol);
			nl_problem.line_search_begin(sol, tmp_sol);
			nl_problem.reduced_to_full(tmp_sol, tmp_sol_full);

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
		nl_solver->setLineSearch(line_search_method);
		nl_problem.init(sol);
		updated_barrier_stiffness(sol);
		nl_solver->minimize(nl_problem, tmp_sol);
		json nl_solver_info;
		nl_solver->getInfo(nl_solver_info);
		solver_info.push_back(
			{{"type", "rc"},
			 {"t", t}, // TODO: null if static?
			 {"info", nl_solver_info}});
		nl_problem.reduced_to_full(tmp_sol, sol);

		post_solve();
	}

} // namespace polyfem::solver