#pragma once

#include <polyfem/solver/NLProblem.hpp>
#include <polyfem/solver/NonlinearSolver.hpp>
#include <polyfem/Common.hpp>

#include <Eigen/Core>

#include <functional>
#include <vector>

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
		const std::function<void(void)> post_solve = []() {},
		bool force_al = false);

}