#pragma once

#include <polyfem/solver/NLProblem.hpp>
#include <polyfem/solver/NonlinearSolver.hpp>
#include <polyfem/solver/forms/ALForm.hpp>
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

	class ALSolver
	{
	public:
		ALSolver(
			std::shared_ptr<cppoptlib::NonlinearSolver<NLProblem>> nl_solver,
			std::shared_ptr<ALForm> al_form,
			const double initial_al_weight,
			const double max_al_weight,
			const std::function<void(const Eigen::VectorXd &)> updated_barrier_stiffness);

		void solve(
			NLProblem &nl_problem,
			Eigen::MatrixXd &sol,
			json &solver_info,
			bool force_al = false);

		std::function<void(void)> post_subsolve = []() {};

	protected:
		void set_al_weight(NLProblem &nl_problem, const double weight)
		{
			if (al_form == nullptr)
				return;

			if (weight > 0)
			{
				al_form->set_enabled(true);
				al_form->set_weight(weight);
				nl_problem.use_full_size();
			}
			else
			{
				al_form->set_enabled(false);
				nl_problem.use_reduced_size();
			}
		}

		std::shared_ptr<cppoptlib::NonlinearSolver<NLProblem>> nl_solver;
		std::shared_ptr<ALForm> al_form;
		double initial_al_weight;
		double max_al_weight;

		// TODO: replace this with a member function
		std::function<void(const Eigen::VectorXd &)> updated_barrier_stiffness;
	};

} // namespace polyfem::solver