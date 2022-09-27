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
	template <class Problem = NLProblem>
	class ALSolver
	{
		using NLSolver = cppoptlib::NonlinearSolver<Problem>;

	public:
		ALSolver(
			std::shared_ptr<NLSolver> nl_solver,
			std::shared_ptr<ALForm> al_form,
			const double initial_al_weight,
			const double max_al_weight,
			const std::function<void(const Eigen::VectorXd &)> &updated_barrier_stiffness);

		void solve(Problem &nl_problem, Eigen::MatrixXd &sol, bool force_al = false);

		std::function<void(const double)> post_subsolve = [](const double) {};

	protected:
		void set_al_weight(Problem &nl_problem, const Eigen::VectorXd &x, const double weight);

		std::shared_ptr<NLSolver> nl_solver;
		std::shared_ptr<ALForm> al_form;
		double initial_al_weight;
		double max_al_weight;

		// TODO: replace this with a member function
		std::function<void(const Eigen::VectorXd &)> updated_barrier_stiffness;
	};
} // namespace polyfem::solver

#include "ALSolver.tpp"