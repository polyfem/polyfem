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
	class ALSolver
	{
		using NLSolver = cppoptlib::NonlinearSolver<NLProblem>;

	public:
		ALSolver(
			std::shared_ptr<NLSolver> nl_solver,
			std::shared_ptr<ALForm> al_form,
			const double initial_al_weight,
			const double scaling,
			const int max_al_steps,
			const std::function<void(const Eigen::VectorXd &)> &update_barrier_stiffness);

		void solve(NLProblem &nl_problem, Eigen::MatrixXd &sol, bool force_al = false);

		std::function<void(const double)> post_subsolve = [](const double) {};

	protected:
		void set_al_weight(NLProblem &nl_problem, const Eigen::VectorXd &x, const double weight, const std::vector<double> &initial_weight);

		std::shared_ptr<NLSolver> nl_solver;
		std::shared_ptr<ALForm> al_form;
		const double initial_al_weight;
		const double scaling;
		const int max_al_steps;

		// TODO: replace this with a member function
		std::function<void(const Eigen::VectorXd &)> update_barrier_stiffness;
	};
} // namespace polyfem::solver