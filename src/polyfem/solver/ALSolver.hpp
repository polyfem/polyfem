#pragma once

#include <polyfem/solver/NLProblem.hpp>
#include <polysolve/nonlinear/Solver.hpp>
#include <polyfem/solver/forms/lagrangian/AugmentedLagrangianForm.hpp>
#include <polyfem/Common.hpp>

#include <Eigen/Core>

#include <functional>
#include <vector>

namespace polyfem::solver
{
	class ALSolver
	{
		using NLSolver = polysolve::nonlinear::Solver;

	public:
		ALSolver(
			const std::vector<std::shared_ptr<AugmentedLagrangianForm>> &alagr_form,
			const double initial_al_weight,
			const double scaling,
			const double max_al_weight,
			const double eta_tol,
			const std::function<void(const Eigen::VectorXd &)> &update_barrier_stiffness);
		virtual ~ALSolver() = default;

		void solve_al(std::shared_ptr<NLSolver> nl_solver, NLProblem &nl_problem, Eigen::MatrixXd &sol);
		void solve_reduced(std::shared_ptr<NLSolver> nl_solver, NLProblem &nl_problem, Eigen::MatrixXd &sol);

		std::function<void(const double)> post_subsolve = [](const double) {};

	protected:
		void set_al_weight(NLProblem &nl_problem, const Eigen::VectorXd &x, const double weight);

		std::vector<std::shared_ptr<AugmentedLagrangianForm>> alagr_forms;
		const double initial_al_weight;
		const double scaling;
		const double max_al_weight;
		const double eta_tol;

		// TODO: replace this with a member function
		std::function<void(const Eigen::VectorXd &)> update_barrier_stiffness;
	};
} // namespace polyfem::solver