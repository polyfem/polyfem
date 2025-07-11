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
			const std::function<void(const Eigen::VectorXd &)> &update_barrier_stiffness,
			const std::function<void(const Eigen::VectorXd &)> &update_al_weight);
		virtual ~ALSolver() = default;

		void solve_al(NLProblem &nl_problem, Eigen::MatrixXd &sol,
					  std::shared_ptr<polysolve::nonlinear::Solver> nl_solver)
		{
			solve_al(nl_problem, sol, json{}, json{}, 1, nl_solver);
		}

		void solve_al(NLProblem &nl_problem, Eigen::MatrixXd &sol,
					  const json &nl_solver_params,
					  const json &linear_solver,
					  const double characteristic_length,
					  std::shared_ptr<polysolve::nonlinear::Solver> nl_solver = nullptr);

		void solve_reduced(NLProblem &nl_problem, Eigen::MatrixXd &sol,
						   std::shared_ptr<polysolve::nonlinear::Solver> nl_solver)
		{
			solve_al(nl_problem, sol, json{}, json{}, 1, nl_solver);
		}

		void solve_reduced(NLProblem &nl_problem, Eigen::MatrixXd &sol,
						   const json &nl_solver_params,
						   const json &linear_solver,
						   const double characteristic_length,
						   std::shared_ptr<polysolve::nonlinear::Solver> nl_solver = nullptr);

		std::function<void(const double)> post_subsolve = [](const double) {};


	protected:
		std::vector<std::shared_ptr<AugmentedLagrangianForm>> alagr_forms;
		double initial_al_weight;
		const double scaling;
		const double max_al_weight;
		const double eta_tol;

		// TODO: replace this with a member function
		std::function<void(const Eigen::VectorXd &)> update_barrier_stiffness;
		std::function<void(const Eigen::VectorXd &)> update_al_weight;
	};
} // namespace polyfem::solver