#pragma once

#include <polyfem/Common.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/optimization/DiffCache.hpp>

#include <Eigen/Core>
#include <memory>
#include <vector>

namespace polysolve::nonlinear
{
	class Solver;
}

namespace polyfem::solver
{
	class AdjointNLProblem;
	class AdjointForm;
	class Parametrization;
	class VariableToSimulation;
	class VariableToSimulationGroup;

	struct AdjointOptUtils
	{
		static json apply_opt_json_spec(const json &input_args, bool strict_validation);

		static std::shared_ptr<polysolve::nonlinear::Solver> make_nl_solver(
			const json &solver_params,
			const json &linear_solver_params,
			const double characteristic_length,
			const bool strict_validation = true);

		static Eigen::VectorXd inverse_evaluation(const json &args, const int ndof, const std::vector<int> &variable_sizes, VariableToSimulationGroup &var2sim);

		static void solve_pde(varform::DifferentiableVarForm &varform);

		static int compute_variable_size(const json &args, const std::vector<std::shared_ptr<varform::DifferentiableVarForm>> &varforms);
	};
} // namespace polyfem::solver
