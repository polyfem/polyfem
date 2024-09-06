#pragma once

#include <polyfem/Common.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/solver/DiffCache.hpp>

namespace polyfem
{
	class State;
}

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

		static std::shared_ptr<polysolve::nonlinear::Solver> make_nl_solver(const json &solver_params, const json &linear_solver_params, const double characteristic_length);

		static std::shared_ptr<State> create_state(const json &args, CacheLevel level, const size_t max_threads);

		static std::vector<std::shared_ptr<State>> create_states(const json &state_args, const CacheLevel &level, const size_t max_threads);

		static Eigen::VectorXd inverse_evaluation(const json &args, const int ndof, const std::vector<int> &variable_sizes, VariableToSimulationGroup &var2sim);

		static void solve_pde(State &state);

		static std::shared_ptr<AdjointForm> create_form(const json &args, const VariableToSimulationGroup &var2sim, const std::vector<std::shared_ptr<State>> &states);

		// forms that only depends on one simulator
		static std::shared_ptr<AdjointForm> create_simple_form(const std::string &obj_type, const std::string &param_type, const std::shared_ptr<State> &state, const json &args);

		static std::shared_ptr<Parametrization> create_parametrization(const json &args, const std::vector<std::shared_ptr<State>> &states, const std::vector<int> &variable_sizes);

		static std::unique_ptr<VariableToSimulation> create_variable_to_simulation(const json &args, const std::vector<std::shared_ptr<State>> &states, const std::vector<int> &variable_sizes);

		static int compute_variable_size(const json &args, const std::vector<std::shared_ptr<State>> &states);
	};
} // namespace polyfem::solver
