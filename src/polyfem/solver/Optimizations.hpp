#pragma once

#include <polyfem/Common.hpp>

namespace polyfem
{
	class State;
}

namespace cppoptlib
{
	template <typename ProblemType>
	class NonlinearSolver;
}

namespace polyfem::solver
{
	class AdjointNLProblem;
	class AdjointForm;
	class Parametrization;
	class VariableToSimulation;

	json apply_opt_json_spec(const json &input_args, bool strict_validation);

	std::shared_ptr<cppoptlib::NonlinearSolver<AdjointNLProblem>> make_nl_solver(const json &solver_params);

	std::shared_ptr<State> create_state(const json &args, const int max_threads = 32);

	void solve_pde(State &state);

	std::shared_ptr<AdjointForm> create_form(const json &args, const std::vector<std::shared_ptr<VariableToSimulation>> &var2sim, const std::vector<std::shared_ptr<State>> &states);

	std::shared_ptr<Parametrization> create_parametrization(const json &args, const std::vector<std::shared_ptr<State>> &states, const std::vector<int> &variable_sizes);

	std::shared_ptr<VariableToSimulation> create_variable_to_simulation(const json &args, const std::vector<std::shared_ptr<State>> &states, const std::vector<int> &variable_sizes);

	int compute_variable_size(const json &args, const std::vector<std::shared_ptr<State>> &states);

} // namespace polyfem