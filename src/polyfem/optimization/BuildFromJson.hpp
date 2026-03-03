#pragma once

#include <polyfem/State.hpp>
#include <polyfem/Common.hpp>

#include <polyfem/optimization/CacheLevel.hpp>
#include <polyfem/optimization/DiffCache.hpp>
#include <polyfem/optimization/forms/AdjointForm.hpp>
#include <polyfem/optimization/forms/VariableToSimulation.hpp>
#include <polyfem/optimization/parametrization/Parametrization.hpp>

#include <string>
#include <memory>
#include <vector>
#include <cstddef>

namespace polyfem::from_json
{
	// Build a single State from an in-memory JSON configuration.
	// This mirrors the initialization done by build_states(), but does not load JSON from disk.
	std::shared_ptr<State> build_state(
		const json &args,
		const solver::CacheLevel &level,
		const size_t max_threads);

	std::vector<std::shared_ptr<State>> build_states(
		const std::string &root_path,
		const json &args,
		const solver::CacheLevel &level,
		const size_t max_threads);

	std::shared_ptr<solver::Parametrization> build_parametrization(
		const json &args,
		const std::vector<std::shared_ptr<State>> &states,
		const std::vector<int> &variable_sizes);

	std::unique_ptr<solver::VariableToSimulation> build_variable_to_simulation(
		const json &args,
		const std::vector<std::shared_ptr<State>> &states,
		const std::vector<std::shared_ptr<DiffCache>> &diff_caches,
		const std::vector<int> &variable_sizes);

	solver::VariableToSimulationGroup build_variable_to_simulation_group(
		const json &args,
		const std::vector<std::shared_ptr<State>> &states,
		const std::vector<std::shared_ptr<DiffCache>> &diff_caches,
		const std::vector<int> &variable_sizes);

	std::shared_ptr<solver::AdjointForm> build_form(
		const json &args,
		const solver::VariableToSimulationGroup &var2sim,
		const std::vector<std::shared_ptr<State>> &states,
		const std::vector<std::shared_ptr<DiffCache>> &diff_caches);

} // namespace polyfem::from_json
