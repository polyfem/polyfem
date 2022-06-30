#pragma once

#include <polyfem/CompositeFunctional.hpp>

namespace polyfem
{
	void initial_condition_optimization(State &state, const std::shared_ptr<CompositeFunctional> j, json &opt_params);

	void material_optimization(State &state, const std::shared_ptr<CompositeFunctional> j, json &opt_params);

	void shape_optimization(State &state, const std::shared_ptr<CompositeFunctional> j, json &opt_params);
} // namespace polyfem