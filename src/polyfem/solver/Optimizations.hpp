#pragma once

#include <polyfem/utils/CompositeFunctional.hpp>
#include "OptimizationProblem.hpp"

namespace polyfem
{
	void initial_condition_optimization(State &state, const std::shared_ptr<CompositeFunctional> j);

	void material_optimization(State &state, const std::shared_ptr<CompositeFunctional> j);

	void shape_optimization(State &state, const std::shared_ptr<CompositeFunctional> j);

	void general_optimization(State &state, const std::vector<std::shared_ptr<OptimizationProblem>> params, const std::shared_ptr<CompositeFunctional> j);
	
	void topology_optimization(State &state, const std::shared_ptr<CompositeFunctional> j);
} // namespace polyfem