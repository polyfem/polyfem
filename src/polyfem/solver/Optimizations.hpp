#pragma once

#include "OptimizationProblem.hpp"

namespace polyfem
{
	void single_optimization(State &state, const std::shared_ptr<CompositeFunctional> j);

	void general_optimization(State &state, const std::shared_ptr<CompositeFunctional> j);
} // namespace polyfem