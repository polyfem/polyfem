#pragma once

#include "Parameter.hpp"

namespace polyfem
{
	class TopologyOptimizationParameter : Parameter
	{
	public:
		TopologyOptimizationParameter(std::vector<State> states_ptr){};

		void update() override
		{
		}
	};
} // namespace polyfem