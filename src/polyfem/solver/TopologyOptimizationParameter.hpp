#pragma once

#include "Parameter.hpp"

namespace polyfem
{
	class TopologyOptimizationParameter : public Parameter
	{
	public:
		TopologyOptimizationParameter(std::vector<std::shared_ptr<State>> states_ptr): Parameter(states_ptr)
		{
			parameter_name_ = "topology";

			full_dim_ = states_ptr[0]->bases.size();
		}

		void update() override
		{
		}
	};
} // namespace polyfem