#pragma once

#include "Parameter.hpp"

namespace polyfem
{
	class InitialConditionParameter : Parameter
	{
	public:
		InitialConditionParameter(std::vector<State> states_ptr){};

		void update() override
		{
		}
	};
} // namespace polyfem