#pragma once

#include "Parameter.hpp"

namespace polyfem
{
	class ControlParameter : Parameter
	{
	public:
		ControlParameter(std::vector<State> states_ptr){};

		void update() override
		{
		}
	};
} // namespace polyfem