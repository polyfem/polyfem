#pragma once

#include "Parameter.hpp"

namespace polyfem
{
	class DampingParameter : Parameter
	{
	public:
		DampingParameter(std::vector<State> states_ptr){};

		void update() override
		{
		}
	};
} // namespace polyfem