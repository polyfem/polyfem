#pragma once

#include "Parameter.hpp"

namespace polyfem
{
	class FrictionParameter : Parameter
	{
	public:
		FrictionParameter(std::vector<State> states_ptr){};

		void update() override
		{
		}
	};
} // namespace polyfem