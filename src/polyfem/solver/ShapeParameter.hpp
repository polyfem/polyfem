#pragma once

#include "Parameter.hpp"

namespace polyfem
{
	class ShapeParameter : Parameter
	{
	public:
		ShapeParameter(std::vector<State> states_ptr){};

		void update() override
		{
		}
	};
} // namespace polyfem