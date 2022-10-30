#pragma once

#include "Parameter.hpp"

namespace polyfem
{
	class ElasticParameter : Parameter
	{
	public:
		ElasticParameter(std::vector<State> states_ptr){};

		void update() override
		{
		}
	};
} // namespace polyfem