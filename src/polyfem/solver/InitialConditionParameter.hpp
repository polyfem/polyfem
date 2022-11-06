#pragma once

#include "Parameter.hpp"

namespace polyfem
{
	class InitialConditionParameter : public Parameter
	{
	public:
		InitialConditionParameter(std::vector<std::shared_ptr<State>> states_ptr): Parameter(states_ptr) 
		{
			parameter_name_ = "initial";
			full_dim_ = states_ptr[0]->ndof() * 2;
		}

		void update() override
		{
		}
	};
} // namespace polyfem