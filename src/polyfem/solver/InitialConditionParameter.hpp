#pragma once

#include "Parameter.hpp"

namespace polyfem
{
	class InitialConditionParameter : public Parameter
	{
	public:
		InitialConditionParameter(std::vector<std::shared_ptr<State>> states_ptr, const json &args): Parameter(states_ptr, args) 
		{
			parameter_name_ = "initial";
			full_dim_ = states_ptr[0]->ndof() * 2;
		}

		Eigen::VectorXd initial_guess() const override
		{
			assert(false);
			return Eigen::VectorXd();
		}
		
		void update() override
		{
		}
	};
} // namespace polyfem