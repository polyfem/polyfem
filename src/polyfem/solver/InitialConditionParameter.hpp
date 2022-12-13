#pragma once

#include "Parameter.hpp"

namespace polyfem
{
	class InitialConditionParameter : public Parameter
	{
	public:
		InitialConditionParameter(std::vector<std::shared_ptr<State>> &states_ptr, const json &args): Parameter(states_ptr, args) 
		{
			parameter_name_ = "initial";
			full_dim_ = states_ptr[0]->ndof() * 2;
			optimization_dim_ = states_ptr[0]->ndof() * 2;
		}

		Eigen::VectorXd initial_guess() const override
		{
			Eigen::VectorXd x(get_state().ndof() * 2);
			x.head(get_state().ndof()) += get_state().initial_sol_update;
			x.tail(get_state().ndof()) += get_state().initial_vel_update;
			return x;
		}
		
		void update() override
		{
		}
	};
} // namespace polyfem