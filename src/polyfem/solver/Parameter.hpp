#pragma once

#include "OptimizationProblem.hpp"

namespace polyfem
{
	class Parameter
	{
	public:
		Parameter(std::vector<State> states_ptr) : states_ptr_(states_ptr){};

		virtual void update() = 0;

	private:
		std::vector<std::shared_ptr<State>> states_ptr_;
	};
} // namespace polyfem