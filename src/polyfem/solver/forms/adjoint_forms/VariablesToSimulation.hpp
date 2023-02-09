#pragma once

#include <polyfem/solver/forms/parametrization/Parametrization.hpp>

namespace polyfem::solver
{
	class VariablesToSimulation
	{
		VariablesToSimulation(std::shared_ptr<State> state_ptr, const std::vector<CompositeParametrization> &parametrizations, const std::vector<ParameterTypes> &parameter_types) : state_ptr_(state_ptr), parametrizations_(parametrizations), parameter_types_(parameter_types)
		{
		}

	private:
		std::shared_ptr<State> state_ptr_;
		std::vector<CompositeParametrization> parametrizations_;
		std::vector<ParameterType> parameter_types_;
	}

} // namespace polyfem::solver