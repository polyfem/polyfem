#pragma once

#include <polyfem/solver/forms/parametrization/Parametrization.hpp>

namespace polyfem::solver
{
	class VariablesToSimulation
	{
		VariablesToSimulation(std::shared_ptr<State> state_ptr, const CompositeParametrization &parametrizations, const ParameterTypes &parameter_types) : state_ptr_(state_ptr), parametrizations_(parametrizations), parameter_types_(parameter_types)
		{
		}

		inline void update(const Eigen::VectorXd &x)
		{
		}

		inline Eigen::VectorXd apply_jacobian(const Eigen::VectorXd &grad_full, const Eigen::VectorXd &x) const
		{
			return Eigen::VectorXd();
		}

	private:
		std::shared_ptr<State> state_ptr_;
		CompositeParametrization parametrizations_;
		ParameterType parameter_types_;
	}

} // namespace polyfem::solver