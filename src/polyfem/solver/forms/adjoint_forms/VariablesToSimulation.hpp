#pragma once

#include <polyfem/solver/forms/parametrization/Parametrization.hpp>

namespace polyfem::solver
{
	namespace
	{

		void update_shape(std::shared_ptr<State> state_ptr)
		{
		}

		void update_material_params(std::shared_ptr<State> state_ptr)
		{
		}

	} // namespace
	class VariablesToSimulation
	{
		VariablesToSimulation(std::shared_ptr<State> state_ptr, const CompositeParametrization &parametrization, const ParameterTypes &parameter_type) : state_ptr_(state_ptr), parametrization_(parametrization), parameter_type_(parameter_type)
		{
		}

		inline void update(const Eigen::VectorXd &x)
		{
			switch (parameter_type)
			{
			}
		}

		inline Eigen::VectorXd apply_jacobian(const Eigen::VectorXd &grad_full, const Eigen::VectorXd &x) const
		{
			return parametrization_->apply_jacobian(grad_full, x);
		}

	private:
		std::shared_ptr<State> state_ptr_;
		CompositeParametrization parametrization_;
		ParameterType parameter_type_;
	}

} // namespace polyfem::solver