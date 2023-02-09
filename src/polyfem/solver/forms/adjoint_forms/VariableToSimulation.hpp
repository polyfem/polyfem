#pragma once

#include <polyfem/solver/forms/parametrization/Parameterization.hpp>
#include <polyfem/solver/AdjointTools.hpp>

namespace polyfem::solver
{
	class VariableToSimulation
	{
		VariableToSimulation(std::shared_ptr<State> state_ptr, const CompositeParametrization &parametrization) : state_ptr_(state_ptr), parametrization_(parametrization) {}

		inline virtual void update(const Eigen::VectorXd &x) final
		{
			Eigen::VectorXd state_variable = parametrization.eval(x);
			update_state(state_variable);
		}

		inline virtual Eigen::VectorXd apply_jacobian(const Eigen::VectorXd &grad_full, const Eigen::VectorXd &x) const final
		{
			return parametrization_->apply_jacobian(grad_full, x);
		}

		inline std::shared_ptr<State> get_state() const { return state_ptr_; }
		inline CompositeParameterization get_parameterization() const { return parametrization_; }

	protected:
		virtual void update_state(const Eigen::MatrixXd &state_variable) = 0;
		std::shared_ptr<State> state_ptr_;
		CompositeParametrization parametrization_;
	}

	class ShapeVariableToSimulation : public VariableToSimulation
	{
		inline void update_state(const Eigen::VectorXd &state_variable) override
		{
		}
	}

	class MaterialVariableToSimulation : public VariableToSimulation
	{
		inline void update_state(const Eigen::VectorXd &state_variable) override
		{
		}
	}

	class InitialConditionVariableToSimulation : public VariableToSimulation
	{
		inline void update_state(const Eigen::VectorXd &state_variable) override
		{
		}
	}

	class DirichletVariableToSimulation : public VariableToSimulation
	{
		inline void update_state(const Eigen::VectorXd &state_variable) override
		{
		}
	}

} // namespace polyfem::solver