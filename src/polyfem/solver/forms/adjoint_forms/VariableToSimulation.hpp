#pragma once

#include <polyfem/solver/forms/parametrization/Parametrization.hpp>
#include <polyfem/solver/AdjointTools.hpp>

namespace polyfem::solver
{
	class VariableToSimulation
	{
	public:
		VariableToSimulation(const std::shared_ptr<State> &state_ptr, const CompositeParametrization &parametrization) : state_ptr_(state_ptr), parametrization_(parametrization) {}
		virtual ~VariableToSimulation() {}

		inline virtual void update(const Eigen::VectorXd &x) final
		{
			Eigen::VectorXd state_variable = parametrization_.eval(x);
			update_state(state_variable);
		}

		inline const State &get_state() const { return *state_ptr_; }
		inline CompositeParametrization get_parametrization() const { return parametrization_; }
		virtual ParameterType get_parameter_type() const = 0;

	protected:
		virtual void update_state(const Eigen::VectorXd &state_variable) = 0;
		std::shared_ptr<State> state_ptr_;
		CompositeParametrization parametrization_;
	};

	class ShapeVariableToSimulation : public VariableToSimulation
	{
	public:
		ShapeVariableToSimulation(const std::shared_ptr<State> &state_ptr, const CompositeParametrization &parametrization) : VariableToSimulation(state_ptr, parametrization) {}
		virtual ~ShapeVariableToSimulation() {}
		
		ParameterType get_parameter_type() const override { return ParameterType::Shape; }
	protected:
		void update_state(const Eigen::VectorXd &state_variable) override;
	};

	class ElasticVariableToSimulation : public VariableToSimulation
	{
	public:
		ElasticVariableToSimulation(const std::shared_ptr<State> &state_ptr, const CompositeParametrization &parametrization) : VariableToSimulation(state_ptr, parametrization) {}
		virtual ~ElasticVariableToSimulation() {}

		ParameterType get_parameter_type() const override { return ParameterType::Material; }
	protected:
		void update_state(const Eigen::VectorXd &state_variable) override;
	};

	class FrictionCoeffientVariableToSimulation : public VariableToSimulation
	{
	public:
		FrictionCoeffientVariableToSimulation(const std::shared_ptr<State> &state_ptr, const CompositeParametrization &parametrization) : VariableToSimulation(state_ptr, parametrization) {}
		virtual ~FrictionCoeffientVariableToSimulation() {}

		ParameterType get_parameter_type() const override { return ParameterType::FrictionCoeff; }
	protected:
		void update_state(const Eigen::VectorXd &state_variable) override;
	};

	class DampingCoeffientVariableToSimulation : public VariableToSimulation
	{
	public:
		DampingCoeffientVariableToSimulation(const std::shared_ptr<State> &state_ptr, const CompositeParametrization &parametrization) : VariableToSimulation(state_ptr, parametrization) {}
		virtual ~DampingCoeffientVariableToSimulation() {}

		ParameterType get_parameter_type() const override { return ParameterType::DampingCoeff; }
	protected:
		void update_state(const Eigen::VectorXd &state_variable) override;
	};

	class InitialConditionVariableToSimulation : public VariableToSimulation
	{
	public:
		InitialConditionVariableToSimulation(const std::shared_ptr<State> &state_ptr, const CompositeParametrization &parametrization) : VariableToSimulation(state_ptr, parametrization) {}
		virtual ~InitialConditionVariableToSimulation() {}

		ParameterType get_parameter_type() const override { return ParameterType::InitialCondition; }
	protected:
		void update_state(const Eigen::VectorXd &state_variable) override;
	};

	class DirichletVariableToSimulation : public VariableToSimulation
	{
	public:
		DirichletVariableToSimulation(const std::shared_ptr<State> &state_ptr, const CompositeParametrization &parametrization) : VariableToSimulation(state_ptr, parametrization) {}
		virtual ~DirichletVariableToSimulation() {}

		ParameterType get_parameter_type() const override { return ParameterType::DirichletBC; }
	protected:
		void update_state(const Eigen::VectorXd &state_variable) override;
	};

	class MacroStrainVariableToSimulation : public VariableToSimulation
	{
	public:
		MacroStrainVariableToSimulation(const std::shared_ptr<State> &state_ptr, const CompositeParametrization &parametrization) : VariableToSimulation(state_ptr, parametrization) {}
		virtual ~MacroStrainVariableToSimulation() {}

		ParameterType get_parameter_type() const override { return ParameterType::MacroStrain; }
	protected:
		void update_state(const Eigen::VectorXd &state_variable);
	};

} // namespace polyfem::solver