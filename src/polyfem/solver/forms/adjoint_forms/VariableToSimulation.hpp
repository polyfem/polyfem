#pragma once

#include <polyfem/solver/forms/parametrization/Parametrization.hpp>
#include <polyfem/solver/AdjointTools.hpp>

namespace polyfem::solver
{
	class VariableToSimulation
	{
	public:
		VariableToSimulation(const std::vector<std::shared_ptr<State>> &states, const CompositeParametrization &parametrization) : states_(states), parametrization_(parametrization) { assert(states.size() > 0); }
		VariableToSimulation(const std::shared_ptr<State> &state, const CompositeParametrization &parametrization) : states_({state}), parametrization_(parametrization) {}
		virtual ~VariableToSimulation() {}

		inline virtual void update(const Eigen::VectorXd &x)
		{
			update_state(parametrization_.eval(x), get_output_indexing(x));
		}

		virtual std::string name() const = 0;

		inline int n_states() const { return states_.size(); }
		inline const std::vector<std::shared_ptr<State>> &get_states() const { return states_; }

		inline CompositeParametrization &get_parametrization() { return parametrization_; }
		virtual ParameterType get_parameter_type() const = 0;
		virtual Eigen::VectorXd compute_adjoint_term(const Eigen::VectorXd &x) const = 0;
		virtual Eigen::VectorXd inverse_eval();

		void set_output_indexing(const Eigen::VectorXi &output_indexing) { output_indexing_ = output_indexing; }
		Eigen::VectorXi get_output_indexing(const Eigen::VectorXd &x) const;

		virtual Eigen::VectorXd apply_parametrization_jacobian(const Eigen::VectorXd &term, const Eigen::VectorXd &x) const;

	protected:
		virtual void update_state(const Eigen::VectorXd &state_variable, const Eigen::VectorXi &indices);
		std::vector<std::shared_ptr<State>> states_;
		CompositeParametrization parametrization_;

		Eigen::VectorXi output_indexing_; // if a derived class overrides apply_parametrization_jacobian(term, x), this is not necessarily used.
	};

	// state variable dof = dim * n_vertices
	class ShapeVariableToSimulation : public VariableToSimulation
	{
	public:
		using VariableToSimulation::VariableToSimulation;
		virtual ~ShapeVariableToSimulation() {}

		std::string name() const override { return "shape"; }

		ParameterType get_parameter_type() const override { return ParameterType::Shape; }

		Eigen::VectorXd compute_adjoint_term(const Eigen::VectorXd &x) const override;
		virtual Eigen::VectorXd inverse_eval() override;

	protected:
		virtual void update_state(const Eigen::VectorXd &state_variable, const Eigen::VectorXi &indices) override;
	};

	// To optimize per element elastic parameters
	// state variable dof = 2 * n_elements
	class ElasticVariableToSimulation : public VariableToSimulation
	{
	public:
		using VariableToSimulation::VariableToSimulation;
		virtual ~ElasticVariableToSimulation() {}

		std::string name() const override { return "elastic"; }

		ParameterType get_parameter_type() const override { return ParameterType::Material; }

		Eigen::VectorXd compute_adjoint_term(const Eigen::VectorXd &x) const override;
		virtual Eigen::VectorXd inverse_eval() override;

	protected:
		void update_state(const Eigen::VectorXd &state_variable, const Eigen::VectorXi &indices) override;
	};

	// To optimize the friction constant
	// state variable dof = 1
	class FrictionCoeffientVariableToSimulation : public VariableToSimulation
	{
	public:
		using VariableToSimulation::VariableToSimulation;
		virtual ~FrictionCoeffientVariableToSimulation() {}

		std::string name() const override { return "friction"; }

		ParameterType get_parameter_type() const override { return ParameterType::FrictionCoeff; }

		Eigen::VectorXd compute_adjoint_term(const Eigen::VectorXd &x) const override;
		virtual Eigen::VectorXd inverse_eval() override;

	protected:
		void update_state(const Eigen::VectorXd &state_variable, const Eigen::VectorXi &indices) override;
	};

	// To optimize the damping constant psi and phi
	// state variable dof = 2
	class DampingCoeffientVariableToSimulation : public VariableToSimulation
	{
	public:
		using VariableToSimulation::VariableToSimulation;
		virtual ~DampingCoeffientVariableToSimulation() {}

		std::string name() const override { return "damping"; }

		ParameterType get_parameter_type() const override { return ParameterType::DampingCoeff; }

		Eigen::VectorXd compute_adjoint_term(const Eigen::VectorXd &x) const override;
		virtual Eigen::VectorXd inverse_eval() override;

	protected:
		void update_state(const Eigen::VectorXd &state_variable, const Eigen::VectorXi &indices) override;
	};

	// To optimize the per node initial condition
	// state variable dof = dim * n_bases
	class InitialConditionVariableToSimulation : public VariableToSimulation
	{
	public:
		using VariableToSimulation::VariableToSimulation;
		virtual ~InitialConditionVariableToSimulation() {}

		std::string name() const override { return "initial"; }

		ParameterType get_parameter_type() const override { return ParameterType::InitialCondition; }

		Eigen::VectorXd compute_adjoint_term(const Eigen::VectorXd &x) const override;
		virtual Eigen::VectorXd inverse_eval() override;

	protected:
		void update_state(const Eigen::VectorXd &state_variable, const Eigen::VectorXi &indices) override;
	};

	// To optimize the per node Dirichlet values, only work in transient simulations
	// state variable dof = dim * n_time_steps * n_dirichlet_nodes
	class DirichletVariableToSimulation : public VariableToSimulation
	{
	public:
		using VariableToSimulation::VariableToSimulation;
		virtual ~DirichletVariableToSimulation() {}

		std::string name() const override { return "dirichlet"; }

		ParameterType get_parameter_type() const override { return ParameterType::DirichletBC; }

		Eigen::VectorXd compute_adjoint_term(const Eigen::VectorXd &x) const override;
		virtual Eigen::VectorXd inverse_eval() override;

	protected:
		void update_state(const Eigen::VectorXd &state_variable, const Eigen::VectorXi &indices) override;

	private:
		std::string variable_to_string(const Eigen::VectorXd &variable);
	};

	// To optimize the per node pressure boundaries
	// Each pressure boundary will have the same value
	// state variable dof = dim * n_time_steps
	class PressureVariableToSimulation : public VariableToSimulation
	{
	public:
		PressureVariableToSimulation(
			const std::shared_ptr<State> &states,
			const CompositeParametrization &parametrization,
			const std::vector<int> pressure_boundaries)
			: VariableToSimulation(states, parametrization),
			  pressure_boundaries_(pressure_boundaries) {}
		PressureVariableToSimulation(
			const std::vector<std::shared_ptr<State>> &state,
			const CompositeParametrization &parametrization,
			const std::vector<int> pressure_boundaries)
			: VariableToSimulation(state, parametrization),
			  pressure_boundaries_(pressure_boundaries) {}
		virtual ~PressureVariableToSimulation() {}

		std::string name() const override { return "pressure"; }

		ParameterType get_parameter_type() const override { return ParameterType::PressureBC; }

		Eigen::VectorXd compute_adjoint_term(const Eigen::VectorXd &x) const override;
		virtual Eigen::VectorXd inverse_eval() override;

	protected:
		void update_state(const Eigen::VectorXd &state_variable, const Eigen::VectorXi &indices) override;

	private:
		std::string variable_to_string(const Eigen::VectorXd &variable);

		const std::vector<int> pressure_boundaries_;
	};
} // namespace polyfem::solver