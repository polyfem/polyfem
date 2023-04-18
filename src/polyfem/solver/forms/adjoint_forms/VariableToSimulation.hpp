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

		inline int n_states() const { return states_.size(); }
		inline const std::vector<std::shared_ptr<State>> &get_states() const { return states_; }

		inline CompositeParametrization &get_parametrization() { return parametrization_; }
		virtual ParameterType get_parameter_type() const = 0;
		virtual Eigen::VectorXd compute_adjoint_term(const Eigen::VectorXd &x) const = 0;
		virtual Eigen::VectorXd inverse_eval() = 0;

		void set_output_indexing(const Eigen::VectorXi &output_indexing) { output_indexing_ = output_indexing; }
		Eigen::VectorXi get_output_indexing(const Eigen::VectorXd &x) const;

		virtual Eigen::VectorXd apply_parametrization_jacobian(const Eigen::VectorXd &term, const Eigen::VectorXd &x) const;

	protected:
		virtual void update_state(const Eigen::VectorXd &state_variable, const Eigen::VectorXi &indices);
		std::vector<std::shared_ptr<State>> states_;
		CompositeParametrization parametrization_;

		Eigen::VectorXi output_indexing_;
	};

	class ShapeVariableToSimulation : public VariableToSimulation
	{
	public:
		using VariableToSimulation::VariableToSimulation;
		virtual ~ShapeVariableToSimulation() {}

		ParameterType get_parameter_type() const override { return ParameterType::Shape; }

		Eigen::VectorXd compute_adjoint_term(const Eigen::VectorXd &x) const override;
		virtual Eigen::VectorXd inverse_eval() override;

	protected:
		virtual void update_state(const Eigen::VectorXd &state_variable, const Eigen::VectorXi &indices) override;
	};

	// Only for periodic mesh
	class PeriodicShapeVariableToSimulation : public VariableToSimulation
	{
	public:
		using VariableToSimulation::VariableToSimulation;
		virtual ~PeriodicShapeVariableToSimulation() {}

		ParameterType get_parameter_type() const override { return ParameterType::PeriodicShape; }

		void update(const Eigen::VectorXd &x) override;
		Eigen::VectorXd compute_adjoint_term(const Eigen::VectorXd &x) const override;
		Eigen::VectorXd inverse_eval() override;
	};

	// For optimizing the shape of a parametrized SDF. The mesh connectivity may change when SDF changes, so a new mesh is loaded whenever the optimization variable changes.
	class SDFShapeVariableToSimulation : public ShapeVariableToSimulation
	{
	public:
		SDFShapeVariableToSimulation(const std::vector<std::shared_ptr<State>> &states, const CompositeParametrization &parametrization, const json &args);
		virtual ~SDFShapeVariableToSimulation() {}

		void update(const Eigen::VectorXd &x) override;
		Eigen::VectorXd inverse_eval() override;

	protected:
		const int mesh_id_;
		const std::string mesh_path_;
	};

	// Combination of SDFShapeVariableToSimulation and PeriodicShapeVariableToSimulation, to optimize the shape of a SDF, assuming the mesh of this SDF is periodic
	class SDFPeriodicShapeVariableToSimulation : public PeriodicShapeVariableToSimulation
	{
	public:
		SDFPeriodicShapeVariableToSimulation(const std::vector<std::shared_ptr<State>> &states, const CompositeParametrization &parametrization, const json &args);
		virtual ~SDFPeriodicShapeVariableToSimulation() {}

		void update(const Eigen::VectorXd &x) override;
		Eigen::VectorXd inverse_eval() override;
		Eigen::VectorXd apply_parametrization_jacobian(const Eigen::VectorXd &term, const Eigen::VectorXd &x) const override;

	protected:
		const std::string mesh_path_;
	};

	class ElasticVariableToSimulation : public VariableToSimulation
	{
	public:
		using VariableToSimulation::VariableToSimulation;
		virtual ~ElasticVariableToSimulation() {}

		ParameterType get_parameter_type() const override { return ParameterType::Material; }

		Eigen::VectorXd compute_adjoint_term(const Eigen::VectorXd &x) const override;
		virtual Eigen::VectorXd inverse_eval() override;

	protected:
		void update_state(const Eigen::VectorXd &state_variable, const Eigen::VectorXi &indices) override;
	};

	class FrictionCoeffientVariableToSimulation : public VariableToSimulation
	{
	public:
		using VariableToSimulation::VariableToSimulation;
		virtual ~FrictionCoeffientVariableToSimulation() {}

		ParameterType get_parameter_type() const override { return ParameterType::FrictionCoeff; }

		Eigen::VectorXd compute_adjoint_term(const Eigen::VectorXd &x) const override;
		virtual Eigen::VectorXd inverse_eval() override;

	protected:
		void update_state(const Eigen::VectorXd &state_variable, const Eigen::VectorXi &indices) override;
	};

	class DampingCoeffientVariableToSimulation : public VariableToSimulation
	{
	public:
		using VariableToSimulation::VariableToSimulation;
		virtual ~DampingCoeffientVariableToSimulation() {}

		ParameterType get_parameter_type() const override { return ParameterType::DampingCoeff; }

		Eigen::VectorXd compute_adjoint_term(const Eigen::VectorXd &x) const override;
		virtual Eigen::VectorXd inverse_eval() override;

	protected:
		void update_state(const Eigen::VectorXd &state_variable, const Eigen::VectorXi &indices) override;
	};

	class InitialConditionVariableToSimulation : public VariableToSimulation
	{
	public:
		using VariableToSimulation::VariableToSimulation;
		virtual ~InitialConditionVariableToSimulation() {}

		ParameterType get_parameter_type() const override { return ParameterType::InitialCondition; }

		Eigen::VectorXd compute_adjoint_term(const Eigen::VectorXd &x) const override;
		virtual Eigen::VectorXd inverse_eval() override;

	protected:
		void update_state(const Eigen::VectorXd &state_variable, const Eigen::VectorXi &indices) override;
	};

	class DirichletVariableToSimulation : public VariableToSimulation
	{
	public:
		using VariableToSimulation::VariableToSimulation;
		virtual ~DirichletVariableToSimulation() {}

		ParameterType get_parameter_type() const override { return ParameterType::DirichletBC; }

		Eigen::VectorXd compute_adjoint_term(const Eigen::VectorXd &x) const override;
		virtual Eigen::VectorXd inverse_eval() override;

	protected:
		void update_state(const Eigen::VectorXd &state_variable, const Eigen::VectorXi &indices) override;

	private:
		std::string variable_to_string(const Eigen::VectorXd &variable);
	};

	class MacroStrainVariableToSimulation : public VariableToSimulation
	{
	public:
		using VariableToSimulation::VariableToSimulation;
		virtual ~MacroStrainVariableToSimulation() {}

		ParameterType get_parameter_type() const override { return ParameterType::MacroStrain; }

		Eigen::VectorXd compute_adjoint_term(const Eigen::VectorXd &x) const override;
		virtual Eigen::VectorXd inverse_eval() override;

	protected:
		void update_state(const Eigen::VectorXd &state_variable, const Eigen::VectorXi &indices);
	};

} // namespace polyfem::solver