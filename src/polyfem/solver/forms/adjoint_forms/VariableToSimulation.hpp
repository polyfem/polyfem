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

		ParameterType get_parameter_type() const override { return ParameterType::Shape; }

		Eigen::VectorXd compute_adjoint_term(const Eigen::VectorXd &x) const override;
		virtual Eigen::VectorXd inverse_eval() override;

	protected:
		virtual void update_state(const Eigen::VectorXd &state_variable, const Eigen::VectorXi &indices) override;
	};

	// Only for periodic mesh
	// state variable dof = dim * n_periodic_vertices
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
	// state variable dof = dim * n_vertices
	class SDFShapeVariableToSimulation : public ShapeVariableToSimulation
	{
	public:
		SDFShapeVariableToSimulation(const std::vector<std::shared_ptr<State>> &states, const CompositeParametrization &parametrization, const json &args);
		virtual ~SDFShapeVariableToSimulation() {}

		void update(const Eigen::VectorXd &x) override;

	protected:
		const int mesh_id_;
		const std::string mesh_path_;
	};

	// Combination of SDFShapeVariableToSimulation and PeriodicShapeVariableToSimulation, to optimize the shape of a SDF.
	// Assuming the mesh of this SDF is periodic and the periodic surface should be the intersection of the domain with its bounding box faces
	// state variable dof = dim * n_periodic_vertices
	class SDFPeriodicShapeVariableToSimulation : public PeriodicShapeVariableToSimulation
	{
	public:
		SDFPeriodicShapeVariableToSimulation(const std::vector<std::shared_ptr<State>> &states, const CompositeParametrization &parametrization, const json &args);
		virtual ~SDFPeriodicShapeVariableToSimulation() {}

		void update(const Eigen::VectorXd &x) override;
		Eigen::VectorXd apply_parametrization_jacobian(const Eigen::VectorXd &term, const Eigen::VectorXd &x) const override;

	protected:
		const std::string mesh_path_;
	};

	// To optimize the scaling of a periodic mesh, can be used together with SDFPeriodicShapeVariableToSimulation
	// state variable dof = dim
	class PeriodicShapeScaleVariableToSimulation : public PeriodicShapeVariableToSimulation
	{
	public:
		PeriodicShapeScaleVariableToSimulation(const std::vector<std::shared_ptr<State>> &states, const CompositeParametrization &parametrization, const json &args);
		virtual ~PeriodicShapeScaleVariableToSimulation() {}

		void update(const Eigen::VectorXd &x) override;
		Eigen::VectorXd apply_parametrization_jacobian(const Eigen::VectorXd &term, const Eigen::VectorXd &x) const override;
	
	private:
		int dim;
	};

	// To optimize per element elastic parameters
	// state variable dof = 2 * n_elements
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

	// To optimize the friction constant
	// state variable dof = 1
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

	// To optimize the damping constant psi and phi
	// state variable dof = 2
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

	// To optimize the per node initial condition
	// state variable dof = dim * n_bases
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

	// To optimize the per node Dirichlet values, only work in transient simulations
	// state variable dof = dim * n_time_steps * n_dirichlet_nodes
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

	// To optimize the underlying linear displacement of a homogenization solve
	// state variable dof = dim * dim
	class MacroStrainVariableToSimulation : public VariableToSimulation
	{
	public:
		using VariableToSimulation::VariableToSimulation;
		virtual ~MacroStrainVariableToSimulation() {}

		ParameterType get_parameter_type() const override { return ParameterType::MacroStrain; }

		Eigen::VectorXd compute_adjoint_term(const Eigen::VectorXd &x) const override;
		virtual Eigen::VectorXd inverse_eval() override;

	protected:
		void update_state(const Eigen::VectorXd &state_variable, const Eigen::VectorXi &indices) override;
	};

} // namespace polyfem::solver