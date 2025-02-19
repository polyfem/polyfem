#pragma once

#include <polyfem/solver/forms/parametrization/Parametrization.hpp>
#include <polyfem/solver/AdjointTools.hpp>
#include <polyfem/solver/forms/parametrization/PeriodicMeshToMesh.hpp>

#include <iostream>

namespace polyfem::solver
{
	/// @brief Maps the optimization variable to the state variable
	class VariableToSimulation
	{
	public:
		VariableToSimulation(const std::vector<std::shared_ptr<State>> &states, const CompositeParametrization &parametrization) : states_(states), parametrization_(parametrization) { assert(states.size() > 0); }
		VariableToSimulation(const std::shared_ptr<State> &state, const CompositeParametrization &parametrization) : states_({state}), parametrization_(parametrization) {}
		virtual ~VariableToSimulation() {}

		static std::unique_ptr<VariableToSimulation> create(const std::string &type, const std::vector<std::shared_ptr<State>> &states, CompositeParametrization &&parametrization);

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

		virtual void set_output_indexing(const json &args);
		Eigen::VectorXi get_output_indexing(const Eigen::VectorXd &x) const;

		virtual Eigen::VectorXd apply_parametrization_jacobian(const Eigen::VectorXd &term, const Eigen::VectorXd &x) const;

	protected:
		virtual void update_state(const Eigen::VectorXd &state_variable, const Eigen::VectorXi &indices);
		const std::vector<std::shared_ptr<State>> states_;
		CompositeParametrization parametrization_;

		Eigen::VectorXi output_indexing_; // if a derived class overrides apply_parametrization_jacobian(term, x), this is not necessarily used.
	};

	/// @brief A collection of VariableToSimulation
	class VariableToSimulationGroup
	{
	public:
		using ValueType = std::shared_ptr<VariableToSimulation>;

		VariableToSimulationGroup() = default;
		virtual ~VariableToSimulationGroup() = default;

		void init(const json &args, const std::vector<std::shared_ptr<State>> &states, const std::vector<int> &variable_sizes);

		/// @brief Update parameters in simulators
		/// @param x Optimization variable
		inline void update(const Eigen::VectorXd &x)
		{
			for (auto &v2s : L)
				v2s->update(x);
		}

		/// @brief Evaluate the variable to simulations and overwrite the state_variable based on x
		/// @param x Optimization variable
		/// @param state_variable The state variable in state_ptr with type
		void compute_state_variable(const ParameterType type, const State *state_ptr, const Eigen::VectorXd &x, Eigen::VectorXd &state_variable) const;

		/// @brief Computes the sum of adjoint terms for all VariableToSimulation
		/// @param x Optimization variable
		/// @return Sum of adjoint terms
		Eigen::VectorXd compute_adjoint_term(const Eigen::VectorXd &x) const;

		/// @brief Maps the partial gradient wrt. the state variable to the partial gradient wrt. the optimization variable
		/// @param type Type of state variable
		/// @param state_ptr The state that stores the state variable
		/// @param x Optimization variable
		/// @param grad Partial gradient wrt. the state variable, lambda function to allow lazy evaluation of the gradient
		/// @return Partial gradient wrt. the optimization variable
		virtual Eigen::VectorXd apply_parametrization_jacobian(const ParameterType type, const State *state_ptr, const Eigen::VectorXd &x, const std::function<Eigen::VectorXd()> &grad) const;

		typedef std::vector<ValueType>::const_iterator const_iterator;

		inline ValueType& operator[](size_t i) { return L[i]; }
		inline const ValueType& operator[](size_t i) const { return L[i]; }
		inline const_iterator begin() const { return L.begin(); }
		inline const_iterator end() const { return L.end(); }
		inline void push_back(const ValueType &v2s) { L.push_back(v2s); }
		inline void clear() { L.clear(); }

	private:
		std::vector<ValueType> L;
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
		Eigen::VectorXd inverse_eval() override;

		void set_output_indexing(const json &args) override;

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

		ParameterType get_parameter_type() const override { return ParameterType::LameParameter; }

		Eigen::VectorXd compute_adjoint_term(const Eigen::VectorXd &x) const override;
		Eigen::VectorXd inverse_eval() override;

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

		ParameterType get_parameter_type() const override { return ParameterType::FrictionCoefficient; }

		Eigen::VectorXd compute_adjoint_term(const Eigen::VectorXd &x) const override;
		Eigen::VectorXd inverse_eval() override;

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

		ParameterType get_parameter_type() const override { return ParameterType::DampingCoefficient; }

		Eigen::VectorXd compute_adjoint_term(const Eigen::VectorXd &x) const override;
		Eigen::VectorXd inverse_eval() override;

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
		Eigen::VectorXd inverse_eval() override;

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

		void set_dirichlet_boundaries(const std::vector<int> &dirichlet_boundaries)
		{
			dirichlet_boundaries_ = dirichlet_boundaries;
		}

		ParameterType get_parameter_type() const override { return ParameterType::DirichletBC; }

		Eigen::VectorXd compute_adjoint_term(const Eigen::VectorXd &x) const override;
		Eigen::VectorXd inverse_eval() override;

		void set_output_indexing(const json &args) override;

	protected:
		void update_state(const Eigen::VectorXd &state_variable, const Eigen::VectorXi &indices) override;

	private:
		std::string variable_to_string(const Eigen::VectorXd &variable);

		std::vector<int> dirichlet_boundaries_;
	};

	// To optimize the per node pressure boundaries
	// Each pressure boundary will have the same value
	// state variable dof = dim * n_time_steps
	class PressureVariableToSimulation : public VariableToSimulation
	{
	public:
		using VariableToSimulation::VariableToSimulation;
		virtual ~PressureVariableToSimulation() {}

		std::string name() const override { return "pressure"; }

		void set_pressure_boundaries(const std::vector<int> &pressure_boundaries)
		{
			pressure_boundaries_ = pressure_boundaries;
		}

		ParameterType get_parameter_type() const override { return ParameterType::PressureBC; }

		Eigen::VectorXd compute_adjoint_term(const Eigen::VectorXd &x) const override;
		Eigen::VectorXd inverse_eval() override;

		void set_output_indexing(const json &args) override;

	protected:
		void update_state(const Eigen::VectorXd &state_variable, const Eigen::VectorXi &indices) override;

	private:
		std::string variable_to_string(const Eigen::VectorXd &variable);

		std::vector<int> pressure_boundaries_;
	};

	// state variable dof = dim * n_vertices - periodic dof
	class PeriodicShapeVariableToSimulation : public VariableToSimulation
	{
	public:
		using VariableToSimulation::VariableToSimulation;
		virtual ~PeriodicShapeVariableToSimulation() {}

		std::string name() const override { return "periodic-shape"; }

		ParameterType get_parameter_type() const override { return ParameterType::PeriodicShape; }

		Eigen::VectorXd compute_adjoint_term(const Eigen::VectorXd &x) const override;
		Eigen::VectorXd inverse_eval() override;

		void update(const Eigen::VectorXd &x) override;

		Eigen::VectorXd apply_parametrization_jacobian(const Eigen::VectorXd &term, const Eigen::VectorXd &x) const override;

	protected:
		std::unique_ptr<PeriodicMeshToMesh> periodic_mesh_map;
		Eigen::VectorXd periodic_mesh_representation;
	};
} // namespace polyfem::solver