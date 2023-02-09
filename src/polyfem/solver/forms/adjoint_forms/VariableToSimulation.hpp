#pragma once

#include <polyfem/solver/forms/parameterization/Parameterization.hpp>
#include <polyfem/solver/AdjointTools.hpp>

namespace polyfem::solver
{
	class VariableToSimulation
	{
	public:
		VariableToSimulation(std::shared_ptr<State> state_ptr, const CompositeParameterization &parameterization) : state_ptr_(state_ptr), parameterization_(parameterization) {}
		virtual ~VariableToSimulation() {}

		inline virtual void update(const Eigen::VectorXd &x) final
		{
			Eigen::VectorXd state_variable = parameterization_.eval(x);
			update_state(state_variable);
		}

		inline const State &get_state() const { return *state_ptr_; }
		inline CompositeParameterization get_parameterization() const { return parameterization_; }
		virtual ParameterType get_parameter_type() const = 0;

	protected:
		virtual void update_state(const Eigen::VectorXd &state_variable) = 0;
		std::shared_ptr<State> state_ptr_;
		CompositeParameterization parameterization_;
	};

	class ShapeVariableToSimulation : public VariableToSimulation
	{
	public:
		virtual ~ShapeVariableToSimulation() {}
		
		ParameterType get_parameter_type() const override { return ParameterType::Shape; }
	protected:
		inline void update_state(const Eigen::VectorXd &state_variable) override
		{
			Eigen::MatrixXd V_rest, V;
			Eigen::MatrixXi F;
			state_ptr_->get_vf(V_rest, F);
			// TODO: Insert nodes here
			V = utils::unflatten(state_variable, V_rest.cols());

			state_ptr_->set_mesh_vertices(V);
			state_ptr_->build_basis();
		}
	};

	class ElasticVariableToSimulation : public VariableToSimulation
	{
	public:
		virtual ~ElasticVariableToSimulation() {}

		ParameterType get_parameter_type() const override { return ParameterType::Material; }
	protected:
		inline void update_state(const Eigen::VectorXd &state_variable) override
		{
			const int n_elem = state_ptr_->bases.size();
			state_ptr_->assembler.update_lame_params(state_variable.segment(0, n_elem), state_variable.segment(n_elem, n_elem));
		}
	};

	class FrictionCoeffientVariableToSimulation : public VariableToSimulation
	{
	public:
		virtual ~FrictionCoeffientVariableToSimulation() {}

		ParameterType get_parameter_type() const override { return ParameterType::FrictionCoeff; }
	protected:
		inline void update_state(const Eigen::VectorXd &state_variable) override
		{
			state_ptr_->args["contact"]["friction_coefficient"] = state_variable(0);
		}
	};

	class DampingCoeffientVariableToSimulation : public VariableToSimulation
	{
	public:
		virtual ~DampingCoeffientVariableToSimulation() {}

		ParameterType get_parameter_type() const override { return ParameterType::DampingCoeff; }
	protected:
		inline void update_state(const Eigen::VectorXd &state_variable) override
		{
			json damping_param = {
				{"psi", state_variable(0)},
				{"phi", state_variable(1)},
			};
			state_ptr_->assembler.add_multimaterial(0, damping_param);
			logger().info("Current damping params: {}, {}", state_variable(0), state_variable(1));
		}
	};

	class InitialConditionVariableToSimulation : public VariableToSimulation
	{
	public:
		virtual ~InitialConditionVariableToSimulation() {}

		ParameterType get_parameter_type() const override { return ParameterType::InitialCondition; }
	protected:
		inline void update_state(const Eigen::VectorXd &state_variable) override
		{
		}
	};

	class DirichletVariableToSimulation : public VariableToSimulation
	{
	public:
		virtual ~DirichletVariableToSimulation() {}

		ParameterType get_parameter_type() const override { return ParameterType::DirichletBC; }
	protected:
		inline void update_state(const Eigen::VectorXd &state_variable) override
		{
			// auto constraint_string = constraint_to_string(state_variable);
			// for (const auto &kv : boundary_id_to_reduced_param)
			// {
			// 	json dirichlet_bc = constraint_string[kv.first];
			// 	// Need time_steps + 1 entry, though unused.
			// 	for (int k = 0; k < states_ptr_[0]->mesh->dimension(); ++k)
			// 		dirichlet_bc[k].push_back(dirichlet_bc[k][time_steps - 1]);
			// 	logger().trace("Updating boundary id {} to dirichlet bc {}", kv.first, dirichlet_bc);
			// 	problem.update_dirichlet_boundary(kv.first, dirichlet_bc, true, true, true, "");
			// }
		}
	};

	class MacroStrainVariableToSimulation : public VariableToSimulation
	{
	public:
		virtual ~MacroStrainVariableToSimulation() {}

		ParameterType get_parameter_type() const override { return ParameterType::MacroStrain; }
	protected:
		inline void update_state(const Eigen::VectorXd &state_variable) override
		{
			state_ptr_->disp_grad = utils::unflatten(state_variable, state_ptr_->mesh->dimension());
		}
	};

} // namespace polyfem::solver