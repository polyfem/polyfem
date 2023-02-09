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
			Eigen::MatrixXd V_rest, V;
			Eigen::MatrixXi F;
			state_ptr_->get_vf(V_rest, F);
			// TODO: Insert nodes here
			V = V_rest;
			mesh_flipped = is_flipped(V, F);
			if (mesh_flipped)
			{
				if (V.rows() == 2)
				{
					V.conservativeResize(V.rows(), 3);
					V.col(0) = Eigen::VectorXd::Zero(V.rows());
				}
				igl::writeOBJ("flipped.obj", V, F);

				log_and_throw_error("Mesh Flipped!")
			}

			state_ptr->set_mesh_vertices(V);
			state_ptr->build_basis();
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
			auto constraint_string = constraint_to_string(state_variable);
			for (const auto &kv : boundary_id_to_reduced_param)
			{
				json dirichlet_bc = constraint_string[kv.first];
				// Need time_steps + 1 entry, though unused.
				for (int k = 0; k < states_ptr_[0]->mesh->dimension(); ++k)
					dirichlet_bc[k].push_back(dirichlet_bc[k][time_steps - 1]);
				logger().trace("Updating boundary id {} to dirichlet bc {}", kv.first, dirichlet_bc);
				problem.update_dirichlet_boundary(kv.first, dirichlet_bc, true, true, true, "");
			}
		}
	}

} // namespace polyfem::solver