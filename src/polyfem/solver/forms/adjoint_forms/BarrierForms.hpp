#pragma once

#include <polyfem/solver/forms/adjoint_forms/ParametrizationForm.hpp>
#include "VariableToSimulation.hpp"

#include <polyfem/solver/forms/ContactForm.hpp>

namespace polyfem::solver
{
	class CollisionBarrierForm : public AdjointForm
	{
	public:
		CollisionBarrierForm(const std::vector<std::shared_ptr<VariableToSimulation>> variable_to_simulation, const State &state, const double dhat) : AdjointForm(variable_to_simulation), state_(state), dhat_(dhat)
		{
			State::build_collision_mesh(
				*state_.mesh, state_.n_geom_bases, state_.geom_bases(), state_.geom_bases(),
				state_.total_local_boundary, state_.obstacle, state_.args,
				[this](const std::string &p) { return this->state_.resolve_input_path(p); },
				state_.in_node_to_node, collision_mesh_);

			Eigen::MatrixXd V;
			state_.get_vertices(V);
			X_init = utils::flatten(V);

			broad_phase_method_ = ipc::BroadPhaseMethod::HASH_GRID;
		}

		double value_unweighted(const Eigen::VectorXd &x) const override
		{
			const Eigen::MatrixXd displaced_surface = collision_mesh_.vertices(utils::unflatten(get_updated_mesh_nodes(x), state_.mesh->dimension()));

			return constraint_set.compute_potential(collision_mesh_, displaced_surface, dhat_);
		}

		void compute_partial_gradient_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const override
		{
			const Eigen::MatrixXd displaced_surface = collision_mesh_.vertices(utils::unflatten(get_updated_mesh_nodes(x), state_.mesh->dimension()));

			Eigen::VectorXd grad = collision_mesh_.to_full_dof(constraint_set.compute_potential_gradient(collision_mesh_, displaced_surface, dhat_));

			gradv.setZero(x.size());
			for (auto &p : variable_to_simulations_)
			{
				for (const auto &state : p->get_states())
					if (state.get() != &state_)
						continue;
				if (p->get_parameter_type() != ParameterType::Shape)
					continue;
				gradv += p->apply_parametrization_jacobian(grad, x);
			}
		}

		void solution_changed(const Eigen::VectorXd &x) override
		{
			AdjointForm::solution_changed(x);

			const Eigen::MatrixXd displaced_surface = collision_mesh_.vertices(utils::unflatten(get_updated_mesh_nodes(x), state_.mesh->dimension()));
			build_constraint_set(displaced_surface);
		}

		Eigen::MatrixXd compute_adjoint_rhs_unweighted(const Eigen::VectorXd &x, const State &state) const override
		{
			return Eigen::MatrixXd::Zero(state.ndof(), state.diff_cached.size());
		}

		bool is_step_collision_free(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const override
		{
			const Eigen::MatrixXd V0 = utils::unflatten(get_updated_mesh_nodes(x0), state_.mesh->dimension());
			const Eigen::MatrixXd V1 = utils::unflatten(get_updated_mesh_nodes(x1), state_.mesh->dimension());

			// Skip CCD if the displacement is zero.
			if ((V1 - V0).lpNorm<Eigen::Infinity>() == 0.0)
				return true;

			bool is_valid = ipc::is_step_collision_free(
				collision_mesh_,
				collision_mesh_.vertices(V0),
				collision_mesh_.vertices(V1),
				broad_phase_method_,
				1e-6, 1e6);

			return is_valid;
		}

		double max_step_size(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const override
		{
			const Eigen::MatrixXd V0 = utils::unflatten(get_updated_mesh_nodes(x0), state_.mesh->dimension());
			const Eigen::MatrixXd V1 = utils::unflatten(get_updated_mesh_nodes(x1), state_.mesh->dimension());

			double max_step = ipc::compute_collision_free_stepsize(
				collision_mesh_,
				collision_mesh_.vertices(V0),
				collision_mesh_.vertices(V1),
				broad_phase_method_, 1e-6, 1e6);

			return max_step;
		}

	private:
		void build_constraint_set(const Eigen::MatrixXd &displaced_surface)
		{
			static Eigen::MatrixXd cached_displaced_surface;
			if (cached_displaced_surface.size() == displaced_surface.size() && cached_displaced_surface == displaced_surface)
				return;

			constraint_set.build(collision_mesh_, displaced_surface, dhat_, 0, broad_phase_method_);

			cached_displaced_surface = displaced_surface;
		}

		Eigen::VectorXd get_updated_mesh_nodes(const Eigen::VectorXd &x) const
		{
			Eigen::VectorXd X = X_init;

			for (auto &p : variable_to_simulations_)
			{
				for (const auto &state : p->get_states())
					if (state.get() != &state_)
						continue;
				if (p->get_parameter_type() != ParameterType::Shape)
					continue;
				auto state_variable = p->get_parametrization().eval(x);
				auto output_indexing = p->get_output_indexing(x);
				for (int i = 0; i < output_indexing.size(); ++i)
					X(output_indexing(i)) = state_variable(i);
			}

			return AdjointTools::map_primitive_to_node_order(state_, X);
		}

		const State &state_;

		Eigen::VectorXd X_init;

		ipc::CollisionMesh collision_mesh_;
		ipc::CollisionConstraints constraint_set;
		const double dhat_;
		ipc::BroadPhaseMethod broad_phase_method_;
	};

	// class LayerThicknessForm : public ParametrizationForm
	// {
	// public:
	// 	LayerThicknessForm(const std::vector<std::shared_ptr<VariableToSimulation>> &variable_to_simulations, const CompositeParametrization &parametrizations, const State &state) : ParametrizationForm(variable_to_simulations, parametrizations), state_(state)
	// 	{
	// 	}
	// }
} // namespace polyfem::solver