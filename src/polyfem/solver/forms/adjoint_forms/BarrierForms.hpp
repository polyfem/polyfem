#pragma once

#include <polyfem/solver/forms/ParametrizationForm.hpp>
#include "VariableToSimulation.hpp"

#include <polyfem/solver/forms/ContactForm.hpp>

namespace polyfem::solver
{
	class CollisionBarrierForm : public ParametrizationForm
	{
	public:
		CollisionBarrierForm(const std::vector<std::shared_ptr<VariableToSimulation>> &variable_to_simulations, const CompositeParametrization &parametrizations, const State &state, const double dhat) : ParametrizationForm(variable_to_simulations, parametrizations), state_(state), dhat_(dhat)
		{
			state.build_collision_mesh(collision_mesh_, state.n_geom_bases, state.geom_bases());

			broad_phase_method = ipc::BroadPhaseMethod::HASH_GRID;
		}

		double value_unweighted_with_param(const Eigen::VectorXd &x) const override
		{
			assert(x.size() == state_.mesh->n_elements());

			const Eigen::MatrixXd displaced_surface = collision_mesh_.vertices(utils::unflatten(x, state_.mesh->dimension()));

			return ipc::compute_barrier_potential(collision_mesh_, displaced_surface, constraint_set, dhat_);
		}

		void first_derivative_unweighted_with_param(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const override
		{
			assert(x.size() == state_.mesh->n_elements());

			const Eigen::MatrixXd displaced_surface = collision_mesh_.vertices(utils::unflatten(x, state_.mesh->dimension()));

			gradv = collision_mesh_.to_full_dof(ipc::compute_barrier_potential_gradient(collision_mesh_, displaced_surface, constraint_set, dhat_));
		}

		void solution_changed_with_param(const Eigen::VectorXd &x) override
		{
			const Eigen::MatrixXd displaced_surface = collision_mesh_.vertices(utils::unflatten(x, state_.mesh->dimension()));
			build_constraint_set(displaced_surface);
		}

	private:
		void build_constraint_set(const Eigen::MatrixXd &displaced_surface)
		{
			static Eigen::MatrixXd cached_displaced_surface;
			if (cached_displaced_surface.size() == displaced_surface.size() && cached_displaced_surface == displaced_surface)
				return;

			constraint_set.build(collision_mesh_, displaced_surface, dhat_, 0, broad_phase_method);

			cached_displaced_surface = displaced_surface;
		}

		const State &state_;

		ipc::CollisionMesh collision_mesh_;
		ipc::Constraints constraint_set;
		const double dhat_;
		ipc::BroadPhaseMethod broad_phase_method;
	};

	// class LayerThicknessForm : public ParametrizationForm
	// {
	// public:
	// 	LayerThicknessForm(const std::vector<std::shared_ptr<VariableToSimulation>> &variable_to_simulations, const CompositeParametrization &parametrizations, const State &state) : ParametrizationForm(variable_to_simulations, parametrizations), state_(state)
	// 	{
	// 	}
	// }
} // namespace polyfem::solver