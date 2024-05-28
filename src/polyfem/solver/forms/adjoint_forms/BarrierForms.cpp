#include "BarrierForms.hpp"
#include <polyfem/State.hpp>

namespace polyfem::solver
{
	CollisionBarrierForm::CollisionBarrierForm(const VariableToSimulationGroup& variable_to_simulation, const State &state, const double dhat)
		: AdjointForm(variable_to_simulation), state_(state), dhat_(dhat), barrier_potential_(dhat)
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

	double CollisionBarrierForm::value_unweighted(const Eigen::VectorXd &x) const
	{
		const Eigen::MatrixXd displaced_surface = collision_mesh_.vertices(utils::unflatten(get_updated_mesh_nodes(x), state_.mesh->dimension()));
		return barrier_potential_(collision_set, collision_mesh_, displaced_surface);
	}

	void CollisionBarrierForm::compute_partial_gradient(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		
		gradv = weight() * variable_to_simulations_.apply_parametrization_jacobian(ParameterType::Shape, &state_, x, [this, &x]() {
			const Eigen::MatrixXd displaced_surface = collision_mesh_.vertices(utils::unflatten(get_updated_mesh_nodes(x), state_.mesh->dimension()));
			const Eigen::VectorXd grad = collision_mesh_.to_full_dof(barrier_potential_.gradient(collision_set, collision_mesh_, displaced_surface));
			return AdjointTools::map_node_to_primitive_order(state_, grad);
		});
	}

	void CollisionBarrierForm::solution_changed(const Eigen::VectorXd &x)
	{
		AdjointForm::solution_changed(x);

		const Eigen::MatrixXd displaced_surface = collision_mesh_.vertices(utils::unflatten(get_updated_mesh_nodes(x), state_.mesh->dimension()));
		build_collision_set(displaced_surface);
	}

	bool CollisionBarrierForm::is_step_collision_free(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const
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

	double CollisionBarrierForm::max_step_size(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const
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

	void CollisionBarrierForm::build_collision_set(const Eigen::MatrixXd &displaced_surface)
	{
		static Eigen::MatrixXd cached_displaced_surface;
		if (cached_displaced_surface.size() == displaced_surface.size() && cached_displaced_surface == displaced_surface)
			return;

		collision_set.build(collision_mesh_, displaced_surface, dhat_, 0, broad_phase_method_);

		cached_displaced_surface = displaced_surface;
	}

	Eigen::VectorXd CollisionBarrierForm::get_updated_mesh_nodes(const Eigen::VectorXd &x) const
	{
		Eigen::VectorXd X = X_init;
		variable_to_simulations_.compute_state_variable(ParameterType::Shape, &state_, x, X);
		return AdjointTools::map_primitive_to_node_order(state_, X);
	}

	DeformedCollisionBarrierForm::DeformedCollisionBarrierForm(const VariableToSimulationGroup& variable_to_simulation, const State &state, const double dhat)
		: AdjointForm(variable_to_simulation), state_(state), dhat_(dhat), barrier_potential_(dhat)
	{
		if (state_.n_bases != state_.n_geom_bases)
			log_and_throw_adjoint_error("[{}] Should use linear FE basis!", name());

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

	double DeformedCollisionBarrierForm::value_unweighted(const Eigen::VectorXd &x) const
	{
		const Eigen::MatrixXd displaced_surface = collision_mesh_.vertices(utils::unflatten(get_updated_mesh_nodes(x), state_.mesh->dimension()));

		return barrier_potential_(collision_set, collision_mesh_, displaced_surface);
	}

	void DeformedCollisionBarrierForm::compute_partial_gradient(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		gradv = weight() * variable_to_simulations_.apply_parametrization_jacobian(ParameterType::Shape, &state_, x, [this, &x]() {
			const Eigen::MatrixXd displaced_surface = collision_mesh_.vertices(utils::unflatten(get_updated_mesh_nodes(x), state_.mesh->dimension()));
			const Eigen::VectorXd grad = collision_mesh_.to_full_dof(barrier_potential_.gradient(collision_set, collision_mesh_, displaced_surface));
			return AdjointTools::map_node_to_primitive_order(state_, grad);
		});
	}

	void DeformedCollisionBarrierForm::solution_changed(const Eigen::VectorXd &x)
	{
		AdjointForm::solution_changed(x);

		const Eigen::MatrixXd displaced_surface = collision_mesh_.vertices(utils::unflatten(get_updated_mesh_nodes(x), state_.mesh->dimension()));
		build_collision_set(displaced_surface);
	}

	bool DeformedCollisionBarrierForm::is_step_collision_free(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const
	{
		// const Eigen::MatrixXd V0 = utils::unflatten(get_updated_mesh_nodes(x0), state_.mesh->dimension());
		// const Eigen::MatrixXd V1 = utils::unflatten(get_updated_mesh_nodes(x1), state_.mesh->dimension());

		// // Skip CCD if the displacement is zero.
		// if ((V1 - V0).lpNorm<Eigen::Infinity>() == 0.0)
		//     return true;

		// bool is_valid = ipc::is_step_collision_free(
		//     collision_mesh_,
		//     collision_mesh_.vertices(V0),
		//     collision_mesh_.vertices(V1),
		//     broad_phase_method_,
		//     1e-6, 1e6);

		return true; // is_valid;
	}

	double DeformedCollisionBarrierForm::max_step_size(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const
	{
		// const Eigen::MatrixXd V0 = utils::unflatten(get_updated_mesh_nodes(x0), state_.mesh->dimension());
		// const Eigen::MatrixXd V1 = utils::unflatten(get_updated_mesh_nodes(x1), state_.mesh->dimension());

		// double max_step = ipc::compute_collision_free_stepsize(
		//     collision_mesh_,
		//     collision_mesh_.vertices(V0),
		//     collision_mesh_.vertices(V1),
		//     broad_phase_method_, 1e-6, 1e6);

		return 1; // max_step;
	}

	void DeformedCollisionBarrierForm::build_collision_set(const Eigen::MatrixXd &displaced_surface)
	{
		static Eigen::MatrixXd cached_displaced_surface;
		if (cached_displaced_surface.size() == displaced_surface.size() && cached_displaced_surface == displaced_surface)
			return;

		collision_set.build(collision_mesh_, displaced_surface, dhat_, 0, broad_phase_method_);

		cached_displaced_surface = displaced_surface;
	}

	Eigen::VectorXd DeformedCollisionBarrierForm::get_updated_mesh_nodes(const Eigen::VectorXd &x) const
	{
		Eigen::VectorXd X = X_init;
		variable_to_simulations_.compute_state_variable(ParameterType::Shape, &state_, x, X);
		return AdjointTools::map_primitive_to_node_order(state_, X) + state_.diff_cached.u(0);
	}
} // namespace polyfem::solver