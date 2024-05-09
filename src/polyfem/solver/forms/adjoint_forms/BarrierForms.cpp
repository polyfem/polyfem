#include "BarrierForms.hpp"
#include <polyfem/State.hpp>
#include <polyfem/utils/BoundarySampler.hpp>

namespace polyfem::solver
{
	CollisionBarrierForm::CollisionBarrierForm(const std::vector<std::shared_ptr<VariableToSimulation>> variable_to_simulation, const State &state, const double dhat)
		: AdjointForm(variable_to_simulation), state_(state), dhat_(dhat), barrier_potential_(dhat)
	{
		State::build_collision_mesh(
			*state_.mesh, state_.n_geom_bases, state_.geom_bases(), state_.geom_bases(),
			state_.total_local_boundary, state_.obstacle, state_.args,
			[this](const std::string &p) { return this->state_.resolve_input_path(p); },
			state_.in_node_to_node, state_.node_to_body_id, collision_mesh_);

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

	void CollisionBarrierForm::compute_partial_gradient_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		const Eigen::MatrixXd displaced_surface = collision_mesh_.vertices(utils::unflatten(get_updated_mesh_nodes(x), state_.mesh->dimension()));

		Eigen::VectorXd grad = collision_mesh_.to_full_dof(barrier_potential_.gradient(collision_set, collision_mesh_, displaced_surface));

		grad = AdjointTools::map_node_to_primitive_order(state_, grad);

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

	void CollisionBarrierForm::solution_changed(const Eigen::VectorXd &x)
	{
		AdjointForm::solution_changed(x);

		const Eigen::MatrixXd displaced_surface = collision_mesh_.vertices(utils::unflatten(get_updated_mesh_nodes(x), state_.mesh->dimension()));
		build_collision_set(displaced_surface);
	}

	Eigen::MatrixXd CollisionBarrierForm::compute_adjoint_rhs_unweighted(const Eigen::VectorXd &x, const State &state) const
	{
		return Eigen::MatrixXd::Zero(state.ndof(), state.diff_cached.size());
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

	DeformedCollisionBarrierForm::DeformedCollisionBarrierForm(const std::vector<std::shared_ptr<VariableToSimulation>> variable_to_simulation, const State &state, const double dhat)
		: AdjointForm(variable_to_simulation), state_(state), dhat_(dhat), barrier_potential_(dhat)
	{
		if (state_.n_bases != state_.n_geom_bases)
			log_and_throw_adjoint_error("[{}] Should use linear FE basis!", name());

		State::build_collision_mesh(
			*state_.mesh, state_.n_geom_bases, state_.geom_bases(), state_.geom_bases(),
			state_.total_local_boundary, state_.obstacle, state_.args,
			[this](const std::string &p) { return this->state_.resolve_input_path(p); },
			state_.in_node_to_node, state_.node_to_body_id, collision_mesh_);

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

	void DeformedCollisionBarrierForm::compute_partial_gradient_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		const Eigen::MatrixXd displaced_surface = collision_mesh_.vertices(utils::unflatten(get_updated_mesh_nodes(x), state_.mesh->dimension()));

		Eigen::VectorXd grad = collision_mesh_.to_full_dof(barrier_potential_.gradient(collision_set, collision_mesh_, displaced_surface));

		grad = AdjointTools::map_node_to_primitive_order(state_, grad);

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

	void DeformedCollisionBarrierForm::solution_changed(const Eigen::VectorXd &x)
	{
		AdjointForm::solution_changed(x);

		const Eigen::MatrixXd displaced_surface = collision_mesh_.vertices(utils::unflatten(get_updated_mesh_nodes(x), state_.mesh->dimension()));
		build_collision_set(displaced_surface);
	}

	Eigen::MatrixXd DeformedCollisionBarrierForm::compute_adjoint_rhs_unweighted(const Eigen::VectorXd &x, const State &state) const
	{
		return Eigen::MatrixXd::Zero(state.ndof(), state.diff_cached.size());
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

		return AdjointTools::map_primitive_to_node_order(state_, X) + state_.diff_cached.u(0);
	}

	template <int dim>
	SmoothContactForceForm<dim>::SmoothContactForceForm(
		const std::vector<std::shared_ptr<VariableToSimulation>> &variable_to_simulations,
		const State &state,
		const json &args)
		: StaticForm(variable_to_simulations),
		  state_(state),
		  params_(state.args["contact"]["dhat"], state.args["contact"]["alpha_t"], state.args["contact"]["beta_t"], state.args["contact"]["alpha_n"], state.args["contact"]["beta_n"], state.mesh->is_volume() ? 2 : 1),
		  potential_(params_)
	{
		assert(dim == state.mesh->dimension());

		auto tmp_ids = args["surface_selection"].get<std::vector<int>>();
		boundary_ids_ = std::set(tmp_ids.begin(), tmp_ids.end());

		build_collision_mesh();

		broad_phase_method_ = ipc::BroadPhaseMethod::HASH_GRID;

		if (state.problem->is_time_dependent())
		{
			int time_steps = state.args["time"]["time_steps"].get<int>() + 1;
			collision_set_indicator_.setZero(time_steps);
			for (int i = 0; i < time_steps + 1; ++i)
			{
				collision_sets_.push_back(std::make_shared<ipc::SmoothCollisions<dim>>());
				// collision_sets_.back()->set_use_convergent_formulation(true);
				collision_sets_.back()->set_are_shape_derivatives_enabled(true);
			}
		}
		else
		{
			collision_set_indicator_.setZero(1);
			collision_sets_.push_back(std::make_shared<ipc::SmoothCollisions<dim>>());
			// collision_sets_.back()->set_use_convergent_formulation(true);
			collision_sets_.back()->set_are_shape_derivatives_enabled(true);
		}
	}

	template <int dim>
	void SmoothContactForceForm<dim>::build_collision_mesh()
	{
		boundary_ids_to_dof_.clear();
		can_collide_cache_.resize(0, 0);

		Eigen::MatrixXd node_positions;
		Eigen::MatrixXi boundary_edges, boundary_triangles;
		std::vector<Eigen::Triplet<double>> displacement_map_entries;
		io::OutGeometryData::extract_boundary_mesh(*state_.mesh, state_.n_bases, state_.bases, state_.total_local_boundary,
												   node_positions, boundary_edges, boundary_triangles, displacement_map_entries);

		std::vector<bool> is_on_surface;
		is_on_surface.resize(node_positions.rows(), false);

		assembler::ElementAssemblyValues vals;
		Eigen::MatrixXd points, uv, normals;
		Eigen::VectorXd weights;
		Eigen::VectorXi global_primitive_ids;
		for (const auto &lb : state_.total_local_boundary)
		{
			const int e = lb.element_id();
			bool has_samples = utils::BoundarySampler::boundary_quadrature(lb, state_.n_boundary_samples(), *state_.mesh, false, uv, points, normals, weights, global_primitive_ids);

			if (!has_samples)
				continue;

			const basis::ElementBases &bs = state_.bases[e];
			const basis::ElementBases &gbs = state_.geom_bases()[e];

			vals.compute(e, state_.mesh->is_volume(), points, bs, gbs);

			for (int i = 0; i < lb.size(); ++i)
			{
				const int primitive_global_id = lb.global_primitive_id(i);
				const auto nodes = bs.local_nodes_for_primitive(primitive_global_id, *state_.mesh);
				const int boundary_id = state_.mesh->get_boundary_id(primitive_global_id);

				if (!std::count(boundary_ids_.begin(), boundary_ids_.end(), boundary_id))
					continue;

				for (long n = 0; n < nodes.size(); ++n)
				{
					const assembler::AssemblyValues &v = vals.basis_values[nodes(n)];
					is_on_surface[v.global[0].index] = true;
					if (v.global[0].index >= node_positions.rows())
						log_and_throw_adjoint_error("Error building collision mesh in SmoothContactForceForm!");
					boundary_ids_to_dof_[boundary_id].insert(v.global[0].index);
				}
			}
		}

		Eigen::SparseMatrix<double> displacement_map;
		if (!displacement_map_entries.empty())
		{
			displacement_map.resize(node_positions.rows(), state_.n_bases);
			displacement_map.setFromTriplets(displacement_map_entries.begin(), displacement_map_entries.end());
		}

		// Fix boundary edges and boundary triangles to exclude vertices not on triangles
		Eigen::MatrixXi boundary_edges_alt(0, 2), boundary_triangles_alt(0, 3);
		{
			for (int i = 0; i < boundary_edges.rows(); ++i)
			{
				bool on_surface = true;
				for (int j = 0; j < boundary_edges.cols(); ++j)
					on_surface &= is_on_surface[boundary_edges(i, j)];
				if (on_surface)
				{
					boundary_edges_alt.conservativeResize(boundary_edges_alt.rows() + 1, 2);
					boundary_edges_alt.row(boundary_edges_alt.rows() - 1) = boundary_edges.row(i);
				}
			}

			if (state_.mesh->is_volume())
			{
				for (int i = 0; i < boundary_triangles.rows(); ++i)
				{
					bool on_surface = true;
					for (int j = 0; j < boundary_triangles.cols(); ++j)
						on_surface &= is_on_surface[boundary_triangles(i, j)];
					if (on_surface)
					{
						boundary_triangles_alt.conservativeResize(boundary_triangles_alt.rows() + 1, 3);
						boundary_triangles_alt.row(boundary_triangles_alt.rows() - 1) = boundary_triangles.row(i);
					}
				}
			}
			else
				boundary_triangles_alt.resize(0, 0);
		}

		collision_mesh_ = ipc::CollisionMesh(is_on_surface,
											 node_positions,
											 boundary_edges_alt,
											 boundary_triangles_alt,
											 displacement_map);

		can_collide_cache_.resize(collision_mesh_.num_vertices(), collision_mesh_.num_vertices());
		for (int i = 0; i < can_collide_cache_.rows(); ++i)
		{
			int dof_idx_i = collision_mesh_.to_full_vertex_id(i);
			if (!is_on_surface[dof_idx_i])
				continue;
			for (int j = 0; j < can_collide_cache_.cols(); ++j)
			{
				int dof_idx_j = collision_mesh_.to_full_vertex_id(j);
				if (!is_on_surface[dof_idx_j])
					continue;

				bool collision_allowed = true;
				for (const auto &id : boundary_ids_)
					if (boundary_ids_to_dof_[id].count(dof_idx_i) && boundary_ids_to_dof_[id].count(dof_idx_j))
						collision_allowed = false;
				can_collide_cache_(i, j) = collision_allowed;
			}
		}

		collision_mesh_.can_collide = [this](size_t vi, size_t vj) {
			return (bool)can_collide_cache_(vi, vj);
		};

		collision_mesh_.init_area_jacobians();
	}

	template <int dim>
	const ipc::SmoothCollisions<dim> &SmoothContactForceForm<dim>::get_or_compute_collision_set(const int time_step, const Eigen::MatrixXd &displaced_surface) const
	{
		if (!collision_set_indicator_(time_step))
		{
			collision_sets_[time_step]->build(
				collision_mesh_, displaced_surface, params_, false, broad_phase_method_);
			collision_set_indicator_(time_step) = 1;
		}
		return *collision_sets_[time_step];
	}

	template <int dim>
	double SmoothContactForceForm<dim>::value_unweighted_step(const int time_step, const Eigen::VectorXd &x) const
	{
		assert(state_.solve_data.time_integrator != nullptr);
		assert(state_.solve_data.contact_form != nullptr);

		const Eigen::MatrixXd displaced_surface = collision_mesh_.displace_vertices(utils::unflatten(state_.diff_cached.u(time_step), collision_mesh_.dim()));
		auto collision_set = get_or_compute_collision_set(time_step, displaced_surface);

		Eigen::VectorXd forces = potential_.gradient(collision_set, collision_mesh_, displaced_surface);
		
		return collision_mesh_.to_full_dof(forces).squaredNorm();
	}

	template <int dim>
	Eigen::VectorXd SmoothContactForceForm<dim>::compute_adjoint_rhs_unweighted_step(const int time_step, const Eigen::VectorXd &x, const State &state) const
	{
		assert(state_.solve_data.time_integrator != nullptr);
		assert(state_.solve_data.contact_form != nullptr);

		const Eigen::MatrixXd displaced_surface = collision_mesh_.displace_vertices(utils::unflatten(state_.diff_cached.u(time_step), collision_mesh_.dim()));
		auto collision_set = get_or_compute_collision_set(time_step, displaced_surface);

		Eigen::VectorXd forces = potential_.gradient(collision_set, collision_mesh_, displaced_surface);
		forces = collision_mesh_.to_full_dof(forces);

		StiffnessMatrix hessian = potential_.hessian(collision_set, collision_mesh_, displaced_surface, false);
		hessian = collision_mesh_.to_full_dof(hessian);

		Eigen::VectorXd gradu = 2 * hessian.transpose() * forces;

		return gradu;
	}

	template <int dim>
	void SmoothContactForceForm<dim>::compute_partial_gradient_unweighted_step(const int time_step, const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		assert(state_.solve_data.time_integrator != nullptr);
		assert(state_.solve_data.contact_form != nullptr);

		const Eigen::MatrixXd displaced_surface = collision_mesh_.displace_vertices(utils::unflatten(state_.diff_cached.u(time_step), collision_mesh_.dim()));
		auto collision_set = get_or_compute_collision_set(time_step, displaced_surface);

		Eigen::VectorXd forces = potential_.gradient(collision_set, collision_mesh_, displaced_surface);
		forces = collision_mesh_.to_full_dof(forces);

		StiffnessMatrix hessian = potential_.hessian(collision_set, collision_mesh_, displaced_surface, false);
		hessian = collision_mesh_.to_full_dof(hessian);

		Eigen::VectorXd grads = 2 * hessian.transpose() * forces;
		grads = state_.gbasis_nodes_to_basis_nodes * grads;
		grads = AdjointTools::map_node_to_primitive_order(state_, grads);

		gradv.setZero(x.size());

		for (const auto &param_map : variable_to_simulations_)
		{
			const auto &param_type = param_map->get_parameter_type();

			for (const auto &state : param_map->get_states())
			{
				if (state.get() != &state_)
					continue;

				if (param_type == ParameterType::Shape)
					gradv += param_map->apply_parametrization_jacobian(grads, x);
			}
		}
	}

	template class SmoothContactForceForm<2>;
	template class SmoothContactForceForm<3>;
} // namespace polyfem::solver