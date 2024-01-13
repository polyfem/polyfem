#include "SmoothContactForm.hpp"
#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/Types.hpp>
#include <polyfem/utils/Timer.hpp>
#include <polyfem/utils/MatrixUtils.hpp>
#include <polyfem/utils/MaybeParallelFor.hpp>

#include <ipc/barrier/adaptive_stiffness.hpp>
#include <ipc/utils/world_bbox_diagonal_length.hpp>

namespace polyfem::solver
{
    template <int _dim>
    SmoothContactForm<_dim>::SmoothContactForm(const ipc::CollisionMesh &collision_mesh,
                const json &args,
                const double avg_mass,
                const bool use_adaptive_barrier_stiffness,
                const bool is_time_dependent,
                const ipc::BroadPhaseMethod broad_phase_method,
                const double ccd_tolerance,
                const int ccd_max_iterations): ContactForm(collision_mesh, args["dhat"], avg_mass, false, use_adaptive_barrier_stiffness, is_time_dependent, false, broad_phase_method, ccd_tolerance, ccd_max_iterations), params(dhat_*dhat_, args["alpha"], args["a"], args["r"], args["high_order_quadrature"])
    {
		collision_set_ = std::make_shared<ipc::SmoothCollisions<_dim>>(args["high_order_quadrature"].get<int>() > 1, args["use_adaptive_epsilon"]);
        contact_potential_ = std::make_shared<ipc::SmoothContactPotential<ipc::SmoothCollisions<_dim>>>(params);
        if (params.a > 0)
            logger().error("The contact candidate search size is likely wrong!");
    }

    template <int _dim>
    void SmoothContactForm<_dim>::update_barrier_stiffness(const Eigen::VectorXd &x, const Eigen::MatrixXd &grad_energy)
    {
        log_and_throw_error("[{}] Barrier stiffness update not implemented!", name());
    }

    template <int _dim>
    void SmoothContactForm<_dim>::update_collision_set(const Eigen::MatrixXd &displaced_surface)
	{
		// Store the previous value used to compute the constraint set to avoid duplicate computation.
		static Eigen::MatrixXd cached_displaced_surface;
		if (cached_displaced_surface.size() == displaced_surface.size() && cached_displaced_surface == displaced_surface)
			return;

		if (use_cached_candidates_)
			collision_set_->build(
				candidates_, collision_mesh_, displaced_surface, barrier_support_size());
		else
			collision_set_->build(
				collision_mesh_, displaced_surface, barrier_support_size(), dmin_, broad_phase_method_);
		cached_displaced_surface = displaced_surface;
	}

    template <int _dim>
	double SmoothContactForm<_dim>::value_unweighted(const Eigen::VectorXd &x) const
	{
		return (*contact_potential_)(*collision_set_, collision_mesh_, compute_displaced_surface(x));
	}

    template <int _dim>
	Eigen::VectorXd SmoothContactForm<_dim>::value_per_element_unweighted(const Eigen::VectorXd &x) const
	{
		log_and_throw_error("value_per_element_unweighted not implemented!");
		// const Eigen::MatrixXd V = compute_displaced_surface(x);
		// assert(V.rows() == collision_mesh_.num_vertices());

		// const size_t num_vertices = collision_mesh_.num_vertices();

		// if (collision_set_->empty())
		// {
		// 	return Eigen::VectorXd::Zero(collision_mesh_.full_num_vertices());
		// }

		// const Eigen::MatrixXi &E = collision_mesh_.edges();
		// const Eigen::MatrixXi &F = collision_mesh_.faces();

		// auto storage = utils::create_thread_storage<Eigen::VectorXd>(Eigen::VectorXd::Zero(num_vertices));

		// utils::maybe_parallel_for(collision_set_->size(), [&](int start, int end, int thread_id) {
		// 	Eigen::VectorXd &local_storage = utils::get_local_thread_storage(storage, thread_id);

		// 	for (size_t i = start; i < end; i++)
		// 	{
		// 		// Quadrature weight is premultiplied by compute_potential
		// 		const double potential = (*contact_potential_)((*collision_set_)[i], (*collision_set_)[i].dof(V, E, F));

		// 		const int n_v = (*collision_set_)[i].num_vertices();
		// 		const std::array<long, 4> vis = (*collision_set_)[i].vertex_ids(E, F);
		// 		for (int j = 0; j < n_v; j++)
		// 		{
		// 			assert(0 <= vis[j] && vis[j] < num_vertices);
		// 			local_storage[vis[j]] += potential / n_v;
		// 		}
		// 	}
		// });

		// Eigen::VectorXd out = Eigen::VectorXd::Zero(num_vertices);
		// for (const auto &local_potential : storage)
		// {
		// 	out += local_potential;
		// }

		// Eigen::VectorXd out_full = Eigen::VectorXd::Zero(collision_mesh_.full_num_vertices());
		// for (int i = 0; i < out.size(); i++)
		// 	out_full[collision_mesh_.to_full_vertex_id(i)] = out[i];

		// assert(std::abs(value_unweighted(x) - out_full.sum()) < std::max(1e-10 * out_full.sum(), 1e-10));

		// return out_full;
	}

    template <int _dim>
	void SmoothContactForm<_dim>::first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		gradv = contact_potential_->gradient(*collision_set_, collision_mesh_, compute_displaced_surface(x));
		gradv = collision_mesh_.to_full_dof(gradv);
	}

    template <int _dim>
	void SmoothContactForm<_dim>::second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const
	{
		POLYFEM_SCOPED_TIMER("barrier hessian");
		hessian = contact_potential_->hessian(*collision_set_, collision_mesh_, compute_displaced_surface(x), project_to_psd_);
		hessian = collision_mesh_.to_full_dof(hessian);
	}

    template <int _dim>
	void SmoothContactForm<_dim>::post_step(const polysolve::nonlinear::PostStepData &data)
	{
		const Eigen::MatrixXd displaced_surface = compute_displaced_surface(data.x);

		const double curr_distance = collision_set_->compute_minimum_distance(collision_mesh_, displaced_surface);
		if (!std::isinf(curr_distance))
		{
			const double ratio = sqrt(curr_distance) / dhat();
			const auto log_level = spdlog::level::debug; //(ratio < 1e-4) ? spdlog::level::err : ((ratio < 1e-2) ? spdlog::level::warn : spdlog::level::debug);
			polyfem::logger().log(log_level, "Minimum distance during solve: {}, dhat: {}", sqrt(curr_distance), dhat());
		}

		if (data.iter_num == 0)
			return;

		if (use_adaptive_barrier_stiffness_)
		{
			if (is_time_dependent_)
			{
				const double prev_barrier_stiffness = barrier_stiffness();

				barrier_stiffness_ = ipc::update_barrier_stiffness(
					prev_distance_, curr_distance, max_barrier_stiffness_,
					barrier_stiffness(), ipc::world_bbox_diagonal_length(displaced_surface));

				if (barrier_stiffness() != prev_barrier_stiffness)
				{
					polyfem::logger().debug(
						"updated barrier stiffness from {:g} to {:g}",
						prev_barrier_stiffness, barrier_stiffness());
				}
			}
			else
			{
				// TODO: missing feature
				// update_barrier_stiffness(data.x);
			}
		}

		prev_distance_ = curr_distance;
	}

	template class SmoothContactForm<2>;
	template class SmoothContactForm<3>;
}