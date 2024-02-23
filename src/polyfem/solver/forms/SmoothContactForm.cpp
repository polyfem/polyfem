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
                const int ccd_max_iterations): ContactForm(collision_mesh, args["dhat"], avg_mass, false, use_adaptive_barrier_stiffness, is_time_dependent, false, broad_phase_method, ccd_tolerance, ccd_max_iterations), params(dhat_, args["alpha_t"], args["beta_t"], args["alpha_n"], args["beta_n"], _dim == 3 ? 2 : 1), use_adaptive_dhat(args["use_adaptive_dhat"])
    {
		collision_set_ = std::make_shared<ipc::SmoothCollisions<_dim>>();
        contact_potential_ = std::make_shared<ipc::SmoothContactPotential<ipc::SmoothCollisions<_dim>>>(params);
		params.set_adaptive_dhat_ratio(args["min_distance_ratio"]);
		if (use_adaptive_dhat)
		{
			collision_set_->compute_adaptive_dhat(collision_mesh, collision_mesh.rest_positions(), params, broad_phase_method_);
			if (use_adaptive_barrier_stiffness)
				logger().error("Adaptive dhat is not compatible with adaptive barrier stiffness");
		}
	}

    template <int _dim>
    void SmoothContactForm<_dim>::update_barrier_stiffness(const Eigen::VectorXd &x, const Eigen::MatrixXd &grad_energy)
    {
		if (!use_adaptive_barrier_stiffness())
			return;

		max_barrier_stiffness_ = barrier_stiffness() * 4;
		// const Eigen::MatrixXd displaced_surface = compute_displaced_surface(x);

		// // The adative stiffness is designed for the non-convergent formulation,
		// // so we need to compute the gradient of the non-convergent barrier.
		// // After we can map it to a good value for the convergent formulation.
		// ipc::SmoothCollisions<_dim> collisions;
		// collisions.build(
		// 	collision_mesh_, displaced_surface, params, false, broad_phase_method_);
		// Eigen::VectorXd grad_barrier = contact_potential_->gradient(
		// 	collisions, collision_mesh_, displaced_surface);
		// grad_barrier = collision_mesh_.to_full_dof(grad_barrier);

		// barrier_stiffness_ = ipc::initial_barrier_stiffness(grad_energy, grad_barrier, min_barrier_stiffness_ * weight_, max_barrier_stiffness_ * weight_);

		// // Remove the acceleration scaling from the barrier stiffness because it will be applied later.
		// barrier_stiffness_ /= weight_;

		// logger().debug("adaptive barrier form stiffness {}", barrier_stiffness());
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
				candidates_, collision_mesh_, displaced_surface, params, use_adaptive_dhat);
		else
			collision_set_->build(
				collision_mesh_, displaced_surface, params, use_adaptive_dhat, broad_phase_method_);
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
			const auto log_level = (ratio < 1e-6) ? spdlog::level::err : ((ratio < 1e-4) ? spdlog::level::warn : spdlog::level::debug);
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
					barrier_stiffness(), ipc::world_bbox_diagonal_length(displaced_surface), 1e-7);

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