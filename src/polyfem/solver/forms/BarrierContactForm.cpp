#include "BarrierContactForm.hpp"
#include <ipc/potentials/barrier_potential.hpp>
#include <polyfem/utils/Logger.hpp>

#include <ipc/barrier/adaptive_stiffness.hpp>
#include <ipc/utils/world_bbox_diagonal_length.hpp>

namespace polyfem::solver
{
    BarrierContactForm::BarrierContactForm(const ipc::CollisionMesh &collision_mesh,
					const double dhat,
					const double avg_mass,
					const bool use_convergent_formulation,
					const bool use_adaptive_barrier_stiffness,
					const bool is_time_dependent,
					const bool enable_shape_derivatives,
					const ipc::BroadPhaseMethod broad_phase_method,
					const double ccd_tolerance,
					const int ccd_max_iterations): ContactForm(collision_mesh, dhat, avg_mass, use_convergent_formulation, use_adaptive_barrier_stiffness, is_time_dependent, enable_shape_derivatives, broad_phase_method, ccd_tolerance, ccd_max_iterations)
    {
        contact_potential_ = std::make_shared<ipc::BarrierPotential>(dhat);
    }
	void BarrierContactForm::force_shape_derivative(const ipc::Collisions &collision_set, const Eigen::MatrixXd &solution, const Eigen::VectorXd &adjoint_sol, Eigen::VectorXd &term)
	{
		// Eigen::MatrixXd U = collision_mesh_.vertices(utils::unflatten(solution, collision_mesh_.dim()));
		// Eigen::MatrixXd X = collision_mesh_.vertices(boundary_nodes_pos_);
		const Eigen::MatrixXd displaced_surface = compute_displaced_surface(solution);

		StiffnessMatrix dq_h = collision_mesh_.to_full_dof(get_barrier_potential().shape_derivative(collision_set, collision_mesh_, displaced_surface));
		term = barrier_stiffness() * dq_h.transpose() * adjoint_sol;
	}

	void BarrierContactForm::update_barrier_stiffness(const Eigen::VectorXd &x, const Eigen::MatrixXd &grad_energy)
	{
		if (!use_adaptive_barrier_stiffness())
			return;

		const Eigen::MatrixXd displaced_surface = compute_displaced_surface(x);

		// The adative stiffness is designed for the non-convergent formulation,
		// so we need to compute the gradient of the non-convergent barrier.
		// After we can map it to a good value for the convergent formulation.
		ipc::Collisions nonconvergent_constraints;
		nonconvergent_constraints.set_use_convergent_formulation(false);
		nonconvergent_constraints.build(
			collision_mesh_, displaced_surface, dhat_, dmin_, broad_phase_method_);
		Eigen::VectorXd grad_barrier = contact_potential_->gradient(
			nonconvergent_constraints, collision_mesh_, displaced_surface);
		grad_barrier = collision_mesh_.to_full_dof(grad_barrier);

		barrier_stiffness_ = ipc::initial_barrier_stiffness(
			ipc::world_bbox_diagonal_length(displaced_surface), get_barrier_potential().barrier(), dhat_, avg_mass_,
			grad_energy, grad_barrier, max_barrier_stiffness_);

		if (use_convergent_formulation())
		{
			double scaling_factor = 0;
			if (!nonconvergent_constraints.empty())
			{
				const double nonconvergent_potential = (*contact_potential_)(
					nonconvergent_constraints, collision_mesh_, displaced_surface);

				update_collision_set(displaced_surface);
				const double convergent_potential = (*contact_potential_)(
					collision_set_, collision_mesh_, displaced_surface);

				scaling_factor = nonconvergent_potential / convergent_potential;
			}
			else
			{
				// Hardcoded difference between the non-convergent and convergent barrier
				scaling_factor = dhat_ * std::pow(dhat_ + 2 * dmin_, 2);
			}
			barrier_stiffness_ *= scaling_factor;
			max_barrier_stiffness_ *= scaling_factor;
		}

		// Remove the acceleration scaling from the barrier stiffness because it will be applied later.
		barrier_stiffness_ /= weight_;

		logger().debug("adaptive barrier form stiffness {}", barrier_stiffness());
	}

}