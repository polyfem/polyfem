#include "BarrierContactForceDerivative.hpp"

#include <Eigen/Core>
#include <polyfem/solver/forms/BarrierContactForm.hpp>
#include <polyfem/utils/Types.hpp>
#include <ipc/collisions/normal/normal_collisions.hpp>

namespace polyfem::solver
{
	void BarrierContactForceDerivative::force_shape_derivative(
		const BarrierContactForm &form,
		const ipc::NormalCollisions &collision_set,
		const Eigen::MatrixXd &solution,
		const Eigen::VectorXd &adjoint_sol,
		Eigen::VectorXd &term)
	{
		// Eigen::MatrixXd U = collision_mesh_.vertices(utils::unflatten(solution, collision_mesh_.dim()));
		// Eigen::MatrixXd X = collision_mesh_.vertices(boundary_nodes_pos_);
		const Eigen::MatrixXd displaced_surface = form.compute_displaced_surface(solution);

		StiffnessMatrix dq_h = form.collision_mesh_.to_full_dof(form.barrier_potential_.shape_derivative(collision_set, form.collision_mesh_, displaced_surface));
		term = form.barrier_stiffness() * dq_h.transpose() * adjoint_sol;
	}
} // namespace polyfem::solver
