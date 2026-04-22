#include "SmoothContactForceDerivative.hpp"

#include <Eigen/Core>
#include <polyfem/solver/forms/SmoothContactForm.hpp>
#include <polyfem/utils/Types.hpp>
#include <ipc/smooth_contact/smooth_collisions.hpp>
#include <ipc/potentials/potential.hpp>

namespace polyfem::solver
{
	void SmoothContactForceDerivative::force_shape_derivative(
		const SmoothContactForm &form,
		const ipc::SmoothCollisions &collision_set,
		const Eigen::MatrixXd &solution,
		const Eigen::VectorXd &adjoint_sol,
		Eigen::VectorXd &term)
	{
		StiffnessMatrix hessian = form.barrier_potential_.hessian(collision_set, form.collision_mesh_, form.compute_displaced_surface(solution), ipc::PSDProjectionMethod::NONE);
		term = form.barrier_stiffness() * form.collision_mesh_.to_full_dof(hessian) * adjoint_sol;
	}
} // namespace polyfem::solver
