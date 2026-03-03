#include "TangentialAdhesionForceDerivative.hpp"

#include <Eigen/Core>
#include <polyfem/solver/forms/TangentialAdhesionForm.hpp>
#include <polyfem/utils/MatrixUtils.hpp>
#include <polyfem/utils/Types.hpp>
#include <ipc/collisions/tangential/tangential_collisions.hpp>
#include <ipc/potentials/tangential_adhesion_potential.hpp>

namespace polyfem::solver
{
	void TangentialAdhesionForceDerivative::force_shape_derivative(
		TangentialAdhesionForm &form,
		const Eigen::MatrixXd &prev_solution,
		const Eigen::MatrixXd &solution,
		const Eigen::MatrixXd &adjoint,
		const ipc::TangentialCollisions &tangential_constraints_set,
		Eigen::VectorXd &term)
	{
		Eigen::MatrixXd U = form.collision_mesh_.vertices(utils::unflatten(solution, form.collision_mesh_.dim()));
		Eigen::MatrixXd U_prev = form.collision_mesh_.vertices(utils::unflatten(prev_solution, form.collision_mesh_.dim()));

		// TODO: use the time integration to compute the velocity
		const Eigen::MatrixXd velocities = (U - U_prev) / form.time_integrator_->dt();

		StiffnessMatrix hess = -form.tangential_adhesion_potential_.force_jacobian(
			tangential_constraints_set,
			form.collision_mesh_, form.collision_mesh_.rest_positions(),
			/*lagged_displacements=*/U_prev, velocities,
			form.normal_adhesion_form_.normal_adhesion_potential(),
			1,
			ipc::TangentialPotential::DiffWRT::REST_POSITIONS);

		term = form.collision_mesh_.to_full_dof(hess).transpose() * adjoint;
	}
} // namespace polyfem::solver
