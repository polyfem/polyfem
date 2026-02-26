#include "FrictionForceDerivative.hpp"

#include <Eigen/Core>
#include <polyfem/solver/forms/FrictionForm.hpp>
#include <polyfem/solver/forms/BarrierContactForm.hpp>
#include <polyfem/utils/MatrixUtils.hpp>
#include <polyfem/utils/Types.hpp>
#include <ipc/collisions/tangential/tangential_collisions.hpp>
#include <ipc/potentials/friction_potential.hpp>

namespace polyfem::solver
{
	void FrictionForceDerivative::force_shape_derivative(
		FrictionForm &form,
		const Eigen::MatrixXd &prev_solution,
		const Eigen::MatrixXd &solution,
		const Eigen::MatrixXd &adjoint,
		const ipc::TangentialCollisions &friction_constraints_set,
		Eigen::VectorXd &term)
	{
		Eigen::MatrixXd U = form.collision_mesh_.vertices(utils::unflatten(solution, form.collision_mesh_.dim()));
		Eigen::MatrixXd U_prev = form.collision_mesh_.vertices(utils::unflatten(prev_solution, form.collision_mesh_.dim()));

		// TODO: use the time integration to compute the velocity
		const Eigen::MatrixXd velocities = (U - U_prev) / form.time_integrator_->dt();

		StiffnessMatrix hess;
		if (const auto barrier_contact = dynamic_cast<const BarrierContactForm *>(&form.contact_form_))
		{
			hess = -form.friction_potential_.force_jacobian(
				friction_constraints_set,
				form.collision_mesh_, form.collision_mesh_.rest_positions(),
				/*lagged_displacements=*/U_prev, velocities,
				barrier_contact->barrier_potential(),
				barrier_contact->barrier_stiffness(),
				ipc::FrictionPotential::DiffWRT::REST_POSITIONS);
		}

		term = form.collision_mesh_.to_full_dof(hess).transpose() * adjoint;
	}
} // namespace polyfem::solver
