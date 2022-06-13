#pragma once

#include <polyfem/utils/Types.hpp>

namespace polyfem
{
	namespace solver
	{
		FrictionForm::FrictionForm()
		{
			_epsv = state.args["contact"]["epsv"];
			assert(_epsv > 0);
			_mu = state.args["contact"]["friction_coefficient"];
			assert(_mu > 0);
		}

		double FrictionForm::value(const Eigen::VectorXd &x)
		{
			return ipc::compute_friction_potential(
				state.collision_mesh, state.collision_mesh.vertices(displaced_prev()),
				displaced_surface, _friction_constraint_set, _epsv * dt());
		}
		void FrictionForm::gradient(const Eigen::VectorXd &x, Eigen::VectorXd &gradv)
		{
			Eigen::VectorXd grad_friction = ipc::compute_friction_potential_gradient(
				state.collision_mesh, displaced_surface_prev, displaced_surface,
				_friction_constraint_set, _epsv * dt());
			grad += state.collision_mesh.to_full_dof(grad_friction);
		}
		void FrictionForm::hessian(const Eigen::VectorXd &x, StiffnessMatrix &hessian)
		{
			POLYFEM_SCOPED_TIMER("\t\tfriction hessian time");
			friction_hessian = ipc::compute_friction_potential_hessian(
				state.collision_mesh, displaced_surface_prev, displaced_surface,
				_friction_constraint_set, _epsv * dt(), project_to_psd);
			friction_hessian = state.collision_mesh.to_full_dof(friction_hessian);
		}

		//more than one step?
		void FrictionForm::init_lagging(const Eigen::VectorXd &x)
		{
			update_lagging(x);
		}

		void FrictionForm::update_lagging(const Eigen::VectorXd &x)
		{
			Eigen::MatrixXd displaced_surface = state.collision_mesh.vertices(displaced);

			update_constraint_set(displaced_surface);
			ipc::construct_friction_constraint_set(
				state.collision_mesh, displaced_surface, _constraint_set,
				_dhat, _barrier_stiffness, _mu, _friction_constraint_set);
		}

	} // namespace solver
} // namespace polyfem
