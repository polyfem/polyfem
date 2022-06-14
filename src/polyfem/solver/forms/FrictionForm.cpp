#include "FrictionForm.hpp"

#include <ipc/ipc.hpp>

namespace polyfem
{
	namespace solver
	{
		FrictionForm::FrictionForm(const double epsv, const double mu, const ipc::CollisionMesh &collision_mesh)
			: epsv_(epsv), mu_(mu), collision_mesh_(collision_mesh)
		{
			//TODO
			// epsv_ = state.args["contact"]["epsv"];
			assert(epsv_ > 0);
			// mu_ = state.args["contact"]["friction_coefficient"];
			assert(mu_ > 0);
		}

		double FrictionForm::value(const Eigen::VectorXd &x)
		{
			return ipc::compute_friction_potential(
				collision_mesh_, collision_mesh_.vertices(displaced_prev()),
				displaced_surface, friction_constraint_set_, epsv_ * dt());
		}
		void FrictionForm::gradient(const Eigen::VectorXd &x, Eigen::VectorXd &gradv)
		{
			Eigen::VectorXd grad_friction = ipc::compute_friction_potential_gradient(
				collision_mesh_, displaced_surface_prev, displaced_surface,
				friction_constraint_set_, epsv_ * dt());
			gradv = collision_mesh_.to_full_dof(grad_friction);
		}

		void FrictionForm::hessian(const Eigen::VectorXd &x, StiffnessMatrix &hessian)
		{
			POLYFEM_SCOPED_TIMER("\t\tfriction hessian time");
			hessian = ipc::compute_friction_potential_hessian(
				collision_mesh_, displaced_surface_prev, displaced_surface,
				friction_constraint_set_, epsv_ * dt(), project_to_psd_);

			hessian = collision_mesh_.to_full_dof(hessian);
		}

		//more than one step?
		void FrictionForm::init_lagging(const Eigen::VectorXd &x)
		{
			update_lagging(x);
		}

		void FrictionForm::update_lagging(const Eigen::VectorXd &x)
		{
			Eigen::MatrixXd displaced_surface = collision_mesh_.vertices(displaced);

			update_constraint_set(displaced_surface);
			ipc::constructfriction_constraint_set_(
				collision_mesh_, displaced_surface, _constraint_set,
				_dhat, _barrier_stiffness, mu_, friction_constraint_set_);
		}

	} // namespace solver
} // namespace polyfem
