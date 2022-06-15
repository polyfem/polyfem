#include "FrictionForm.hpp"

#include <polyfem/utils/Timer.hpp>

#include <ipc/ipc.hpp>

namespace polyfem
{
	namespace solver
	{
		FrictionForm::FrictionForm(const State &state, const double epsv, const double mu, const double dt, const ipc::CollisionMesh &collision_mesh)
			: epsv_(epsv), mu_(mu), dt_(dt), collision_mesh_(collision_mesh), state_(state)
		{
			//TODO
			// epsv_ = state_.args["contact"]["epsv"];
			assert(epsv_ > 0);
			// mu_ = state_.args["contact"]["friction_coefficient"];
			assert(mu_ > 0);
		}

		double FrictionForm::value(const Eigen::VectorXd &x)
		{
			const Eigen::MatrixXd displaced = state_.boundary_nodes_pos + utils::unflatten(x, state_.mesh->dimension());
			const Eigen::MatrixXd displaced_surface = state_.collision_mesh.vertices(displaced);

			return ipc::compute_friction_potential(
				collision_mesh_, collision_mesh_.vertices(displaced_prev_),
				displaced_surface, friction_constraint_set_, epsv_ * dt_);
		}
		void FrictionForm::first_derivative(const Eigen::VectorXd &x, Eigen::VectorXd &gradv)
		{
			const Eigen::MatrixXd displaced = state_.boundary_nodes_pos + utils::unflatten(x, state_.mesh->dimension());
			const Eigen::MatrixXd displaced_surface = state_.collision_mesh.vertices(displaced);
			const Eigen::MatrixXd displaced_surface_prev = state_.collision_mesh.vertices(displaced_prev_);

			const Eigen::VectorXd grad_friction = ipc::compute_friction_potential_gradient(
				collision_mesh_, displaced_surface_prev, displaced_surface,
				friction_constraint_set_, epsv_ * dt_);
			gradv = collision_mesh_.to_full_dof(grad_friction);
		}

		void FrictionForm::second_derivative(const Eigen::VectorXd &x, StiffnessMatrix &hessian)
		{
			POLYFEM_SCOPED_TIMER("\t\tfriction hessian time");
			const Eigen::MatrixXd displaced = state_.boundary_nodes_pos + utils::unflatten(x, state_.mesh->dimension());
			const Eigen::MatrixXd displaced_surface = state_.collision_mesh.vertices(displaced);
			const Eigen::MatrixXd displaced_surface_prev = state_.collision_mesh.vertices(displaced_prev_);

			hessian = ipc::compute_friction_potential_hessian(
				collision_mesh_, displaced_surface_prev, displaced_surface,
				friction_constraint_set_, epsv_ * dt_, project_to_psd_);

			hessian = collision_mesh_.to_full_dof(hessian);
		}

		//more than one step?
		void FrictionForm::init_lagging(const Eigen::VectorXd &x)
		{
			displaced_prev_ = x;
			update_lagging(x);
		}

		void FrictionForm::update_lagging(const Eigen::VectorXd &x)
		{
			const Eigen::MatrixXd displaced = state_.boundary_nodes_pos + utils::unflatten(x, state_.mesh->dimension());
			const Eigen::MatrixXd displaced_surface = collision_mesh_.vertices(displaced);

			//TODO fix me
			// update_constraint_set(displaced_surface);

			// ipc::constructfriction_constraint_set(
			// 	collision_mesh_, displaced_surface, _constraint_set,
			// 	_dhat, _barrier_stiffness, mu_, friction_constraint_set_);
		}

	} // namespace solver
} // namespace polyfem
