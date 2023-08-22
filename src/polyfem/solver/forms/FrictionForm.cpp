#include "FrictionForm.hpp"
#include "ContactForm.hpp"

#include <polyfem/utils/Timer.hpp>
#include <polyfem/utils/MatrixUtils.hpp>

namespace polyfem::solver
{
	FrictionForm::FrictionForm(
		const ipc::CollisionMesh &collision_mesh,
		const std::shared_ptr<time_integrator::ImplicitTimeIntegrator> time_integrator,
		const double epsv,
		const double mu,
		const double dhat,
		const ipc::BroadPhaseMethod broad_phase_method,
		const ContactForm &contact_form,
		const int n_lagging_iters)
		: collision_mesh_(collision_mesh),
		  time_integrator_(time_integrator),
		  epsv_(epsv),
		  mu_(mu),
		  dhat_(dhat),
		  broad_phase_method_(broad_phase_method),
		  n_lagging_iters_(n_lagging_iters < 0 ? std::numeric_limits<int>::max() : n_lagging_iters),
		  contact_form_(contact_form)

	{
		assert(epsv_ > 0);
	}

	void FrictionForm::force_shape_derivative(
		const Eigen::MatrixXd &prev_solution,
		const Eigen::MatrixXd &solution,
		const Eigen::MatrixXd &adjoint,
		const ipc::FrictionConstraints &friction_constraints_set,
		Eigen::VectorXd &term)
	{
		Eigen::MatrixXd U = collision_mesh_.vertices(utils::unflatten(solution, collision_mesh_.dim()));
		Eigen::MatrixXd U_prev = collision_mesh_.vertices(utils::unflatten(prev_solution, collision_mesh_.dim()));

		// TODO: use the time integration to compute the velocity
		const Eigen::MatrixXd velocities = (U - U_prev) / time_integrator_->dt();

		StiffnessMatrix hess = -friction_constraints_set.compute_force_jacobian(
			collision_mesh_, collision_mesh_.rest_positions(),
			/*lagged_displacements=*/U_prev, velocities,
			dhat_, contact_form_.barrier_stiffness(), epsv_,
			ipc::FrictionConstraint::DiffWRT::REST_POSITIONS);

		// {
		// 	Eigen::MatrixXd X = collision_mesh_.rest_positions();
		// 	Eigen::VectorXd x = utils::flatten(X);
		// 	const double barrier_stiffness = contact_form_.barrier_stiffness();
		// 	const double dhat = dhat_;
		// 	const double mu = mu_;
		// 	const double epsv = epsv_;
		// 	const double dt = time_integrator_->dt();

		// 	Eigen::MatrixXd fgrad;
		// 	fd::finite_jacobian(
		// 		x, [&](const Eigen::VectorXd &y) -> Eigen::VectorXd
		// 		{
		// 			Eigen::MatrixXd fd_X = utils::unflatten(y, X.cols());

		// 			ipc::CollisionMesh fd_mesh(fd_X, collision_mesh_.edges(), collision_mesh_.faces());
		// 			fd_mesh.init_area_jacobians();

		// 			ipc::FrictionConstraints fd_friction_constraints;
		// 			ipc::CollisionConstraints fd_constraints;
		// 			fd_constraints.set_use_convergent_formulation(contact_form_.use_convergent_formulation());
		// 			fd_constraints.set_are_shape_derivatives_enabled(true);
		// 			fd_constraints.build(fd_mesh, fd_X + U_prev, dhat);

		// 			fd_friction_constraints.build(
		// 				fd_mesh, fd_X + U_prev, fd_constraints, dhat, barrier_stiffness,
		// 				mu);

		// 			return fd_friction_constraints.compute_potential_gradient(fd_mesh, (U - U_prev) / dt, epsv);

		// 		}, fgrad, fd::AccuracyOrder::SECOND, 1e-8);

		// 	std::cout << "force shape derivative error " << (fgrad - hess).norm() << " " << hess.norm() << "\n";
		// }

		term = collision_mesh_.to_full_dof(hess).transpose() * adjoint;
	}

	Eigen::MatrixXd FrictionForm::compute_displaced_surface(const Eigen::VectorXd &x) const
	{
		return contact_form_.compute_displaced_surface(x);
	}

	Eigen::MatrixXd FrictionForm::compute_surface_velocities(const Eigen::VectorXd &x) const
	{
		// In the case of a static problem, the velocity is the displacement
		const Eigen::VectorXd v = time_integrator_ != nullptr ? time_integrator_->compute_velocity(x) : x;
		return collision_mesh_.map_displacements(utils::unflatten(v, collision_mesh_.dim()));
	}

	double FrictionForm::dv_dx() const
	{
		return time_integrator_ != nullptr ? time_integrator_->dv_dx() : 1;
	}

	double FrictionForm::value_unweighted(const Eigen::VectorXd &x) const
	{
		return friction_constraint_set_.compute_potential(collision_mesh_, compute_surface_velocities(x), epsv_) / dv_dx();
	}

	void FrictionForm::first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		const Eigen::VectorXd grad_friction = friction_constraint_set_.compute_potential_gradient(
			collision_mesh_, compute_surface_velocities(x), epsv_);
		gradv = collision_mesh_.to_full_dof(grad_friction);
	}

	void FrictionForm::second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const
	{
		POLYFEM_SCOPED_TIMER("friction hessian");

		hessian = dv_dx() * friction_constraint_set_.compute_potential_hessian( //
					  collision_mesh_, compute_surface_velocities(x), epsv_, project_to_psd_);

		hessian = collision_mesh_.to_full_dof(hessian);
	}

	void FrictionForm::update_lagging(const Eigen::VectorXd &x, const int iter_num)
	{
		const Eigen::MatrixXd displaced_surface = compute_displaced_surface(x);

		ipc::CollisionConstraints constraint_set;
		constraint_set.set_use_convergent_formulation(contact_form_.use_convergent_formulation());
		constraint_set.set_are_shape_derivatives_enabled(contact_form_.enable_shape_derivatives());
		constraint_set.build(
			collision_mesh_, displaced_surface, dhat_,
			/*dmin=*/0, broad_phase_method_);

		friction_constraint_set_.build(
			collision_mesh_, displaced_surface, constraint_set,
			dhat_, contact_form_.barrier_stiffness(), mu_);
	}
} // namespace polyfem::solver
