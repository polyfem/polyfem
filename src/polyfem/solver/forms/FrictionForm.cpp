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
		  contact_form_(contact_form),
		  n_lagging_iters_(n_lagging_iters < 0 ? std::numeric_limits<int>::max() : n_lagging_iters)
	{
		assert(epsv_ > 0);
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
		constraint_set.build(
			collision_mesh_, displaced_surface, dhat_,
			/*dmin=*/0, broad_phase_method_);

		friction_constraint_set_.build(
			collision_mesh_, displaced_surface, constraint_set,
			dhat_, contact_form_.barrier_stiffness(), mu_);
	}
} // namespace polyfem::solver
