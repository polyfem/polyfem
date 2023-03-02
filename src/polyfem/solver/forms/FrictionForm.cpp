#include "FrictionForm.hpp"
#include "ContactForm.hpp"

#include <polyfem/utils/Timer.hpp>
#include <polyfem/utils/MatrixUtils.hpp>

namespace polyfem::solver
{
	FrictionForm::FrictionForm(
		const ipc::CollisionMesh &collision_mesh,
		const double epsv,
		const double mu,
		const double dhat,
		const ipc::BroadPhaseMethod broad_phase_method,
		const double dt,
		const ContactForm &contact_form,
		const int n_lagging_iters)
		: collision_mesh_(collision_mesh),
		  epsv_(epsv),
		  mu_(mu),
		  dt_(dt),
		  dhat_(dhat),
		  broad_phase_method_(broad_phase_method),
		  contact_form_(contact_form),
		  n_lagging_iters_(n_lagging_iters < 0 ? std::numeric_limits<int>::max() : n_lagging_iters)
	{
		assert(epsv_ > 0);
	}

	Eigen::MatrixXd FrictionForm::compute_displaced_surface(const Eigen::VectorXd &x) const
	{
		return collision_mesh_.displace_vertices(utils::unflatten(x, collision_mesh_.dim()));
	}

	double FrictionForm::value_unweighted(const Eigen::VectorXd &x) const
	{
		assert(displaced_surface_prev_.rows() == collision_mesh_.num_vertices());
		assert(displaced_surface_prev_.cols() == collision_mesh_.dim());

		return ipc::compute_friction_potential(
			collision_mesh_, displaced_surface_prev_, compute_displaced_surface(x),
			friction_constraint_set_, epsv_ * dt_);
	}
	void FrictionForm::first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		assert(displaced_surface_prev_.rows() == collision_mesh_.num_vertices());
		assert(displaced_surface_prev_.cols() == collision_mesh_.dim());

		const Eigen::VectorXd grad_friction = ipc::compute_friction_potential_gradient(
			collision_mesh_, displaced_surface_prev_, compute_displaced_surface(x),
			friction_constraint_set_, epsv_ * dt_);
		gradv = collision_mesh_.to_full_dof(grad_friction);
	}

	void FrictionForm::second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const
	{
		POLYFEM_SCOPED_TIMER("friction hessian");

		assert(displaced_surface_prev_.rows() == collision_mesh_.num_vertices());
		assert(displaced_surface_prev_.cols() == collision_mesh_.dim());

		hessian = ipc::compute_friction_potential_hessian(
			collision_mesh_, displaced_surface_prev_, compute_displaced_surface(x),
			friction_constraint_set_, epsv_ * dt_, project_to_psd_);

		hessian = collision_mesh_.to_full_dof(hessian);
	}

	// TODO: handle lagging with more than one step
	void FrictionForm::init_lagging(const Eigen::VectorXd &x)
	{
		displaced_surface_prev_ = compute_displaced_surface(x);
		update_lagging(x, 0);
	}

	void FrictionForm::update_lagging(const Eigen::VectorXd &x, const int iter_num)
	{
		const Eigen::MatrixXd displaced_surface = compute_displaced_surface(x);

		ipc::Constraints constraint_set;
		constraint_set.use_convergent_formulation = contact_form_.use_convergent_formulation();
		constraint_set.build(
			collision_mesh_, displaced_surface, dhat_,
			/*dmin=*/0, broad_phase_method_);

		ipc::construct_friction_constraint_set(
			collision_mesh_, displaced_surface, constraint_set,
			dhat_, contact_form_.barrier_stiffness(), mu_, friction_constraint_set_);
	}
} // namespace polyfem::solver
