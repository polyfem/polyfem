#include "FrictionForm.hpp"
#include "ContactForm.hpp"

#include <polyfem/utils/Timer.hpp>

namespace polyfem::solver
{
	FrictionForm::FrictionForm(const State &state,
							   const double epsv,
							   const double mu,
							   const double dhat,
							   const ipc::BroadPhaseMethod broad_phase_method,
							   const double dt,
							   const ContactForm &contact_form,
							   const int n_lagging_iters)
		: state_(state),
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
		return state_.collision_mesh.vertices(
			state_.boundary_nodes_pos + utils::unflatten(x, state_.mesh->dimension()));
	}

	double FrictionForm::value_unweighted(const Eigen::VectorXd &x) const
	{
		return ipc::compute_friction_potential(
			state_.collision_mesh, displaced_surface_prev_, compute_displaced_surface(x),
			friction_constraint_set_, epsv_ * dt_);
	}
	void FrictionForm::first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		const Eigen::VectorXd grad_friction = ipc::compute_friction_potential_gradient(
			state_.collision_mesh, displaced_surface_prev_, compute_displaced_surface(x),
			friction_constraint_set_, epsv_ * dt_);
		gradv = state_.collision_mesh.to_full_dof(grad_friction);
	}

	void FrictionForm::second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian)
	{
		POLYFEM_SCOPED_TIMER("\t\tfriction hessian");

		hessian = ipc::compute_friction_potential_hessian(
			state_.collision_mesh, displaced_surface_prev_, compute_displaced_surface(x),
			friction_constraint_set_, epsv_ * dt_, project_to_psd_);

		hessian = state_.collision_mesh.to_full_dof(hessian);
	}

	// TODO: hanlde lagging with more than one step
	void FrictionForm::init_lagging(const Eigen::VectorXd &x)
	{
		displaced_surface_prev_ = compute_displaced_surface(x);
		update_lagging(x, 0);
	}

	void FrictionForm::update_lagging(const Eigen::VectorXd &x, const int iter_num)
	{
		// Only update the friction constraints if we are not out of lagging iterations
		if (iter_num >= n_lagging_iters_)
			throw std::runtime_error("max friction lagging iterations exceeded");

		const Eigen::MatrixXd displaced_surface = compute_displaced_surface(x);

		ipc::Constraints constraint_set;
		ipc::construct_constraint_set(
			state_.collision_mesh, displaced_surface, dhat_,
			constraint_set, /*dmin=*/0, broad_phase_method_);

		ipc::construct_friction_constraint_set(
			state_.collision_mesh, displaced_surface, constraint_set,
			dhat_, contact_form_.barrier_stiffness(), mu_, friction_constraint_set_);
	}
} // namespace polyfem::solver
