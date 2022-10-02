#include "RayleighDampingForm.hpp"

namespace polyfem::solver
{
	RayleighDampingForm::RayleighDampingForm(
		Form &form_to_damp,
		const time_integrator::ImplicitTimeIntegrator &time_integrator,
		const double stiffness_ratio,
		const int n_lagging_iters)
		: form_to_damp_(form_to_damp),
		  time_integrator_(time_integrator),
		  stiffness_ratio_(stiffness_ratio),
		  n_lagging_iters_(n_lagging_iters)
	{
		assert(0 < stiffness_ratio);
		assert(n_lagging_iters_ > 0);
	}

	double RayleighDampingForm::value_unweighted(const Eigen::VectorXd &x) const
	{
		const Eigen::VectorXd v = time_integrator_.compute_velocity(x);
		return 0.5 * stiffness() * (1 / time_integrator_.dv_dx()) * v.transpose() * (lagged_stiffness_matrix_ * v);
	}

	void RayleighDampingForm::first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		const Eigen::VectorXd v = time_integrator_.compute_velocity(x);
		gradv = stiffness() * lagged_stiffness_matrix_ * v;
	}

	void RayleighDampingForm::second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian)
	{
		// NOTE: Assumes that v(x) is linear in x, so ∂²v/∂x² = 0
		hessian = stiffness() * lagged_stiffness_matrix_;
	}

	void RayleighDampingForm::init_lagging(const Eigen::VectorXd &x)
	{
		update_lagging(x, 0);
	}

	void RayleighDampingForm::update_lagging(const Eigen::VectorXd &x, const int iter_num)
	{
		form_to_damp_.second_derivative(x, lagged_stiffness_matrix_);
	}

	double RayleighDampingForm::stiffness() const
	{
		return 0.75 * stiffness_ratio_ * std::pow(time_integrator_.dt(), 3) / form_to_damp_.weight();
	}
} // namespace polyfem::solver
