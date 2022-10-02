#include "RayleighDampingForm.hpp"

namespace polyfem::solver
{
	RayleighDampingForm::RayleighDampingForm(
		ElasticForm &elastic_form,
		const time_integrator::ImplicitTimeIntegrator &time_integrator,
		const double stiffness_ratio,
		const int n_lagging_iters)
		: elastic_form_(elastic_form),
		  time_integrator_(time_integrator),
		  stiffness_ratio_(stiffness_ratio),
		  n_lagging_iters_(n_lagging_iters)
	{
		assert(0 < stiffness_ratio && stiffness_ratio < 1);
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
		// NOTE: Assumes that v(x) is linear in x
		hessian = stiffness() * lagged_stiffness_matrix_;
	}

	void RayleighDampingForm::init_lagging(const Eigen::VectorXd &x)
	{
		update_lagging(x, 0);
	}

	void RayleighDampingForm::update_lagging(const Eigen::VectorXd &x, const int iter_num)
	{
		elastic_form_.second_derivative_unweighted(x, lagged_stiffness_matrix_);
	}
} // namespace polyfem::solver
