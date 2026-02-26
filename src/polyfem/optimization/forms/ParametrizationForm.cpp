#include "ParametrizationForm.hpp"
#include <polyfem/State.hpp>

namespace polyfem::solver
{
	void ParametrizationForm::init(const Eigen::VectorXd &x)
	{
		init_with_param(apply_parametrizations(x));
	}

	bool ParametrizationForm::is_step_valid(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const
	{
		return is_step_valid_with_param(apply_parametrizations(x0), apply_parametrizations(x1));
	}

	double ParametrizationForm::max_step_size(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const
	{
		return max_step_size_with_param(apply_parametrizations(x0), apply_parametrizations(x1));
	}

	void ParametrizationForm::line_search_begin(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1)
	{
		line_search_begin_with_param(apply_parametrizations(x0), apply_parametrizations(x1));
	}

	void ParametrizationForm::line_search_end()
	{
		line_search_end_with_param();
	}

	void ParametrizationForm::post_step(const polysolve::nonlinear::PostStepData &data)
	{
		post_step_with_param(polysolve::nonlinear::PostStepData(
			data.iter_num,
			data.solver_info,
			apply_parametrizations(data.x),
			parametrizations_.apply_jacobian(data.grad, data.x)));
	}

	double ParametrizationForm::value_unweighted(const Eigen::VectorXd &x) const
	{
		Eigen::VectorXd y = apply_parametrizations(x);
		return value_unweighted_with_param(y);
	}

	void ParametrizationForm::solution_changed(const Eigen::VectorXd &new_x)
	{
		solution_changed_with_param(apply_parametrizations(new_x));
	}

	void ParametrizationForm::compute_partial_gradient(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		compute_partial_gradient_with_param(apply_parametrizations(x), gradv);
		gradv = parametrizations_.apply_jacobian(gradv, x);
	}

	bool ParametrizationForm::is_step_collision_free(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const
	{
		return is_step_collision_free_with_param(apply_parametrizations(x0), apply_parametrizations(x1));
	}
} // namespace polyfem::solver