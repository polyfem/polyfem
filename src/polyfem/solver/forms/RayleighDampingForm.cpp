#include "RayleighDampingForm.hpp"

#include <polyfem/utils/Logger.hpp>

namespace polyfem::solver
{
	RayleighDampingForm::RayleighDampingForm(
		Form &form_to_damp,
		const time_integrator::ImplicitTimeIntegrator &time_integrator,
		const bool use_stiffness_as_ratio,
		const double stiffness,
		const int n_lagging_iters)
		: form_to_damp_(form_to_damp),
		  time_integrator_(time_integrator),
		  use_stiffness_as_ratio_(use_stiffness_as_ratio),
		  stiffness_(stiffness),
		  n_lagging_iters_(n_lagging_iters)
	{
		assert(0 < stiffness);
		assert(n_lagging_iters_ > 0);
	}

	std::shared_ptr<RayleighDampingForm> RayleighDampingForm::create(
		const json &params,
		const std::unordered_map<std::string, std::shared_ptr<Form>> &forms,
		const time_integrator::ImplicitTimeIntegrator &time_integrator)
	{
		const std::string form_name = params["form"];
		if (forms.find(form_name) == forms.end())
			log_and_throw_error("Unknown form to damp: {}", form_name);

		std::shared_ptr<Form> form_to_damp = forms.at(form_name);
		if (form_to_damp == nullptr)
			log_and_throw_error("Cannot use Rayleigh damping on {0} form because {0} is disabled", form_name);

		const bool use_stiffness_as_ratio = params.contains("stiffness_ratio");
		const double stiffness = use_stiffness_as_ratio ? params["stiffness_ratio"] : params["stiffness"];

		return std::make_shared<RayleighDampingForm>(
			*form_to_damp, time_integrator, use_stiffness_as_ratio, stiffness,
			params["lagging_iterations"]);
	}

	double RayleighDampingForm::value_unweighted(const Eigen::VectorXd &x) const
	{
		const Eigen::VectorXd v = time_integrator_.compute_velocity(x);
		return 0.5 * stiffness() / time_integrator_.dv_dx() * double(v.transpose() * lagged_stiffness_matrix_ * v);
	}

	void RayleighDampingForm::first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		const Eigen::VectorXd v = time_integrator_.compute_velocity(x);
		gradv = stiffness() * (lagged_stiffness_matrix_ * v);
	}

	void RayleighDampingForm::second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const
	{
		// NOTE: Assumes that v(x) is linear in x, so ∂²v/∂x² = 0
		hessian = (stiffness() * time_integrator_.dv_dx()) * lagged_stiffness_matrix_;
	}

	void RayleighDampingForm::init_lagging(const Eigen::VectorXd &x)
	{
		update_lagging(x, 0);
	}

	void RayleighDampingForm::update_lagging(const Eigen::VectorXd &x, const int iter_num)
	{
		form_to_damp_.second_derivative(x, lagged_stiffness_matrix_);
		// Divide by form_to_damp_.weight() to cancel out the weighting in form_to_damp_.second_derivative
		lagged_stiffness_matrix_ /= form_to_damp_.weight();
	}

	double RayleighDampingForm::stiffness() const
	{
		if (use_stiffness_as_ratio_)
			return 0.75 * stiffness_ * std::pow(time_integrator_.dt(), 3);
		else
			return stiffness_;
	}
} // namespace polyfem::solver
