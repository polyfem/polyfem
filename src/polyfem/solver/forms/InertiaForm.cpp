#include "InertiaForm.hpp"

#include <polyfem/time_integrator/ImplicitTimeIntegrator.hpp>

namespace polyfem
{
	namespace solver
	{
		InertiaForm::InertiaForm(const StiffnessMatrix &mass, std::shared_ptr<time_integrator::ImplicitTimeIntegrator> time_integrator)
			: mass_(mass), time_integrator_(time_integrator)
		{
			//TODO
			// time_integrator_ = time_integrator::ImplicitTimeIntegrator::construct_time_integrator(state.args["time"]["integrator"]);
			// time_integrator_->set_parameters(state.args["time"]["BDF"]);
			// time_integrator_->set_parameters(state.args["time"]["newmark"]);
		}

		double InertiaForm::value(const Eigen::VectorXd &x)
		{
			const Eigen::VectorXd tmp = x - time_integrator_->x_tilde();
			const double prod = tmp.transpose() * mass_ * tmp;
			const double energy = 0.5 * prod / time_integrator_->acceleration_scaling();
			return energy;
		}

		void InertiaForm::gradient(const Eigen::VectorXd &x, Eigen::VectorXd &gradv)
		{
			gradv = (mass_ * (x - time_integrator_->x_tilde())) / time_integrator_->acceleration_scaling();
		}

		void InertiaForm::hessian(const Eigen::VectorXd &x, StiffnessMatrix &hessian)
		{
			hessian = mass_ / time_integrator_->acceleration_scaling();
		}

		void InertiaForm::update_quantities(const double, const Eigen::VectorXd &x)
		{
			time_integrator_->update_quantities(x);
		}

	} // namespace solver
} // namespace polyfem
