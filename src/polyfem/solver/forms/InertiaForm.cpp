#pragma once

#include <polyfem/utils/Types.hpp>

namespace polyfem
{
	namespace solver
	{
		InertiaForm::InertiaForm()
		{
			_time_integrator = time_integrator::ImplicitTimeIntegrator::construct_time_integrator(state.args["time"]["integrator"]);
			_time_integrator->set_parameters(state.args["time"]["BDF"]);
			_time_integrator->set_parameters(state.args["time"]["newmark"]);
		}

		double InertiaForm::value(const Eigen::VectorXd &x)
		{
			const Eigen::VectorXd tmp = x - time_integrator()->x_tilde();
			return 0.5 * (tmp.transpose() * state.mass * tmp) / time_integrator()->acceleration_scaling();
		}

		void InertiaForm::gradient(const Eigen::VectorXd &x, Eigen::VectorXd &gradv)
		{
			return state.mass * full / time_integrator()->acceleration_scaling();
		}

		void InertiaForm::hessian(const Eigen::VectorXd &x, StiffnessMatrix &hessian)
		{
			inertia_hessian = state.mass / time_integrator()->acceleration_scaling();
		}

		void InertiaForm::update_quantities(const double t, const Eigen::VectorXd &x)
		{
			_time_integrator->update_quantities(x);
			this->t = t;
		}

	} // namespace solver
} // namespace polyfem
