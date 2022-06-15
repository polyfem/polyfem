#pragma once

#include "Form.hpp"
#include <polyfem/utils/Types.hpp>

namespace polyfem
{
	namespace time_integrator
	{
		class ImplicitTimeIntegrator;
	}

	namespace solver
	{
		class InertiaForm : public Form
		{
		public:
			InertiaForm(const StiffnessMatrix &mass, const time_integrator::ImplicitTimeIntegrator &time_integrator);

			double value(const Eigen::VectorXd &x) override;
			void first_derivative(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) override;
			void second_derivative(const Eigen::VectorXd &x, StiffnessMatrix &hessian) override;

		private:
			const StiffnessMatrix &mass_;
			const time_integrator::ImplicitTimeIntegrator &time_integrator_;
		};
	} // namespace solver
} // namespace polyfem
