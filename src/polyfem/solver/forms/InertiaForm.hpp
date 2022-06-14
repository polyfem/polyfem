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
			InertiaForm(const StiffnessMatrix &mass, std::shared_ptr<time_integrator::ImplicitTimeIntegrator> time_integrator);

			double value(const Eigen::VectorXd &x) override;
			void gradient(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) override;
			void hessian(const Eigen::VectorXd &x, StiffnessMatrix &hessian) override;

			void update_quantities(const double t, const Eigen::VectorXd &x) override;

		private:
			const StiffnessMatrix &mass_;
			std::shared_ptr<time_integrator::ImplicitTimeIntegrator> time_integrator_;
		};
	} // namespace solver
} // namespace polyfem
