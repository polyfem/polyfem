#pragma once

#include "ImplicitTimeIntegrator.hpp"

namespace polyfem
{
	namespace time_integrator
	{
		class ImplicitEuler : public ImplicitTimeIntegrator
		{
		public:
			ImplicitEuler() {}

			void update_quantities(const Eigen::VectorXd &x) override;

			Eigen::VectorXd x_tilde() const override;

			double acceleration_scaling() const override;
		};
	} // namespace time_integrator
} // namespace polyfem
