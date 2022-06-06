#pragma once

#include "ImplicitTimeIntegrator.hpp"

namespace polyfem
{
	namespace time_integrator
	{
		class ImplicitNewmark : public ImplicitTimeIntegrator
		{
		public:
			ImplicitNewmark() {}

			void set_parameters(const nlohmann::json &params) override;

			void update_quantities(const Eigen::VectorXd &x) override;

			Eigen::VectorXd x_tilde() const override;

			double acceleration_scaling() const override;

		protected:
			double gamma = 0.5, beta = 0.25;
		};
	} // namespace time_integrator
} // namespace polyfem
