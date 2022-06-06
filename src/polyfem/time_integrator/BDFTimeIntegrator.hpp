#pragma once

#include "ImplicitTimeIntegrator.hpp"

namespace polyfem
{
	namespace time_integrator
	{
		class BDFTimeIntegrator : public ImplicitTimeIntegrator
		{
		public:
			BDFTimeIntegrator() {}

			void set_parameters(const nlohmann::json &params) override;

			using ImplicitTimeIntegrator::init;

			void init(const std::vector<Eigen::VectorXd> &x_prevs,
					  const std::vector<Eigen::VectorXd> &v_prevs,
					  const std::vector<Eigen::VectorXd> &a_prevs,
					  double dt);

			void update_quantities(const Eigen::VectorXd &x) override;

			Eigen::VectorXd x_tilde() const override;

			double acceleration_scaling() const override;

		protected:
			int num_steps;

			// https://en.wikipedia.org/wiki/Backward_differentiation_formula#General_formula
			static const std::vector<double> &alphas(const int i);
			static double betas(const int i);
			static const std::vector<double> test_betas;

			Eigen::VectorXd weighted_sum_x_prevs() const;
			Eigen::VectorXd weighted_sum_v_prevs() const;
		};
	} // namespace time_integrator
} // namespace polyfem
