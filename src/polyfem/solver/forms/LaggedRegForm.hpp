#pragma once

#include "Form.hpp"

#include <polyfem/utils/Types.hpp>

namespace polyfem
{
	namespace solver
	{
		class LaggedRegForm : public Form
		{
		public:
			LaggedRegForm(const double lagged_damping_weight);

			double value(const Eigen::VectorXd &x) override;
			void gradient(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) override;
			void hessian(const Eigen::VectorXd &x, StiffnessMatrix &hessian) override;

			void update_lagging(const Eigen::VectorXd &x) override;

		private:
			Eigen::VectorXd x_lagged_; ///< @brief The full variables from the previous lagging solve.

			const double lagged_damping_weight_;
		};
	} // namespace solver
} // namespace polyfem
