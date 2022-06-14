#include "LaggedRegForm.hpp"

#include <polyfem/utils/MatrixUtils.hpp>

namespace polyfem
{
	namespace solver
	{
		LaggedRegForm::LaggedRegForm(const double lagged_damping_weight)
			: lagged_damping_weight_(lagged_damping_weight)
		{
			//TODO
			// lagged_damping_weight_ = state.args["solver"]["contact"]["lagged_damping_weight"].get<double>();
		}

		double LaggedRegForm::value(const Eigen::VectorXd &x)
		{
			return lagged_damping_weight_ * (x - x_lagged_).squaredNorm();
		}

		void LaggedRegForm::gradient(const Eigen::VectorXd &x, Eigen::VectorXd &gradv)
		{
			gradv = lagged_damping_weight_ * (x - x_lagged_);
		}

		void LaggedRegForm::hessian(const Eigen::VectorXd &x, StiffnessMatrix &hessian)
		{
			hessian = lagged_damping_weight_ * utils::sparse_identity(x.size(), x.size());
		}

		void LaggedRegForm::update_lagging(const Eigen::VectorXd &x)
		{
			x_lagged_ = x;
		};

	} // namespace solver
} // namespace polyfem