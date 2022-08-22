#include "LaggedRegForm.hpp"

#include <polyfem/utils/MatrixUtils.hpp>

namespace polyfem::solver
{
	LaggedRegForm::LaggedRegForm(const double weight)
		: weight_(weight)
	{
		// TODO:
		// lagged_damping_weight_ = state.args["solver"]["contact"]["lagged_damping_weight"].get<double>();
	}

	double LaggedRegForm::value(const Eigen::VectorXd &x) const
	{
		return 0.5 * weight_ * (x - x_lagged_).squaredNorm();
	}

	void LaggedRegForm::first_derivative(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		gradv = weight_ * (x - x_lagged_);
	}

	void LaggedRegForm::second_derivative(const Eigen::VectorXd &x, StiffnessMatrix &hessian)
	{
		hessian = weight_ * utils::sparse_identity(x.size(), x.size());
	}

	void LaggedRegForm::init_lagging(const Eigen::VectorXd &x)
	{
		update_lagging(x);
	}

	void LaggedRegForm::update_lagging(const Eigen::VectorXd &x)
	{
		x_lagged_ = x;
	}
} // namespace polyfem::solver