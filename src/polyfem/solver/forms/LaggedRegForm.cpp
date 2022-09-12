#include "LaggedRegForm.hpp"

#include <polyfem/utils/MatrixUtils.hpp>

namespace polyfem::solver
{
	LaggedRegForm::LaggedRegForm()
	{
	}

	double LaggedRegForm::value_unweighted(const Eigen::VectorXd &x) const
	{
		return 0.5 * (x - x_lagged_).squaredNorm();
	}

	void LaggedRegForm::first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		gradv = (x - x_lagged_);
	}

	void LaggedRegForm::second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian)
	{
		hessian.resize(x.size(), x.size());
		hessian.setIdentity();
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