#include "LaggedRegForm.hpp"

#include <polyfem/utils/MatrixUtils.hpp>

#include <polyfem/utils/Logger.hpp>

namespace polyfem::solver
{
	LaggedRegForm::LaggedRegForm(const int n_lagging_iters)
		: n_lagging_iters_(n_lagging_iters < 0 ? std::numeric_limits<int>::max() : n_lagging_iters)
	{
		if (n_lagging_iters_ < 1)
			disable();
	}

	double LaggedRegForm::value_unweighted(const Eigen::VectorXd &x) const
	{
		return 0.5 * (x - x_lagged_).squaredNorm();
	}

	void LaggedRegForm::first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		gradv = (x - x_lagged_);
	}

	void LaggedRegForm::second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const
	{
		hessian.resize(x.size(), x.size());
		hessian.setIdentity();
	}

	void LaggedRegForm::init_lagging(const Eigen::VectorXd &x)
	{
		update_lagging(x, 0);
	}

	void LaggedRegForm::update_lagging(const Eigen::VectorXd &x, const int iter_num)
	{
		x_lagged_ = x;

		const bool enabled_before = enabled();
		set_enabled(iter_num >= 0 && iter_num < n_lagging_iters_);
		if (!enabled_before && enabled())
			logger().debug("Enabling lagged regularization");
		else if (enabled_before && !enabled())
			logger().debug("Disabling lagged regularization");
	}
} // namespace polyfem::solver