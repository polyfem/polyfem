#include "GrowthDissipationForm.hpp"

#include <polyfem/utils/Logger.hpp>

#include <algorithm>

namespace polyfem::solver
{
	GrowthDissipationForm::GrowthDissipationForm(
		const Eigen::VectorXd &weights,
		const Eigen::VectorXd &theta_n,
		const double floor_fraction)
		: weights_(weights), theta_n_(theta_n)
	{
		if (weights_.size() != theta_n_.size())
			log_and_throw_error("GrowthDissipationForm: {} weights for {} growth DOFs.", weights_.size(), theta_n_.size());
		if (!(floor_fraction > 0 && floor_fraction < 1))
			log_and_throw_error("GrowthDissipationForm: floor_fraction must lie in (0, 1), got {}.", floor_fraction);
		if ((theta_n_.array() <= 0).any())
			log_and_throw_error("GrowthDissipationForm: theta_n must be positive (min {}).", theta_n_.minCoeff());
		floor_ = floor_fraction * theta_n_;
	}

	double GrowthDissipationForm::value_unweighted(const Eigen::VectorXd &x) const
	{
		assert(x.size() == weights_.size());
		return weights_.dot(x - theta_n_);
	}

	void GrowthDissipationForm::first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		assert(x.size() == weights_.size());
		gradv = weights_;
	}

	void GrowthDissipationForm::second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const
	{
		hessian.resize(x.size(), x.size());
		hessian.setZero();
	}

	double GrowthDissipationForm::max_step_size(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const
	{
		// Largest alpha in (0, 1] with x0 + alpha (x1 - x0) >= floor. A DOF
		// already below its floor (only possible from outside data) is ignored
		// here and rejected by is_step_valid; returning 0 would abort the line
		// search outright.
		double alpha = 1.0;
		for (int i = 0; i < x0.size(); ++i)
		{
			const double dx = x1(i) - x0(i);
			if (dx >= 0 || x0(i) <= floor_(i))
				continue;
			alpha = std::min(alpha, (floor_(i) - x0(i)) / dx * 0.99);
		}
		return std::max(alpha, 0.0);
	}

	bool GrowthDissipationForm::is_step_valid(const Eigen::VectorXd &, const Eigen::VectorXd &x1) const
	{
		return (x1.array() > 0).all();
	}
} // namespace polyfem::solver
