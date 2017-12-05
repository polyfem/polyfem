#include "QuadraticBSpline.hpp"

#include <cmath>

namespace poly_fem {
	void QuadraticBSpline::init(const std::vector<double> &knots)
	{
		knots_ = knots;
		assert(knots_.size() == 4);
	}

	void QuadraticBSpline::interpolate(const Eigen::MatrixXd &ts, Eigen::MatrixXd &result) const
	{
		result.resize(ts.size(), 1);

		for(long i = 0; i < ts.size(); ++i)
			result(i) = interpolate(ts(i));
	}

	double QuadraticBSpline::interpolate(const double t) const
	{
		if(t >= knots_[0] && t < knots_[1])
			return (t - knots_[0]) * (t - knots_[0]) / (knots_[2] - knots_[0]) / (knots_[1] - knots_[0]);

		if(t >= knots_[1] && t < knots_[2])
			return (t - knots_[0]) / (knots_[2] - knots_[0]) * (knots_[2] - t) / (knots_[2] - knots_[1]) + (knots_[3] - t) / (knots_[3] - knots_[1]) * (t - knots_[1]) / (knots_[2] - knots_[1]);

		if(t >= knots_[2] && t < knots_[3])
			return (knots_[3] - t) * (knots_[3] - t) / (knots_[3] - knots_[1]) / (knots_[3] - knots_[2]);

		return 0;
	}

	void QuadraticBSpline::derivative(const Eigen::MatrixXd &ts, Eigen::MatrixXd &result) const
	{
		result.resize(ts.size(), 1);

		for(long i = 0; i < ts.size(); ++i)
			result(i) = derivative(ts(i));
	}

	double QuadraticBSpline::derivative(const double t) const
	{
		if(t >= knots_[0] && t < knots_[1])
			return 2 / (knots_[2] - knots_[0]) * (t - knots_[0]) / (knots_[1] - knots_[0]);

		if(t >= knots_[1] && t < knots_[2])
			return ((-2 * t + 2 * knots_[0]) * knots_[1] + (2 * t - 2 * knots_[3]) * knots_[2] - 2 * t * (knots_[0] - knots_[3])) / (-knots_[2] + knots_[0]) / (-knots_[2] + knots_[1]) / (-knots_[3] + knots_[1]);

		if(t >= knots_[2] && t < knots_[3])
			return (2 * t - 2 * knots_[3]) / (-knots_[3] + knots_[1]) / (-knots_[3] + knots_[2]);

		return 0;
	}
}


