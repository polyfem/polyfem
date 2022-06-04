#include "QuadraticBSpline.hpp"

#include <cmath>

namespace polyfem
{
	namespace basis
	{
		void QuadraticBSpline::init(const std::array<double, 4> &knots)
		{
			knots_ = knots;
		}

		void QuadraticBSpline::interpolate(const Eigen::MatrixXd &ts, Eigen::MatrixXd &result) const
		{
			result.resize(ts.size(), 1);

			for (long i = 0; i < ts.size(); ++i)
				result(i) = interpolate(ts(i));
		}

		double QuadraticBSpline::interpolate(const double t) const
		{
			if (t >= knots_[0] && t < knots_[1])
				return (t - knots_[0]) * (t - knots_[0]) / (knots_[2] - knots_[0]) / (knots_[1] - knots_[0]);

			if (t >= knots_[1] && t < knots_[2])
				return (t - knots_[0]) / (knots_[2] - knots_[0]) * (knots_[2] - t) / (knots_[2] - knots_[1]) + (knots_[3] - t) / (knots_[3] - knots_[1]) * (t - knots_[1]) / (knots_[2] - knots_[1]);

			if (t >= knots_[2] && t <= knots_[3])
			{
				if (fabs(knots_[3] - knots_[2]) < 1e-12 && fabs(knots_[3] - knots_[1]) < 1e-12)
					return knots_[3];
				if (fabs(knots_[3] - knots_[2]) < 1e-12)
					return 0;

				return (knots_[3] - t) * (knots_[3] - t) / (knots_[3] - knots_[1]) / (knots_[3] - knots_[2]);
			}

			return 0;
		}

		void QuadraticBSpline::derivative(const Eigen::MatrixXd &ts, Eigen::MatrixXd &result) const
		{
			result.resize(ts.size(), 1);

			for (long i = 0; i < ts.size(); ++i)
				result(i) = derivative(ts(i));
		}

		double QuadraticBSpline::derivative(const double t) const
		{
			if (t >= knots_[0] && t < knots_[1])
				return 2 / (knots_[2] - knots_[0]) * (t - knots_[0]) / (knots_[1] - knots_[0]);

			if (t >= knots_[1] && t < knots_[2])
				return ((-2 * t + 2 * knots_[0]) * knots_[1] + (2 * t - 2 * knots_[3]) * knots_[2] - 2 * t * (knots_[0] - knots_[3])) / (-knots_[2] + knots_[0]) / (-knots_[2] + knots_[1]) / (-knots_[3] + knots_[1]);

			if (t >= knots_[2] && t <= knots_[3])
			{
				if (fabs(knots_[3] - knots_[2]) < 1e-12 && fabs(knots_[3] - knots_[1]) < 1e-12)
					return 2 * knots_[3];
				if (fabs(knots_[3] - knots_[2]) < 1e-12)
					return -2 * knots_[3];

				return (2 * t - 2 * knots_[3]) / (-knots_[3] + knots_[1]) / (-knots_[3] + knots_[2]);
			}

			return 0;
		}
	} // namespace basis
} // namespace polyfem
