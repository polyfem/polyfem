#pragma once

#include "QuadraticBSpline.hpp"

#include <array>
#include <cassert>

#include <Eigen/Dense>

namespace polyfem
{
	namespace basis
	{
		class QuadraticBSpline2d
		{
		public:
			QuadraticBSpline2d()
			{
			}

			QuadraticBSpline2d(const std::array<double, 4> &knots_u, const std::array<double, 4> &knots_v)
				: spline_u_(knots_u), spline_v_(knots_v)
			{
			}

			void init(const std::array<double, 4> &knots_u, const std::array<double, 4> &knots_v);

			void interpolate(const Eigen::MatrixXd &ts, Eigen::MatrixXd &result) const;
			double interpolate(const double u, const double v) const;

			void derivative(const Eigen::MatrixXd &ts, Eigen::MatrixXd &result) const;

		private:
			QuadraticBSpline spline_u_;
			QuadraticBSpline spline_v_;
		};
	} // namespace basis
} // namespace polyfem
