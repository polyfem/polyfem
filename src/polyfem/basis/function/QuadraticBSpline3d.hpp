#pragma once

#include "QuadraticBSpline.hpp"

#include <array>
#include <cassert>

#include <Eigen/Dense>

namespace polyfem
{
	namespace basis
	{
		class QuadraticBSpline3d
		{
		public:
			QuadraticBSpline3d()
			{
			}

			QuadraticBSpline3d(const std::array<double, 4> &knots_u, const std::array<double, 4> &knots_v, const std::array<double, 4> &knots_w)
				: spline_u_(knots_u), spline_v_(knots_v), spline_w_(knots_w)
			{
			}

			void init(const std::array<double, 4> &knots_u, const std::array<double, 4> &knots_v, const std::array<double, 4> &knots_w);

			void interpolate(const Eigen::MatrixXd &ts, Eigen::MatrixXd &result) const;
			double interpolate(const double u, const double v, const double w) const;

			void derivative(const Eigen::MatrixXd &ts, Eigen::MatrixXd &result) const;

		private:
			QuadraticBSpline spline_u_;
			QuadraticBSpline spline_v_;
			QuadraticBSpline spline_w_;
		};
	} // namespace basis
} // namespace polyfem
