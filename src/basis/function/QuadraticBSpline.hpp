#pragma once

#include <cassert>
#include <array>
#include <Eigen/Dense>

namespace polyfem
{
	namespace basis
	{
		class QuadraticBSpline
		{
		public:
			QuadraticBSpline() {}
			QuadraticBSpline(const std::array<double, 4> &knots)
				: knots_(knots)
			{
			}

			void init(const std::array<double, 4> &knots);

			void interpolate(const Eigen::MatrixXd &ts, Eigen::MatrixXd &result) const;
			double interpolate(const double t) const;

			void derivative(const Eigen::MatrixXd &ts, Eigen::MatrixXd &result) const;
			double derivative(const double t) const;

		private:
			std::array<double, 4> knots_;
		};
	} // namespace basis
} // namespace polyfem
