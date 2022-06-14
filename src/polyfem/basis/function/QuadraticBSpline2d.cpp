#include "QuadraticBSpline2d.hpp"

namespace polyfem
{
	namespace basis
	{
		void QuadraticBSpline2d::init(const std::array<double, 4> &knots_u, const std::array<double, 4> &knots_v)
		{
			spline_u_.init(knots_u);
			spline_v_.init(knots_v);
		}

		void QuadraticBSpline2d::interpolate(const Eigen::MatrixXd &ts, Eigen::MatrixXd &result) const
		{
			const int n_t = int(ts.rows());
			assert(ts.cols() == 2);

			result.resize(n_t, 1);

			for (int i = 0; i < n_t; ++i)
				result(i) = interpolate(ts(i, 0), ts(i, 1));
		}

		double QuadraticBSpline2d::interpolate(const double u, const double v) const
		{
			return spline_u_.interpolate(u) * spline_v_.interpolate(v);
		}

		void QuadraticBSpline2d::derivative(const Eigen::MatrixXd &ts, Eigen::MatrixXd &result) const
		{
			const int n_t = int(ts.rows());
			assert(ts.cols() == 2);

			result.resize(n_t, 2);

			for (int i = 0; i < n_t; ++i)
			{
				const double u = ts(i, 0);
				const double v = ts(i, 1);

				result(i, 0) = spline_u_.derivative(u) * spline_v_.interpolate(v);
				result(i, 1) = spline_u_.interpolate(u) * spline_v_.derivative(v);
			}
		}
	} // namespace basis
} // namespace polyfem
