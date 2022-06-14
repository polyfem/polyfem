#include "QuadraticBSpline3d.hpp"

namespace polyfem
{
	namespace basis
	{
		void QuadraticBSpline3d::init(const std::array<double, 4> &knots_u, const std::array<double, 4> &knots_v, const std::array<double, 4> &knots_w)
		{
			spline_u_.init(knots_u);
			spline_v_.init(knots_v);
			spline_w_.init(knots_w);
		}

		void QuadraticBSpline3d::interpolate(const Eigen::MatrixXd &ts, Eigen::MatrixXd &result) const
		{
			const int n_t = int(ts.rows());
			assert(ts.cols() == 3);

			result.resize(n_t, 1);

			for (int i = 0; i < n_t; ++i)
				result(i) = interpolate(ts(i, 0), ts(i, 1), ts(i, 2));
		}

		double QuadraticBSpline3d::interpolate(const double u, const double v, const double w) const
		{
			return spline_u_.interpolate(u) * spline_v_.interpolate(v) * spline_w_.interpolate(w);
		}

		void QuadraticBSpline3d::derivative(const Eigen::MatrixXd &ts, Eigen::MatrixXd &result) const
		{
			const int n_t = int(ts.rows());
			assert(ts.cols() == 3);

			result.resize(n_t, 3);

			for (int i = 0; i < n_t; ++i)
			{
				const double u = ts(i, 0);
				const double v = ts(i, 1);
				const double w = ts(i, 2);

				result(i, 0) = spline_u_.derivative(u) * spline_v_.interpolate(v) * spline_w_.interpolate(w);
				result(i, 1) = spline_u_.interpolate(u) * spline_v_.derivative(v) * spline_w_.interpolate(w);
				result(i, 2) = spline_u_.interpolate(u) * spline_v_.interpolate(v) * spline_w_.derivative(w);
			}
		}
	} // namespace basis
} // namespace polyfem
