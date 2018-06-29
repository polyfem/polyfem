#ifndef QUADRATIC_B_SPLINE_2D_HPP
#define QUADRATIC_B_SPLINE_2D_HPP

#include <polyfem/QuadraticBSpline.hpp>

#include <array>
#include <cassert>

#include <Eigen/Dense>


namespace poly_fem {
	class QuadraticBSpline2d {
	public:
		QuadraticBSpline2d()
		{ }

		QuadraticBSpline2d(const std::array<double, 4> &knots_u, const std::array<double, 4> &knots_v)
		: spline_u_(knots_u), spline_v_(knots_v)
		{ }
		
		void init(const std::array<double, 4> &knots_u, const std::array<double, 4> &knots_v);

		void interpolate(const Eigen::MatrixXd &ts, Eigen::MatrixXd &result) const;
		double interpolate(const double u, const double v) const;

		void derivative(const Eigen::MatrixXd &ts, Eigen::MatrixXd &result) const;
	private:
		QuadraticBSpline spline_u_;
		QuadraticBSpline spline_v_;
	};
}
#endif //QUADRATIC_B_SPLINE_2D_HPP
