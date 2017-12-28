#ifndef QUADRATIC_B_SPLINE_3D_HPP
#define QUADRATIC_B_SPLINE_3D_HPP

#include "QuadraticBSpline.hpp"

#include <vector>
#include <cassert>

#include <Eigen/Dense>


namespace poly_fem {
	class QuadraticBSpline3d {
	public:
		QuadraticBSpline3d()
		{ }

		QuadraticBSpline3d(const std::vector<double> &knots_u, const std::vector<double> &knots_v, const std::vector<double> &knots_w)
		: spline_u_(knots_u), spline_v_(knots_v), spline_w_(knots_w)
		{ }

		void init(const std::vector<double> &knots_u, const std::vector<double> &knots_v, const std::vector<double> &knots_w);

		void interpolate(const Eigen::MatrixXd &ts, Eigen::MatrixXd &result) const;
		double interpolate(const double u, const double v, const double w) const;

		void derivative(const Eigen::MatrixXd &ts, Eigen::MatrixXd &result) const;
	private:
		QuadraticBSpline spline_u_;
		QuadraticBSpline spline_v_;
		QuadraticBSpline spline_w_;
	};
}
#endif //QUADRATIC_B_SPLINE_3D_HPP
