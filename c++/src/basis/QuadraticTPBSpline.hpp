#ifndef QUADRATIC_TP_B_SPLINE_HPP
#define QUADRATIC_TP_B_SPLINE_HPP

#include "QuadraticBSpline.hpp"

#include <vector>
#include <cassert>

#include <Eigen/Dense>


namespace poly_fem {
	class QuadraticTensorProductBSpline {
	public:
		QuadraticTensorProductBSpline()
		{ }

		QuadraticTensorProductBSpline(const std::vector<double> &knots_u, const std::vector<double> &knots_v)
		: spline_u_(knots_u), spline_v_(knots_v)
		{ }
		
		void init(const std::vector<double> &knots_u, const std::vector<double> &knots_v);

		void interpolate(const Eigen::MatrixXd &ts, Eigen::MatrixXd &result) const;
		double interpolate(const double u, const double v) const;

		void derivative(const Eigen::MatrixXd &ts, Eigen::MatrixXd &result) const;
	private:
		QuadraticBSpline spline_u_;
		QuadraticBSpline spline_v_;
	};
}
#endif //QUADRATIC_TP_B_SPLINE_HPP
