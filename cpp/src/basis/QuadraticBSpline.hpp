#ifndef QUADRATIC_B_SPLINE_HPP
#define QUADRATIC_B_SPLINE_HPP

#include <cassert>
#include <vector>
#include <Eigen/Dense>

namespace poly_fem {
	class QuadraticBSpline
	{
	public:
		QuadraticBSpline() { }
		QuadraticBSpline(const std::vector<double> &knots)
		: knots_(knots)
		{ }
		
		void init(const std::vector<double> &knots);

		void interpolate(const Eigen::MatrixXd &ts, Eigen::MatrixXd &result) const;
		double interpolate(const double t) const;

		void derivative(const Eigen::MatrixXd &ts, Eigen::MatrixXd &result) const;
		double derivative(const double t) const;

	private:
		std::vector<double> knots_;
	};
}
#endif //QUADRATIC_B_SPLINE_HPP
