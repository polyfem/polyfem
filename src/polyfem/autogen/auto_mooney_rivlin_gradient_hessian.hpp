#pragma once

#include <Eigen/Dense>

namespace polyfem
{
	namespace autogen
	{
		void generate_gradient_2d(const double c1, const double c2, const double c3, const double d1, const Eigen::Matrix<double, 2, 2> &def_grad, Eigen::Matrix<double, 2, 2> &gradient_temp);
		void generate_gradient_3d(const double c1, const double c2, const double c3, const double d1, const Eigen::Matrix<double, 3, 3> &def_grad, Eigen::Matrix<double, 3, 3> &gradient_temp);
		void generate_hessian_2d(const double c1, const double c2, const double c3, const double d1, const Eigen::Matrix<double, 2, 2> &def_grad, Eigen::Matrix<double, 4, 4> &hessian_temp);
		void generate_hessian_3d(const double c1, const double c2, const double c3, const double d1, const Eigen::Matrix<double, 3, 3> &def_grad, Eigen::Matrix<double, 9, 9> &hessian_temp);
		void generate_gradient(const double c1, const double c2, const double c3, const double d1, const Eigen::MatrixXd &def_grad, Eigen::MatrixXd &gradient_temp);
		void generate_hessian(const double c1, const double c2, const double c3, const double d1, const Eigen::MatrixXd &def_grad, Eigen::MatrixXd &hessian_temp);
	} // namespace autogen
} // namespace polyfem
