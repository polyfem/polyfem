#pragma once

#include <Eigen/Dense>

namespace polyfem
{
	namespace autogen
	{
		template <int dim>
		void generate_gradient_templated(const double c1, const double c2, const double c3, const double d1, const Eigen::Matrix<double, dim, dim> &def_grad, Eigen::Matrix<double, dim, dim> &gradient) {}
		template <>
		void generate_gradient_templated(const double c1, const double c2, const double c3, const double d1, const Eigen::Matrix<double, 2, 2> &def_grad, Eigen::Matrix<double, 2, 2> &gradient);
		template <>
		void generate_gradient_templated(const double c1, const double c2, const double c3, const double d1, const Eigen::Matrix<double, 3, 3> &def_grad, Eigen::Matrix<double, 3, 3> &gradient);
		template <int dim>
		void generate_hessian_templated(const double c1, const double c2, const double c3, const double d1, const Eigen::Matrix<double, dim, dim> &def_grad, Eigen::Matrix<double, dim * dim, dim * dim> &hessian) {}
		template <>
		void generate_hessian_templated(const double c1, const double c2, const double c3, const double d1, const Eigen::Matrix<double, 2, 2> &def_grad, Eigen::Matrix<double, 4, 4> &hessian);
		template <>
		void generate_hessian_templated(const double c1, const double c2, const double c3, const double d1, const Eigen::Matrix<double, 3, 3> &def_grad, Eigen::Matrix<double, 9, 9> &hessian);
		void generate_gradient(const double c1, const double c2, const double c3, const double d1, const Eigen::MatrixXd &def_grad, Eigen::MatrixXd &gradient);
		void generate_hessian(const double c1, const double c2, const double c3, const double d1, const Eigen::MatrixXd &def_grad, Eigen::MatrixXd &hessian);
	} // namespace autogen
} // namespace polyfem
