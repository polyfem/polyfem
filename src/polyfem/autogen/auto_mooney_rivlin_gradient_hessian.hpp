#pragma once

#include <Eigen/Dense>

namespace polyfem
{
	namespace autogen
	{
		template <int dim>
		void generate_gradient_(const double c1, const double c2, const double c3, const double d1, const Eigen::Matrix<double, dim, dim> &def_grad, Eigen::Matrix<double, dim, dim> &gradient_temp) {}
		template<>
		void generate_gradient_(const double c1, const double c2, const double c3, const double d1, const Eigen::Matrix<double, 2, 2> &def_grad, Eigen::Matrix<double, 2, 2> &gradient_temp);
		template<>
		void generate_gradient_(const double c1, const double c2, const double c3, const double d1, const Eigen::Matrix<double, 3, 3> &def_grad, Eigen::Matrix<double, 3, 3> &gradient_temp);
		template <int dim>
		void generate_hessian_(const double c1, const double c2, const double c3, const double d1, const Eigen::Matrix<double, dim, dim> &def_grad, Eigen::Matrix<double, dim*dim, dim*dim> &hessian_temp) {}
		template <>
		void generate_hessian_(const double c1, const double c2, const double c3, const double d1, const Eigen::Matrix<double, 2, 2> &def_grad, Eigen::Matrix<double, 4, 4> &hessian_temp);
		template <>
		void generate_hessian_(const double c1, const double c2, const double c3, const double d1, const Eigen::Matrix<double, 3, 3> &def_grad, Eigen::Matrix<double, 9, 9> &hessian_temp);
		void generate_gradient(const double c1, const double c2, const double c3, const double d1, const Eigen::MatrixXd &def_grad, Eigen::MatrixXd &gradient_temp);
		void generate_hessian(const double c1, const double c2, const double c3, const double d1, const Eigen::MatrixXd &def_grad, Eigen::MatrixXd &hessian_temp);
	} // namespace autogen
} // namespace polyfem
