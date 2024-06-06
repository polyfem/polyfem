#pragma once
#include <Eigen/Dense>
#include <iostream>
#include <cassert>

namespace polyfem
{
	namespace autogen
	{
		Eigen::Vector3i convert_local_index_to_ijk(const int local_index, const int p);
		Eigen::ArrayXd P(const int m, const int p, const Eigen::ArrayXd &z);
		Eigen::ArrayXd P_prime(const int m, const int p, const Eigen::ArrayXd &z);
		void p_n_nodes_2d(const int p, Eigen::MatrixXd &val);
		void p_n_basis_value_2d(const int p, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val);
		void p_n_basis_grad_value_2d(const int p, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val);
		void p_n_nodes_3d(const int p, Eigen::MatrixXd &val);
		void p_n_basis_value_3d(const int p, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val);
		void p_n_basis_grad_value_3d(const int p, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val);
	} // namespace autogen
} // namespace polyfem