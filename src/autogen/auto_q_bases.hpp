#pragma once

#include <Eigen/Dense>

namespace polyfem {
namespace autogen {
void q_nodes_2d(const int q, Eigen::MatrixXd &val);

void q_basis_value_2d(const int q, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val);

void q_grad_basis_value_2d(const int q, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val);


void q_nodes_3d(const int q, Eigen::MatrixXd &val);

void q_basis_value_3d(const int q, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val);

void q_grad_basis_value_3d(const int q, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val);



static const int MAX_Q_BASES = 3;

}}
