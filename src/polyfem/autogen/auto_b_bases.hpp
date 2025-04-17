#pragma once

#include <Eigen/Dense>
#include <cassert>

namespace polyfem {
namespace autogen {
void b_basis_value_2d(const int b, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val);

void b_grad_basis_value_2d(const int b, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val);


void b_basis_value_3d(const int b, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val);

void b_grad_basis_value_3d(const int b, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val);



static const int MAX_B_BASES = 4;

}}
