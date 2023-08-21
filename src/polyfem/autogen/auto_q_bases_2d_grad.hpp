#pragma once

#include <Eigen/Dense>

namespace polyfem {
namespace autogen {
void q_grad_basis_value_2d(const int q, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val);


}}
