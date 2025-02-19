#pragma once

#include <Eigen/Dense>
#include <cassert>

namespace polyfem {
namespace autogen {
void q_grad_basis_value_3d(const int q, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val);


}}
