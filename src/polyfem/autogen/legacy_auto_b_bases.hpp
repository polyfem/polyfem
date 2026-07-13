#pragma once

#include "auto_b_bases.hpp"
#include <Eigen/Dense>

namespace polyfem {
namespace autogen {

[[deprecated("Use the Span overload instead.")]]
void b_basis_value_2d(const int b, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val);

[[deprecated("Use the Span overload instead.")]]
void b_grad_basis_value_2d(const int b, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val);

[[deprecated("Use the Span overload instead.")]]
void b_basis_value_3d(const int b, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val);

[[deprecated("Use the Span overload instead.")]]
void b_grad_basis_value_3d(const int b, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val);

}}
