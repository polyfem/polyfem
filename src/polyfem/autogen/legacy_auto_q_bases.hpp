#pragma once

#include "auto_q_bases.hpp"
#include <Eigen/Dense>

namespace polyfem {
namespace autogen {

[[deprecated("Use the Span overload instead.")]]
void q_basis_value_1d(const int q, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val);

[[deprecated("Use the Span overload instead.")]]
void q_grad_basis_value_1d(const int q, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val);

[[deprecated("Use the Span overload instead.")]]
void q_basis_value_2d(const int q, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val);

[[deprecated("Use the Span overload instead.")]]
void q_grad_basis_value_2d(const int q, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val);

[[deprecated("Use the Span overload instead.")]]
void q_basis_value_3d(const int q, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val);

[[deprecated("Use the Span overload instead.")]]
void q_grad_basis_value_3d(const int q, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val);

}}
