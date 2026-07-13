#pragma once

#include "auto_p_bases.hpp"
#include "auto_p_bases_nodes.hpp"
#include <Eigen/Dense>

namespace polyfem {
namespace autogen {

[[deprecated("Use the Span overload instead.")]]
void p_basis_value_2d(const bool bernstein, const int p, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val);

[[deprecated("Use the Span overload instead.")]]
void p_grad_basis_value_2d(const bool bernstein, const int p, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val);

[[deprecated("Use the Span overload instead.")]]
void p_basis_value_3d(const bool bernstein, const int p, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val);

[[deprecated("Use the Span overload instead.")]]
void p_grad_basis_value_3d(const bool bernstein, const int p, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val);

}}
