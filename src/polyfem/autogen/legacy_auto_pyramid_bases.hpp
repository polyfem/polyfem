#pragma once

#include "auto_pyramid_bases.hpp"
#include "auto_pyramid_bases_nodes.hpp"
#include <Eigen/Dense>

namespace polyfem {
namespace autogen {

[[deprecated("Use the Span overload instead.")]]
void pyramid_basis_value_3d(const int pyramid, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val);

[[deprecated("Use the Span overload instead.")]]
void pyramid_grad_basis_value_3d(const int pyramid, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val);

}}
