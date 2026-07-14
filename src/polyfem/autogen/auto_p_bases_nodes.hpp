#pragma once

#include <Eigen/Dense>

namespace polyfem {
namespace autogen {
void p_nodes_2d(const int p, Eigen::MatrixXd &val);

void p_nodes_3d(const int p, Eigen::MatrixXd &val);

}}
