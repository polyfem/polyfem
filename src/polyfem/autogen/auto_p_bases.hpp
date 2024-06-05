#pragma once

#include <Eigen/Dense>
#include "p_n_bases.hpp"
#include <cassert>

namespace polyfem {
namespace autogen {
void p_nodes_2d(const int p, Eigen::MatrixXd &val);

void p_basis_value_2d(const int p, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val);

void p_grad_basis_value_2d(const int p, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val);


void p_nodes_3d(const int p, Eigen::MatrixXd &val);

void p_basis_value_3d(const int p, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val);

void p_grad_basis_value_3d(const int p, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val);



static const int MAX_P_BASES = 4;

}}
