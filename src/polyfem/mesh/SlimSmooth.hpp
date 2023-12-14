#pragma once

#include <Eigen/Dense>

namespace polyfem::mesh
{
	bool apply_slim(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, const Eigen::MatrixXd &V_new, Eigen::MatrixXd &V_smooth, const int max_iters = 10);
}