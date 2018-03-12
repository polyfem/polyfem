#pragma once

#include <Eigen/Dense>

namespace poly_fem
{
	double von_mises_stress_for_stress_tensor(const Eigen::MatrixXd &stress);
}