#pragma once

#include <Eigen/Dense>

namespace poly_fem {

	// Show some stats about the matrix M: det, singular values, condition number, etc
	void show_matrix_stats(const Eigen::MatrixXd &M);

} // namespace poly_fem
