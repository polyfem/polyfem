#pragma once

#define MAX_QUAD_POINTS 216

#include <Eigen/Dense>

namespace poly_fem
{
	// Stack-allocated vectors of size either 2 or 3
	typedef Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> VectorNd;
	typedef Eigen::Matrix<double, 1, Eigen::Dynamic, Eigen::RowMajor, 1, 3> RowVectorNd;

	typedef Eigen::Matrix<double, Eigen::Dynamic, 1, 0, MAX_QUAD_POINTS, 1> QuadratureVector;
}

