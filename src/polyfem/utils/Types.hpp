#pragma once

#define MAX_QUAD_POINTS -1

#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace polyfem
{
	// Stack-allocated vectors of size either 2 or 3
	typedef Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> VectorNd;
	typedef Eigen::Matrix<double, 1, Eigen::Dynamic, Eigen::RowMajor, 1, 3> RowVectorNd;
	typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 3, 3> MatrixNd;

	typedef Eigen::Matrix<double, Eigen::Dynamic, 1, 0, MAX_QUAD_POINTS, 1> QuadratureVector;

#ifdef POLYSOLVE_LARGE_INDEX
	typedef Eigen::SparseMatrix<double, Eigen::ColMajor, std::ptrdiff_t> StiffnessMatrix;
#else
	typedef Eigen::SparseMatrix<double, Eigen::ColMajor> StiffnessMatrix;
#endif
} // namespace polyfem
