#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace polyfem
{
	// Stack-allocated vectors of size either 2 or 3
	template <typename T>
	using VectorN = Eigen::Matrix<T, Eigen::Dynamic, 1, Eigen::ColMajor, 3, 1>;
	using VectorNd = VectorN<double>;
	using VectorNi = VectorN<int>;

	template <typename T>
	using RowVectorN = Eigen::Matrix<T, 1, Eigen::Dynamic, Eigen::RowMajor, 1, 3>;
	using RowVectorNd = RowVectorN<double>;

	template <typename T>
	using MatrixN = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 3, 3>;
	using MatrixNd = MatrixN<double>;

	using FlatMatrixNd = Eigen::Matrix<double, Eigen::Dynamic, 1, Eigen::ColMajor, 9, 1>;

	static constexpr int MAX_QUAD_POINTS = -1;
	using QuadratureVector = Eigen::Matrix<double, Eigen::Dynamic, 1, Eigen::ColMajor, MAX_QUAD_POINTS, 1>;

#ifdef POLYSOLVE_LARGE_INDEX
	using StiffnessMatrix = Eigen::SparseMatrix<double, Eigen::ColMajor, std::ptrdiff_t>;
#else
	using StiffnessMatrix = Eigen::SparseMatrix<double, Eigen::ColMajor>;
#endif
} // namespace polyfem
