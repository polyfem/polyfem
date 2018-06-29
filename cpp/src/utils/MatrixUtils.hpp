#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace polyfem {

	// Show some stats about the matrix M: det, singular values, condition number, etc
	void show_matrix_stats(const Eigen::MatrixXd &M);


	template<typename T>
	T determinant(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> &mat)
	{
		assert(mat.rows() == mat.cols());

		if(mat.rows() == 1)
			return mat(0);
		else if(mat.rows() == 2)
			return mat(0, 0) * mat(1, 1) - mat(0, 1) * mat(1, 0);
		else if(mat.rows() == 3)
			return mat(0,0)*(mat(1,1)*mat(2,2)-mat(1,2)*mat(2,1))-mat(0,1)*(mat(1,0)*mat(2,2)-mat(1,2)*mat(2,0))+mat(0,2)*(mat(1,0)*mat(2,1)-mat(1,1)*mat(2,0));

		assert(false);
		return T(0);
	}

    template<typename T>
	void read_matrix(const std::string &path, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &mat);

	Eigen::Vector4d compute_specturm(const Eigen::SparseMatrix<double> &mat);

} // namespace polyfem
