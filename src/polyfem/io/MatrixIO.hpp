#pragma once

#include <polyfem/Common.hpp>
#include <polyfem/utils/Types.hpp>

#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace polyfem::io
{
	/// Reads a matrix from a file. Determines the file format based on the path's extension.
	template <typename T>
	bool read_matrix(const std::string &path, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &mat);

	/// Writes a matrix to a file. Determines the file format based on the path's extension.
	template <typename Mat>
	bool write_matrix(const std::string &path, const Mat &mat);

	/// Writes a matrix to a hdf5 file using key as name.
	template <typename Mat>
	bool write_matrix(const std::string &path, const std::string &key, const Mat &mat, const bool replace = true);

	/// Reads a matrix to a hdf5 file using key as name.
	template <typename Mat>
	bool read_matrix(const std::string &path, const std::string &key, Mat &mat);

	template <typename T>
	bool read_matrix_ascii(const std::string &path, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &mat);

	template <typename Mat>
	bool write_matrix_ascii(const std::string &path, const Mat &mat);

	template <typename T>
	bool read_matrix_binary(const std::string &path, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &mat);

	template <typename Mat>
	bool write_matrix_binary(const std::string &path, const Mat &mat);

	bool write_sparse_matrix_csv(const std::string &path, const Eigen::SparseMatrix<double> &mat);

	template <typename T>
	bool import_matrix(const std::string &path, const json &import, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &mat);
} // namespace polyfem::io
