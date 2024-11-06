#pragma once

#include <polyfem/utils/Types.hpp>

#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace polyfem
{
	namespace utils
	{
		// Show some stats about the matrix M: det, singular values, condition number, etc
		void show_matrix_stats(const Eigen::MatrixXd &M);

		template <typename T>
		T matrix_inner_product(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &A, const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &B)
		{ 
			return (A.array() * B.array()).sum(); 
		}

		template <typename T, int rows, int cols, int option, int maxRow, int maxCol>
		T determinant(const Eigen::Matrix<T, rows, cols, option, maxRow, maxCol> &mat)
		{
			assert(mat.rows() == mat.cols());

			if (mat.rows() == 1)
				return mat(0);
			else if (mat.rows() == 2)
				return mat(0, 0) * mat(1, 1) - mat(0, 1) * mat(1, 0);
			else if (mat.rows() == 3)
				return mat(0, 0) * (mat(1, 1) * mat(2, 2) - mat(1, 2) * mat(2, 1)) - mat(0, 1) * (mat(1, 0) * mat(2, 2) - mat(1, 2) * mat(2, 0)) + mat(0, 2) * (mat(1, 0) * mat(2, 1) - mat(1, 1) * mat(2, 0));

			assert(false);
			return T(0);
		}

		template <typename T>
		Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> inverse(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> &mat)
		{
			assert(mat.rows() == mat.cols());

			if (mat.rows() == 1)
			{
				Eigen::Matrix<T, 1, 1> inv;
				inv(0, 0) = T(1.) / mat(0);

				return inv;
			}
			else if (mat.rows() == 2)
			{
				Eigen::Matrix<T, 2, 2> inv;
				T det = determinant(mat);
				inv(0, 0) = mat(1, 1) / det;
				inv(0, 1) = -mat(0, 1) / det;
				inv(1, 0) = -mat(1, 0) / det;
				inv(1, 1) = mat(0, 0) / det;

				return inv;
			}
			else if (mat.rows() == 3)
			{
				Eigen::Matrix<T, 3, 3> inv;
				T det = determinant(mat);
				inv(0, 0) = (-mat(1, 2) * mat(2, 1) + mat(1, 1) * mat(2, 2)) / det;
				inv(0, 1) = (mat(0, 2) * mat(2, 1) - mat(0, 1) * mat(2, 2)) / det;
				inv(0, 2) = (-mat(0, 2) * mat(1, 1) + mat(0, 1) * mat(1, 2)) / det;
				inv(1, 0) = (mat(1, 2) * mat(2, 0) - mat(1, 0) * mat(2, 2)) / det;
				inv(1, 1) = (-mat(0, 2) * mat(2, 0) + mat(0, 0) * mat(2, 2)) / det;
				inv(1, 2) = (mat(0, 2) * mat(1, 0) - mat(0, 0) * mat(1, 2)) / det;
				inv(2, 0) = (-mat(1, 1) * mat(2, 0) + mat(1, 0) * mat(2, 1)) / det;
				inv(2, 1) = (mat(0, 1) * mat(2, 0) - mat(0, 0) * mat(2, 1)) / det;
				inv(2, 2) = (-mat(0, 1) * mat(1, 0) + mat(0, 0) * mat(1, 1)) / det;

				return inv;
			}

			assert(false);
			Eigen::Matrix<T, 1, 1> inv;
			inv(0, 0) = T(0);
			return inv;
		}

		inline Eigen::SparseMatrix<double> sparse_identity(int rows, int cols)
		{
			Eigen::SparseMatrix<double> I(rows, cols);
			I.setIdentity();
			return I;
		}

		/// Flatten rowwises
		Eigen::VectorXd flatten(const Eigen::MatrixXd &X);

		/// Unflatten rowwises, so every dim elements in x become a row.
		Eigen::MatrixXd unflatten(const Eigen::VectorXd &x, int dim);

		void vector2matrix(const Eigen::VectorXd &vec, Eigen::MatrixXd &mat);

		/// @brief Lump each row of a matrix into the diagonal.
		/// @param M Matrix to lump.
		/// @return Lumped matrix.
		Eigen::SparseMatrix<double> lump_matrix(const Eigen::SparseMatrix<double> &M);

		/// @brief Map a full size matrix to a reduced one by dropping rows and columns.
		/// @param[in] full_size Number of variables in the full system.
		/// @param[in] reduced_size Number of variables in the reduced system.
		/// @param[in] removed_vars Indices of the variables (rows and columns of full) to remove.
		/// @param[in] full Full size matrix.
		/// @param[out] reduced Output reduced size matrix.
		void full_to_reduced_matrix(
			const int full_size,
			const int reduced_size,
			const std::vector<int> &removed_vars,
			const StiffnessMatrix &full,
			StiffnessMatrix &reduced);

		/// @brief Reorder row blocks in a matrix.
		/// @param in Input matrix.
		/// @param in_to_out Mapping from input blocks to output blocks.
		/// @param out_blocks Number of blocks in the output matrix.
		/// @param block_size Size of each block.
		/// @return Reordered matrix.
		Eigen::MatrixXd reorder_matrix(
			const Eigen::MatrixXd &in,
			const Eigen::VectorXi &in_to_out,
			int out_blocks = -1,
			const int block_size = 1);

		/// @brief Undo the reordering of row blocks in a matrix.
		/// @param out Reordered matrix.
		/// @param in_to_out Mapping from input blocks to output blocks.
		/// @param in_blocks Number of blocks in the input matrix.
		/// @param block_size Size of each block.
		/// @return Unreordered matrix.
		Eigen::MatrixXd unreorder_matrix(
			const Eigen::MatrixXd &out,
			const Eigen::VectorXi &in_to_out,
			int in_blocks = -1,
			const int block_size = 1);

		/// @brief Map the entrys of an index matrix to new indices.
		/// @param in Input index matrix.
		/// @param index_mapping Mapping from old to new indices.
		/// @return Mapped index matrix.
		Eigen::MatrixXi map_index_matrix(
			const Eigen::MatrixXi &in,
			const Eigen::VectorXi &index_mapping);

		template <typename DstMat, typename SrcMat>
		void append_rows(DstMat &dst, const SrcMat &src)
		{
			if (src.rows() == 0)
				return;
			if (dst.cols() == 0)
				dst.resize(dst.rows(), src.cols());
			assert(dst.cols() == src.cols());
			dst.conservativeResize(dst.rows() + src.rows(), dst.cols());
			dst.bottomRows(src.rows()) = src;
		}

		template <typename DstMat>
		void append_rows_of_zeros(DstMat &dst, const size_t n_zero_rows)
		{
			assert(dst.cols() > 0);
			if (n_zero_rows == 0)
				return;
			dst.conservativeResize(dst.rows() + n_zero_rows, dst.cols());
			dst.bottomRows(n_zero_rows).setZero();
		}
	} // namespace utils
} // namespace polyfem
