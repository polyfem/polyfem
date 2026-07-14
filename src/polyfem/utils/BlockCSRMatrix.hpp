#pragma once

#include <polyfem/utils/Span.hpp>
#include <polyfem/utils/CudaBoth.hpp>
#include <polyfem/utils/ExecutionPolicy.hpp>
#include <polyfem/utils/Types.hpp>

#include <Eigen/SparseCore>

#include <vector>
#include <unordered_set>
#include <cstdint>
#include <cstddef>
#include <cassert>

#ifdef POLYFEM_WITH_CUDA
#include <polyfem/utils/CudaUtils.hpp>
#endif

namespace polyfem
{

	struct BSRSparsityPattern
	{
		int rows;      //< Scalar row num.
		int cols;      //< Scalar col num.
		int block_dim; //< Block dimension. Can be 1, 2, or 3.

		/// Scalar non zeros packed (row, col).
		std::unordered_set<uint64_t> non_zeros;

		void insert(uint32_t row, uint32_t col);
		void join(const BSRSparsityPattern &other);
	};

	struct BSRMatrixMutableView
	{
		int rows;      //< Scalar row num.
		int cols;      //< Scalar col num.
		int block_dim; //< Block dimension. Can be 1, 2, or 3.

		Span<const int> row_ptr; //< BSR row ptr.
		Span<const int> col_idx; //< BSR col index.
		/// BSR values. each non zero is a row-major (block_dim x block_dim) matrix.
		Span<double> values;

		/// Get block row num.
		POLYFEM_BOTH int block_rows() const { return rows / block_dim; }
		/// Get block col num,.
		POLYFEM_BOTH int block_cols() const { return cols / block_dim; }

		/// @brief Get ptr to block start.
		/// @param block_i Block row.
		/// @param block_j Block col.
		POLYFEM_BOTH double *get_block(int block_i, int block_j) const
		{
			int row_start = row_ptr[block_i];
			int row_end = row_ptr[block_i + 1];
			for (int i = row_start; i < row_end; ++i)
			{
				if (col_idx[i] == block_j)
				{
					return values.data() + static_cast<size_t>(block_dim * block_dim) * static_cast<size_t>(i);
				}
			}
			return nullptr;
		}

		/// @brief Get ptr to scalar entry.
		/// @param i Scalar row.
		/// @param j Scalar col.
		POLYFEM_BOTH double *get_entry(int i, int j) const
		{
			int block_i = i / block_dim;
			int block_j = j / block_dim;
			double *block_ptr = get_block(block_i, block_j);
			if (!block_ptr)
				return nullptr;

			return block_ptr + (i % block_dim) * block_dim + (j % block_dim);
		}
	};

	class BSRMatrix
	{
	private:
		int rows_;          //< Scalar rows.
		int cols_;          //< Scalar cols.
		int block_dim_;     //< Can be 1,2, or 3.
		size_t value_size_; //< Non zero scalar value size.

		std::vector<int> row_ptr_; //< BSR row ptr.
		std::vector<int> col_idx_; //< BSR col idx.
		/// BSR values. each non zero is a row-major (block_dim x block_dim) matrix.
		std::vector<double> static_values_;
		/// Dynamic entries. For dynamic terms like collision and legacy assembler.
		std::vector<Eigen::Triplet<double>> dynamic_values_;

#ifdef POLYFEM_WITH_CUDA
		bool need_host_device_sync_ = true;
		DeviceBuf<int> d_row_ptr_;
		DeviceBuf<int> d_col_idx_;
		DeviceBuf<double> d_static_values_;
#endif

	public:
		BSRMatrix(const BSRSparsityPattern &sparsity);
		/// Construct a block_dim = 1 matrix with no static BSR entries (dynamic-only).
		BSRMatrix(int rows, int cols);

		/// Get Scalar rows.
		int rows() const { return rows_; }
		/// Get Scalar cols.
		int cols() const { return cols_; }
		int block_dim() const { return block_dim_; }

		/// Lazily allocate zero initialized static value array and return matrix view.
		BSRMatrixMutableView static_view();

		/// Access the dynamic (triplet) entries.
		std::vector<Eigen::Triplet<double>> &dynamic_view() { return dynamic_values_; }

		/// Convert static BSR and dynamic triplets into an Eigen StiffnessMatrix.
		/// @warning Might modify static storage!! Do not reuse BSR data after call.
		StiffnessMatrix to_stiffness_matrix(ExecutionPolicy policy = {});

		/// Reset host/device static value arrays to zero if they are allocated, and clear dynamic entries.
		void reset(ExecutionPolicy policy = {});

		bool has_allocate_host_value() const;
		bool has_allocate_device_value() const;

		/// Clear static host value storage and all device storage. Keep topology data.
		void clear_storage();

#ifdef POLYFEM_WITH_CUDA
		/// Lazily copy topology to device and allocate zero initialized static value array.
		BSRMatrixMutableView device_static_view(ExecutionPolicy policy);
#endif

	private:
#ifdef POLYFEM_WITH_CUDA
		/// Device impl of stiffness matrix conversion.
		StiffnessMatrix to_stiffness_matrix_device(ExecutionPolicy policy);
#endif
	};

	void append_sparse_matrix_to_triplets(
		const StiffnessMatrix &matrix,
		std::vector<Eigen::Triplet<double>> &triplets,
		double scale = 1.0);

	void add_sparse_matrix_to_bsr_static(
		const StiffnessMatrix &matrix,
		BSRMatrixMutableView bsr,
		double scale = 1.0);

} // namespace polyfem
