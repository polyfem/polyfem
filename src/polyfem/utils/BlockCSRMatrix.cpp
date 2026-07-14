#include <polyfem/utils/BlockCSRMatrix.hpp>

#include <polyfem/utils/Span.hpp>
#include <polyfem/utils/CudaBoth.hpp>

#include <vector>
#include <unordered_set>
#include <cstdint>
#include <cassert>
#include <utility>
#include <algorithm>
#include <stdexcept>

#ifdef POLYFEM_WITH_CUDA
#include <polyfem/utils/CudaUtils.hpp>
#include <cuda/algorithm>
#endif

namespace polyfem
{

	namespace
	{

		uint64_t pack(uint32_t row, uint32_t col)
		{
			return (static_cast<uint64_t>(row) << 32) | static_cast<uint64_t>(col);
		}

		std::pair<uint32_t, uint32_t> unpack(uint64_t key)
		{
			uint32_t row = static_cast<uint32_t>(key >> 32);
			uint32_t col = static_cast<uint32_t>(key & 0xFFFFFFFF);
			return {row, col};
		}

	} // namespace

	void BSRSparsityPattern::insert(uint32_t row, uint32_t col)
	{
		assert(row >= 0 && row < rows);
		assert(col >= 0 && col < cols);
		non_zeros.insert(pack(row, col));
	}

	void BSRSparsityPattern::join(const BSRSparsityPattern &other)
	{
		assert(rows == other.rows);
		assert(cols == other.cols);

		/// TODO: for mixed assembly, min might be the right choice. Investigate after porting mixed assembly.
		block_dim = std::max(block_dim, other.block_dim);
		non_zeros.insert(other.non_zeros.begin(), other.non_zeros.end());
	}

	BSRMatrix::BSRMatrix(const BSRSparsityPattern &sparsity)
	{
		auto &s = sparsity;
		assert(s.rows >= 0 && s.rows % s.block_dim == 0);
		assert(s.cols >= 0 && s.cols % s.block_dim == 0);
		assert(s.rows == s.cols);
		assert(s.block_dim >= 1 && s.block_dim <= 3);

		rows_ = s.rows;
		cols_ = s.cols;
		block_dim_ = s.block_dim;

		// Build sorted block row, col key.
		std::vector<uint64_t> block_keys;
		if (s.block_dim == 1)
		{
			block_keys.insert(block_keys.end(), s.non_zeros.begin(), s.non_zeros.end());
		}
		else
		{
			std::unordered_set<uint64_t> block_nnz;
			for (uint64_t key : s.non_zeros)
			{
				auto [row, col] = unpack(key);
				uint64_t new_key = pack(row / s.block_dim, col / s.block_dim);
				block_nnz.insert(new_key);
			}
			block_keys.insert(block_keys.end(), block_nnz.begin(), block_nnz.end());
		}
		std::sort(block_keys.begin(), block_keys.end());

		int block_rows = rows_ / block_dim_;
		row_ptr_.assign(block_rows + 1, 0);
		for (uint64_t key : block_keys)
		{
			auto [row, col] = unpack(key);
			row_ptr_[row + 1]++;
			col_idx_.push_back(col);
		}
		// prefix sum.
		for (int row = 0; row < block_rows; ++row)
		{
			row_ptr_[row + 1] += row_ptr_[row];
		}

		value_size_ = static_cast<size_t>(block_dim_) * static_cast<size_t>(block_dim_) * col_idx_.size();
	}

	BSRMatrix::BSRMatrix(int rows, int cols)
		: rows_(rows), cols_(cols), block_dim_(1), value_size_(0)
	{
		assert(rows >= 0);
		assert(cols >= 0);
		// block_dim = 1, no static BSR entries. row_ptr_ has one zero per row + sentry.
		row_ptr_.assign(rows + 1, 0);
	}

	void BSRMatrix::clear_storage()
	{
		static_values_ = {};
		dynamic_values_ = {};

#ifdef POLYFEM_WITH_CUDA
		need_host_device_sync_ = true;
		d_row_ptr_ = {};
		d_col_idx_ = {};
		d_static_values_ = {};
#endif
	}

	bool BSRMatrix::has_allocate_host_value() const
	{
		return !static_values_.empty();
	}

	bool BSRMatrix::has_allocate_device_value() const
	{
#ifdef POLYFEM_WITH_CUDA
		return d_static_values_.has_value();
#else
		return false;
#endif
	}

	void BSRMatrix::reset(ExecutionPolicy policy)
	{
		std::fill(static_values_.begin(), static_values_.end(), 0.0);
		dynamic_values_.clear();

#ifdef POLYFEM_WITH_CUDA
		if (d_static_values_)
		{
			assert(policy.stream);
			cuda::fill_bytes(*policy.stream, *d_static_values_, 0);
			policy.stream->sync();
		}
#endif
	}

	BSRMatrixMutableView BSRMatrix::static_view()
	{
		if (static_values_.empty())
			static_values_.resize(value_size_, 0.0);
		return BSRMatrixMutableView{rows_, cols_, block_dim_, row_ptr_, col_idx_, static_values_};
	}

	StiffnessMatrix BSRMatrix::to_stiffness_matrix(ExecutionPolicy policy)
	{
		assert(block_dim_ > 0);
		assert(rows_ % block_dim_ == 0);
		assert(cols_ % block_dim_ == 0);

#ifdef POLYFEM_WITH_CUDA
		if (policy.mode == ExecutionMode::Hybrid && has_allocate_device_value())
		{
			return to_stiffness_matrix_device(policy);
		}
#endif

		int bd = block_dim_;
		int block_size = bd * bd;

		BSRMatrixMutableView bsr = static_view();

		// Combine static and dynamic values into one triplet vector, then call
		// Eigen set from tripets.

		std::vector<Eigen::Triplet<double>> entries;
		entries.reserve(bsr.values.size() + dynamic_values_.size());

		for (int br = 0; br < bsr.block_rows(); ++br)
		{
			for (int p = bsr.row_ptr[br]; p < bsr.row_ptr[br + 1]; ++p)
			{
				int bc = bsr.col_idx[p];
				const double *block = bsr.values.data() + static_cast<size_t>(p) * static_cast<size_t>(block_size);

				for (int i = 0; i < bd; ++i)
				{
					for (int j = 0; j < bd; ++j)
					{
						double value = block[i * bd + j];
						if (value != 0.0)
						{
							entries.emplace_back(br * bd + i, bc * bd + j, value);
						}
					}
				}
			}
		}

		entries.insert(entries.end(), dynamic_values_.begin(), dynamic_values_.end());

		StiffnessMatrix out(rows_, cols_);
		out.setFromTriplets(entries.begin(), entries.end());
		return out;
	}

#ifdef POLYFEM_WITH_CUDA
	BSRMatrixMutableView BSRMatrix::device_static_view(ExecutionPolicy policy)
	{
		auto &p = policy;
		if (need_host_device_sync_)
		{
			assert(policy.stream && policy.mr);
			d_row_ptr_ = copy_to_device_async<int>(row_ptr_, policy);
			d_col_idx_ = copy_to_device_async<int>(col_idx_, policy);
			d_static_values_ = cuda::make_buffer<double>(*p.stream, *p.mr, value_size_, cuda::no_init);
			cuda::fill_bytes(*p.stream, *d_static_values_, 0);

			p.stream->sync();
			need_host_device_sync_ = false;
		}
		return BSRMatrixMutableView{rows_, cols_, block_dim_, *d_row_ptr_, *d_col_idx_, *d_static_values_};
	}
#endif

	void append_sparse_matrix_to_triplets(
		const StiffnessMatrix &matrix,
		std::vector<Eigen::Triplet<double>> &triplets,
		double scale)
	{
		for (int k = 0; k < matrix.outerSize(); ++k)
		{
			for (StiffnessMatrix::InnerIterator it(matrix, k); it; ++it)
			{
				double value = scale * it.value();
				triplets.emplace_back(it.row(), it.col(), value);
			}
		}
	}

	void add_sparse_matrix_to_bsr_static(
		const StiffnessMatrix &matrix,
		BSRMatrixMutableView bsr,
		double scale)
	{
		assert(matrix.rows() == bsr.rows);
		assert(matrix.cols() == bsr.cols);

		for (int k = 0; k < matrix.outerSize(); ++k)
		{
			for (StiffnessMatrix::InnerIterator it(matrix, k); it; ++it)
			{
				double *entry = bsr.get_entry(it.row(), it.col());
				assert(entry != nullptr);
				*entry += scale * it.value();
			}
		}
	}

} // namespace polyfem
