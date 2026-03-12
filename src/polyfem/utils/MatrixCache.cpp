#include "MatrixCache.hpp"

#include <polyfem/utils/MaybeParallelFor.hpp>
#include <polyfem/utils/Logger.hpp>

namespace polyfem::utils
{
	SparseMatrixCache::SparseMatrixCache(const size_t size)
	{
		init(size);
	}

	SparseMatrixCache::SparseMatrixCache(const size_t rows, const size_t cols)
	{
		init(rows, cols);
	}

	SparseMatrixCache::SparseMatrixCache(const MatrixCache &other)
	{
		init(other);
	}

	SparseMatrixCache::SparseMatrixCache(
		const SparseMatrixCache &other, const bool copy_main_cache_ptr)
	{
		init(other, copy_main_cache_ptr);
	}

	void SparseMatrixCache::init(const size_t size)
	{
		assert(mapping().empty() || size_ == size);

		size_ = size;
		tmp_.resize(size_, size_);
		mat_.resize(size_, size_);
		mat_.setZero();
	}

	void SparseMatrixCache::init(const size_t rows, const size_t cols)
	{
		assert(mapping().empty());

		size_ = rows == cols ? rows : 0;
		tmp_.resize(rows, cols);
		mat_.resize(rows, cols);
		mat_.setZero();
	}

	void SparseMatrixCache::init(const MatrixCache &other)
	{
		assert(this != &other);
		assert(&other == &dynamic_cast<const SparseMatrixCache &>(other));
		init(dynamic_cast<const SparseMatrixCache &>(other));
	}

	void SparseMatrixCache::init(
		const SparseMatrixCache &other, const bool copy_main_cache_ptr)
	{
		assert(this != &other);
		if (copy_main_cache_ptr)
		{
			main_cache_ = other.main_cache_;
		}
		else if (main_cache_ == nullptr)
		{
			main_cache_ = other.main_cache();
			// Only one level of cache
			assert(main_cache_ != this && main_cache_ != nullptr && main_cache_->main_cache_ == nullptr);
		}
		size_ = other.size_;

		values_.resize(other.values_.size());

		tmp_.resize(other.mat_.rows(), other.mat_.cols());
		mat_.resize(other.mat_.rows(), other.mat_.cols());
		mat_.setZero();
		std::fill(values_.begin(), values_.end(), 0);
	}

	void SparseMatrixCache::set_zero()
	{
		tmp_.setZero();
		mat_.setZero();

		std::fill(values_.begin(), values_.end(), 0);
	}

	void SparseMatrixCache::add_value(const int e, const int i, const int j, const double value)
	{
		// caches have yet to be constructed (likely because the matrix has yet to be fully assembled)
		if (mapping().empty())
		{
			// save entry so it can be added to the matrix later
			entries_.emplace_back(i, j, value);

			// save the index information so the cache can be built later
			if (second_cache_entries_.size() <= e)
				second_cache_entries_.resize(e + 1);
			second_cache_entries_[e].emplace_back(i, j);
		}
		else
		{
			if (e != current_e_)
			{
				current_e_ = e;
				current_e_index_ = 0;
			}

			// save entry directly to value buffer at the proper index
			values_[second_cache()[e][current_e_index_]] += value;
			current_e_index_++;
		}
	}

	void SparseMatrixCache::prune()
	{
		// caches have yet to be constructed (likely because the matrix has yet to be fully assembled)
		if (mapping().empty())
		{
			tmp_.setFromTriplets(entries_.begin(), entries_.end());
			tmp_.makeCompressed();
			mat_ += tmp_;

			tmp_.setZero();
			tmp_.data().squeeze();
			mat_.makeCompressed();

			entries_.clear();

			mat_.makeCompressed();
		}
	}

	polyfem::StiffnessMatrix SparseMatrixCache::get_matrix(const bool compute_mapping)
	{
		prune();

		// caches have yet to be constructed (likely because the matrix has yet to be fully assembled)
		if (mapping().empty())
		{
			if (compute_mapping && size_ > 0)
			{
				assert(main_cache_ == nullptr);

				values_.resize(mat_.nonZeros());
				inner_index_.resize(mat_.nonZeros());
				outer_index_.resize(mat_.rows() + 1);
				mapping_.resize(mat_.rows());

				// note: mat_ is column major
				const auto inn_ptr = mat_.innerIndexPtr();
				const auto out_ptr = mat_.outerIndexPtr();
				inner_index_.assign(inn_ptr, inn_ptr + inner_index_.size());
				outer_index_.assign(out_ptr, out_ptr + outer_index_.size());

				size_t index = 0;
				// loop over columns of the matrix
				for (size_t i = 0; i < mat_.cols(); ++i)
				{
					const auto start = outer_index_[i];
					const auto end = outer_index_[i + 1];

					// loop over the nonzero elements of the given column
					for (size_t ii = start; ii < end; ++ii)
					{
						// pick out current row
						const auto j = inner_index_[ii];
						auto &map = mapping_[j];
						map.emplace_back(i, index);
						++index;
					}
				}

				logger().trace("Cache computed");

				second_cache_.clear();
				second_cache_.resize(second_cache_entries_.size());
				// loop over each element
				for (int e = 0; e < second_cache_entries_.size(); ++e)
				{
					// loop over each global index affected by the given element
					for (const auto &p : second_cache_entries_[e])
					{
						const int i = p.first;
						const int j = p.second;

						// pick out column/sparse matrix index pairs for the given column
						const auto &map = mapping()[i];
						int index = -1;

						// loop over column/sparse matrix index pairs
						for (const auto &p : map)
						{
							// match columns
							if (p.first == j)
							{
								assert(p.second < values_.size());
								index = p.second;
								break;
							}
						}
						assert(index >= 0);

						// save the sparse matrix index used by this element
						second_cache_[e].emplace_back(index);
					}
				}

				second_cache_entries_.resize(0);

				logger().trace("Second cache computed");
			}
		}
		else
		{
			assert(size_ > 0);
			const auto &outer_index = main_cache()->outer_index_;
			const auto &inner_index = main_cache()->inner_index_;
			// directly write the values to the matrix
			mat_ = Eigen::Map<const StiffnessMatrix>(
				size_, size_, values_.size(), &outer_index[0], &inner_index[0], &values_[0]);

			current_e_ = -1;
			current_e_index_ = -1;
		}
		std::fill(values_.begin(), values_.end(), 0);
		return mat_;
	}

	std::shared_ptr<MatrixCache> SparseMatrixCache::operator+(const MatrixCache &a) const
	{
		assert(&a == &dynamic_cast<const SparseMatrixCache &>(a));
		return *this + dynamic_cast<const SparseMatrixCache &>(a);
	}

	std::shared_ptr<MatrixCache> SparseMatrixCache::operator+(const SparseMatrixCache &a) const
	{
		std::shared_ptr<SparseMatrixCache> out = std::make_shared<SparseMatrixCache>(a);

		if (a.mapping().empty() || mapping().empty())
		{
			out->mat_ = a.mat_ + mat_;
			const size_t this_e_size = second_cache_entries_.size();
			const size_t a_e_size = a.second_cache_entries_.size();

			out->second_cache_entries_.resize(std::max(this_e_size, a_e_size));
			for (int e = 0; e < std::min(this_e_size, a_e_size); ++e)
			{
				assert(second_cache_entries_[e].size() == 0 || a.second_cache_entries_[e].size() == 0);
				out->second_cache_entries_[e].insert(out->second_cache_entries_[e].end(), second_cache_entries_[e].begin(), second_cache_entries_[e].end());
				out->second_cache_entries_[e].insert(out->second_cache_entries_[e].end(), a.second_cache_entries_[e].begin(), a.second_cache_entries_[e].end());
			}

			for (int e = std::min(this_e_size, a_e_size); e < std::max(this_e_size, a_e_size); ++e)
			{
				if (second_cache_entries_.size() < e)
					out->second_cache_entries_[e].insert(out->second_cache_entries_[e].end(), second_cache_entries_[e].begin(), second_cache_entries_[e].end());
				else
					out->second_cache_entries_[e].insert(out->second_cache_entries_[e].end(), a.second_cache_entries_[e].begin(), a.second_cache_entries_[e].end());
			}
		}
		else
		{
			const auto &outer_index = main_cache()->outer_index_;
			const auto &inner_index = main_cache()->inner_index_;
			const auto &aouter_index = a.main_cache()->outer_index_;
			const auto &ainner_index = a.main_cache()->inner_index_;
			assert(ainner_index.size() == inner_index.size());
			assert(aouter_index.size() == outer_index.size());
			assert(a.values_.size() == values_.size());

			maybe_parallel_for(a.values_.size(), [&](int start, int end, int thread_id) {
				for (int i = start; i < end; ++i)
				{
					out->values_[i] = a.values_[i] + values_[i];
				}
			});
		}

		return out;
	}

	void SparseMatrixCache::operator+=(const MatrixCache &o)
	{
		assert(&o == &dynamic_cast<const SparseMatrixCache &>(o));
		*this += dynamic_cast<const SparseMatrixCache &>(o);
	}

	void SparseMatrixCache::operator+=(const SparseMatrixCache &o)
	{
		if (mapping().empty() || o.mapping().empty())
		{
			mat_ += o.mat_;

			const size_t this_e_size = second_cache_entries_.size();
			const size_t o_e_size = o.second_cache_entries_.size();

			second_cache_entries_.resize(std::max(this_e_size, o_e_size));
			for (int e = 0; e < o_e_size; ++e)
			{
				assert(second_cache_entries_[e].size() == 0 || o.second_cache_entries_[e].size() == 0);
				second_cache_entries_[e].insert(second_cache_entries_[e].end(), o.second_cache_entries_[e].begin(), o.second_cache_entries_[e].end());
			}
		}
		else
		{
			const auto &outer_index = main_cache()->outer_index_;
			const auto &inner_index = main_cache()->inner_index_;
			const auto &oouter_index = o.main_cache()->outer_index_;
			const auto &oinner_index = o.main_cache()->inner_index_;
			assert(inner_index.size() == oinner_index.size());
			assert(outer_index.size() == oouter_index.size());
			assert(values_.size() == o.values_.size());

			maybe_parallel_for(o.values_.size(), [&](int start, int end, int thread_id) {
				for (int i = start; i < end; ++i)
				{
					values_[i] += o.values_[i];
				}
			});
		}
	}

	// ========================================================================

	DenseMatrixCache::DenseMatrixCache(const size_t size)
	{
		mat_.setZero(size, size);
	}

	DenseMatrixCache::DenseMatrixCache(const size_t rows, const size_t cols)
	{
		mat_.setZero(rows, cols);
	}

	DenseMatrixCache::DenseMatrixCache(const MatrixCache &other)
	{
		init(other);
	}

	DenseMatrixCache::DenseMatrixCache(const DenseMatrixCache &other)
	{
		init(other);
	}

	void DenseMatrixCache::init(const size_t size)
	{
		mat_.setZero(size, size);
	}

	void DenseMatrixCache::init(const size_t rows, const size_t cols)
	{
		mat_.setZero(rows, cols);
	}

	void DenseMatrixCache::init(const MatrixCache &other)
	{
		init(dynamic_cast<const DenseMatrixCache &>(other));
	}

	void DenseMatrixCache::init(const DenseMatrixCache &other)
	{
		mat_.setZero(other.mat_.rows(), other.mat_.cols());
	}

	void DenseMatrixCache::set_zero()
	{
		mat_.setZero();
	}

	void DenseMatrixCache::add_value(const int e, const int i, const int j, const double value)
	{
		mat_(i, j) += value;
	}

	void DenseMatrixCache::prune() {}

	polyfem::StiffnessMatrix DenseMatrixCache::get_matrix(const bool compute_mapping)
	{
		return mat_.sparseView();
	}

	std::shared_ptr<MatrixCache> DenseMatrixCache::operator+(const MatrixCache &a) const
	{
		return *this + dynamic_cast<const DenseMatrixCache &>(a);
	}

	std::shared_ptr<MatrixCache> DenseMatrixCache::operator+(const DenseMatrixCache &a) const
	{
		std::shared_ptr<DenseMatrixCache> out = std::make_shared<DenseMatrixCache>(a);
		out->mat_ += mat_;
		return out;
	}

	void DenseMatrixCache::operator+=(const MatrixCache &o)
	{
		*this += dynamic_cast<const DenseMatrixCache &>(o);
	}

	void DenseMatrixCache::operator+=(const DenseMatrixCache &o)
	{
		mat_ += o.mat_;
	}

} // namespace polyfem::utils