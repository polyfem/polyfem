#pragma once

#include <polyfem/utils/Types.hpp>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <memory>

namespace polyfem::utils
{
	/// abstract class used for caching
	class MatrixCache
	{
	public:
		MatrixCache() {}
		virtual ~MatrixCache() = default;

		virtual std::unique_ptr<MatrixCache> copy() const = 0;

		virtual void init(const size_t size) = 0;
		virtual void init(const size_t rows, const size_t cols) = 0;
		virtual void init(const MatrixCache &other) = 0;

		virtual void set_zero() = 0;

		virtual void reserve(const size_t size) = 0;
		virtual size_t entries_size() const = 0;
		virtual size_t capacity() const = 0;
		virtual size_t non_zeros() const = 0;
		virtual size_t triplet_count() const = 0;
		virtual bool is_sparse() const = 0;
		bool is_dense() const { return !is_sparse(); }

		virtual void add_value(const int e, const int i, const int j, const double value) = 0;
		virtual StiffnessMatrix get_matrix(const bool compute_mapping = true) = 0;
		virtual void prune() = 0;

		virtual std::shared_ptr<MatrixCache> operator+(const MatrixCache &a) const = 0;
		virtual void operator+=(const MatrixCache &o) = 0;
	};

	class SparseMatrixCache : public MatrixCache
	{
	public:
		// constructors (call init functions below)
		SparseMatrixCache() {}
		SparseMatrixCache(const size_t size);
		SparseMatrixCache(const size_t rows, const size_t cols);
		SparseMatrixCache(const MatrixCache &other);
		SparseMatrixCache(const SparseMatrixCache &other, const bool copy_main_cache_ptr = false);

		inline std::unique_ptr<MatrixCache> copy() const override
		{
			// just copy main cache pointer
			return std::make_unique<SparseMatrixCache>(*this, true);
		}

		/// set matrix to be size x size
		void init(const size_t size) override;
		/// set matrix to be rows x cols
		void init(const size_t rows, const size_t cols) override;
		/// set matrix to be a matrix of all zeros with same size as other
		void init(const MatrixCache &other) override;
		/// set matrix to be a matrix of all zeros with same size as other (potentially with the same main cache)
		void init(const SparseMatrixCache &other, const bool copy_main_cache_ptr = false);

		/// set matrix values to zero
		/// modifies tmp_, mat_, and values (setting all to zero)
		void set_zero() override;

		inline void reserve(const size_t size) override { entries_.reserve(size); }
		inline size_t entries_size() const override { return entries_.size(); }
		inline size_t capacity() const override { return entries_.capacity(); }
		inline size_t non_zeros() const override { return mapping_.empty() ? mat_.nonZeros() : values_.size(); }
		inline size_t triplet_count() const override { return entries_.size() + mat_.nonZeros(); }
		inline bool is_sparse() const override { return true; }
		inline size_t mapping_size() const { return mapping_.size(); }

		/// e = element_index, i = global row_index, j = global column_index, value = value to add to matrix
		/// if the cache is yet to be constructed, save the row, column, and value to be added to the second cache
		///     in this case, modifies_ entries_ and second_cache_entries_
		/// otherwise, save the value directly in the second cache
		///     in this case, modfies values_
		void add_value(const int e, const int i, const int j, const double value) override;
		/// if the cache is yet to be constructed, save the
		/// cached (ordered) indices in inner_index_ and outer_index_
		/// then fill in map and second_cache_
		///     in this case, modifies inner_index_, outer_index_, map, and second_cache_ to reflect the matrix structure
		///     also empties second_cache_entries and sets values_ to zero
		/// otherwise, update mat_ directly using the cached indices and values_
		///     in this case, modifies mat_ and sets values_ to zero
		StiffnessMatrix get_matrix(const bool compute_mapping = true) override;
		/// if caches have yet to be constructed, add the saved triplets to mat_
		/// modifies tmp_ and mat_, also sets entries_ to be empty after writing its values to mat_
		void prune() override; ///< add saved entries to stored matrix

		std::shared_ptr<MatrixCache> operator+(const MatrixCache &a) const override;
		std::shared_ptr<MatrixCache> operator+(const SparseMatrixCache &a) const;
		void operator+=(const MatrixCache &o) override;
		void operator+=(const SparseMatrixCache &o);

		const StiffnessMatrix &mat() const { return mat_; }
		const std::vector<Eigen::Triplet<double>> &entries() const { return entries_; }

	private:
		size_t size_;
		StiffnessMatrix tmp_, mat_;
		std::vector<Eigen::Triplet<double>> entries_;              ///< contains global matrix indices and corresponding value
		std::vector<std::vector<std::pair<int, size_t>>> mapping_; ///< maps row indices to column index/local index pairs
		std::vector<int> inner_index_, outer_index_;               ///< saves inner/outer indices for sparse matrix
		std::vector<double> values_;                               ///< buffer for values (corresponds to inner/outer_index_ structure for sparse matrix)
		const SparseMatrixCache *main_cache_ = nullptr;

		std::vector<std::vector<int>> second_cache_;                         ///< maps element index to local index
		std::vector<std::vector<std::pair<int, int>>> second_cache_entries_; ///< maps element indices to global matrix indices
		int current_e_ = -1;
		int current_e_index_ = -1;

		inline const SparseMatrixCache *main_cache() const
		{
			return main_cache_ == nullptr ? this : main_cache_;
		}

		inline const std::vector<std::vector<std::pair<int, size_t>>> &mapping() const
		{
			return main_cache()->mapping_;
		}

		inline const std::vector<std::vector<int>> &second_cache() const
		{
			return main_cache()->second_cache_;
		}
	};

	class DenseMatrixCache : public MatrixCache
	{
	public:
		DenseMatrixCache() {}
		DenseMatrixCache(const size_t size);
		DenseMatrixCache(const size_t rows, const size_t cols);
		DenseMatrixCache(const MatrixCache &other);
		DenseMatrixCache(const DenseMatrixCache &other);

		inline std::unique_ptr<MatrixCache> copy() const override
		{
			return std::make_unique<DenseMatrixCache>(*this);
		}

		void init(const size_t size) override;
		void init(const size_t rows, const size_t cols) override;
		void init(const MatrixCache &other) override;
		void init(const DenseMatrixCache &other);

		void set_zero() override;

		inline void reserve(const size_t size) override {}
		inline size_t entries_size() const override { return 0; }
		inline size_t capacity() const override { return mat_.size(); }
		inline size_t non_zeros() const override { return mat_.size(); }
		inline size_t triplet_count() const override { return non_zeros(); }
		inline bool is_sparse() const override { return false; }

		void add_value(const int e, const int i, const int j, const double value) override;
		StiffnessMatrix get_matrix(const bool compute_mapping = true) override;
		void prune() override;

		std::shared_ptr<MatrixCache> operator+(const MatrixCache &a) const override;
		std::shared_ptr<MatrixCache> operator+(const DenseMatrixCache &a) const;
		void operator+=(const MatrixCache &o) override;
		void operator+=(const DenseMatrixCache &o);

		const Eigen::MatrixXd &mat() const { return mat_; }

	private:
		Eigen::MatrixXd mat_;
	};
} // namespace polyfem::utils
