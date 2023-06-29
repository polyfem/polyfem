#pragma once

#include <polyfem/utils/Types.hpp>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <memory>

namespace polyfem::utils
{
	class MatrixCache
	{
	public:
		MatrixCache() {}
		virtual ~MatrixCache() = default;

		virtual std::unique_ptr<MatrixCache> clone() const = 0;

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
		SparseMatrixCache() {}
		SparseMatrixCache(const size_t size);
		SparseMatrixCache(const size_t rows, const size_t cols);
		SparseMatrixCache(const MatrixCache &other);
		SparseMatrixCache(const SparseMatrixCache &other, const bool copy_main_cache_ptr = false);

		inline std::unique_ptr<MatrixCache> clone() const override
		{
			// just copy main cache pointer
			return std::make_unique<SparseMatrixCache>(*this, true);
		}

		void init(const size_t size) override;
		void init(const size_t rows, const size_t cols) override;
		void init(const MatrixCache &other) override;
		void init(const SparseMatrixCache &other, const bool copy_main_cache_ptr = false);

		void set_zero() override;

		inline void reserve(const size_t size) override { entries_.reserve(size); }
		inline size_t entries_size() const override { return entries_.size(); }
		inline size_t capacity() const override { return entries_.capacity(); }
		inline size_t non_zeros() const override { return mapping_.empty() ? mat_.nonZeros() : values_.size(); }
		inline size_t triplet_count() const override { return entries_.size() + mat_.nonZeros(); }
		inline bool is_sparse() const override { return true; }
		inline size_t mapping_size() const { return mapping_.size(); }

		void add_value(const int e, const int i, const int j, const double value) override;
		StiffnessMatrix get_matrix(const bool compute_mapping = true) override;
		void prune() override;

		std::shared_ptr<MatrixCache> operator+(const MatrixCache &a) const override;
		std::shared_ptr<MatrixCache> operator+(const SparseMatrixCache &a) const;
		void operator+=(const MatrixCache &o) override;
		void operator+=(const SparseMatrixCache &o);

		const StiffnessMatrix &mat() const { return mat_; }
		const std::vector<Eigen::Triplet<double>> &entries() const { return entries_; }

	private:
		size_t size_;
		StiffnessMatrix tmp_, mat_;
		std::vector<Eigen::Triplet<double>> entries_;
		std::vector<std::vector<std::pair<int, size_t>>> mapping_;
		std::vector<int> inner_index_, outer_index_;
		std::vector<double> values_;
		const SparseMatrixCache *main_cache_ = nullptr;

		std::vector<std::vector<int>> second_cache_;
		std::vector<std::vector<std::pair<int, int>>> second_cache_entries_;
		bool use_second_cache_ = true;
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

		inline std::unique_ptr<MatrixCache> clone() const override
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
