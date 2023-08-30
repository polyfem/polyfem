#include "MatrixUtils.hpp"

#include <polyfem/utils/MaybeParallelFor.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/Timer.hpp>

#include <igl/list_to_matrix.h>

#include <iostream>
#include <fstream>
#include <iomanip> // setprecision
#include <vector>
#include <filesystem>

void polyfem::utils::show_matrix_stats(const Eigen::MatrixXd &M)
{
	Eigen::FullPivLU<Eigen::MatrixXd> lu(M);
	Eigen::JacobiSVD<Eigen::MatrixXd> svd(M);
	double s1 = svd.singularValues()(0);
	double s2 = svd.singularValues()(svd.singularValues().size() - 1);
	double cond = s1 / s2;

	logger().trace("----------------------------------------");
	logger().trace("-- Determinant: {}", M.determinant());
	logger().trace("-- Singular values: {} {}", s1, s2);
	logger().trace("-- Cond: {}", cond);
	logger().trace("-- Invertible: {}", lu.isInvertible());
	logger().trace("----------------------------------------");
	// logger().trace("{}", lu.solve(M) );
}

polyfem::utils::SparseMatrixCache::SparseMatrixCache(const size_t size)
	: size_(size)
{
	tmp_.resize(size_, size_);
	mat_.resize(size_, size_);
	mat_.setZero();
}

polyfem::utils::SparseMatrixCache::SparseMatrixCache(const size_t rows, const size_t cols)
	: size_(rows == cols ? rows : 0)
{
	tmp_.resize(rows, cols);
	mat_.resize(rows, cols);
	mat_.setZero();
}

polyfem::utils::SparseMatrixCache::SparseMatrixCache(const polyfem::utils::SparseMatrixCache &other)
{
	init(other);
}

void polyfem::utils::SparseMatrixCache::init(const size_t size)
{
	assert(mapping().empty() || size_ == size);

	size_ = size;
	tmp_.resize(size_, size_);
	mat_.resize(size_, size_);
	mat_.setZero();
}

void polyfem::utils::SparseMatrixCache::init(const size_t rows, const size_t cols)
{
	assert(mapping().empty());

	size_ = rows == cols ? rows : 0;
	tmp_.resize(rows, cols);
	mat_.resize(rows, cols);
	mat_.setZero();
}

void polyfem::utils::SparseMatrixCache::init(const SparseMatrixCache &other)
{
	if (main_cache_ == nullptr)
	{
		if (other.main_cache_ == nullptr)
			main_cache_ = &other;
		else
			main_cache_ = other.main_cache_;
	}
	size_ = other.size_;

	values_.resize(other.values_.size());

	tmp_.resize(other.mat_.rows(), other.mat_.cols());
	mat_.resize(other.mat_.rows(), other.mat_.cols());
	mat_.setZero();
	std::fill(values_.begin(), values_.end(), 0);
}

void polyfem::utils::SparseMatrixCache::set_zero()
{
	tmp_.setZero();
	mat_.setZero();

	std::fill(values_.begin(), values_.end(), 0);
}

void polyfem::utils::SparseMatrixCache::add_value(const int e, const int i, const int j, const double value)
{
	if (mapping().empty())
	{
		entries_.emplace_back(i, j, value);
		if (second_cache_entries_.size() <= e)
			second_cache_entries_.resize(e + 1);
		second_cache_entries_[e].emplace_back(i, j);
	}
	else
	{
		if (use_second_cache_)
		{
			if (e != current_e_)
			{
				current_e_ = e;
				current_e_index_ = 0;
			}

			values_[second_cache()[e][current_e_index_]] += value;
			current_e_index_++;
		}
		else
		{
			// mapping()[i].find(j)
			const auto &map = mapping()[i];
			bool found = false;
			for (const auto &p : map)
			{
				if (p.first == j)
				{
					assert(p.second < values_.size());
					values_[p.second] += value;
					found = true;
					break;
				}
			}
			assert(found);
		}
	}
}

void polyfem::utils::SparseMatrixCache::prune()
{
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

polyfem::StiffnessMatrix polyfem::utils::SparseMatrixCache::get_matrix(const bool compute_mapping)
{
	prune();

	if (mapping().empty())
	{
		if (compute_mapping && size_ > 0)
		{
			assert(main_cache_ == nullptr);

			values_.resize(mat_.nonZeros());
			inner_index_.resize(mat_.nonZeros());
			outer_index_.resize(mat_.rows() + 1);
			mapping_.resize(mat_.rows());

			const auto inn_ptr = mat_.innerIndexPtr();
			const auto out_ptr = mat_.outerIndexPtr();
			inner_index_.assign(inn_ptr, inn_ptr + inner_index_.size());
			outer_index_.assign(out_ptr, out_ptr + outer_index_.size());

			size_t index = 0;
			for (size_t i = 0; i < mat_.rows(); ++i)
			{

				const auto start = outer_index_[i];
				const auto end = outer_index_[i + 1];

				for (size_t ii = start; ii < end; ++ii)
				{
					const auto j = inner_index_[ii];
					auto &map = mapping_[j];
					map.emplace_back(i, index);
					++index;
				}
			}

			logger().trace("Cache computed");

			if (use_second_cache_)
			{
				second_cache_.clear();
				second_cache_.resize(second_cache_entries_.size());
				for (int e = 0; e < second_cache_entries_.size(); ++e)
				{
					for (const auto &p : second_cache_entries_[e])
					{
						const int i = p.first;
						const int j = p.second;

						const auto &map = mapping()[i];
						int index = -1;
						for (const auto &p : map)
						{
							if (p.first == j)
							{
								assert(p.second < values_.size());
								index = p.second;
								break;
							}
						}
						assert(index >= 0);

						second_cache_[e].emplace_back(index);
					}
				}

				second_cache_entries_.resize(0);

				logger().trace("Second cache computed");
			}
		}
	}
	else
	{
		assert(size_ > 0);
		const auto &outer_index = main_cache_ == nullptr ? outer_index_ : main_cache_->outer_index_;
		const auto &inner_index = main_cache_ == nullptr ? inner_index_ : main_cache_->inner_index_;
		mat_ = Eigen::Map<const StiffnessMatrix>(
			size_, size_, values_.size(), &outer_index[0], &inner_index[0], &values_[0]);

		if (use_second_cache_)
		{
			current_e_ = -1;
			current_e_index_ = -1;

			logger().trace("Using second cache");
		}
		else
			logger().trace("Using cache");
	}
	std::fill(values_.begin(), values_.end(), 0);
	return mat_;
}

polyfem::utils::SparseMatrixCache polyfem::utils::SparseMatrixCache::operator+(const SparseMatrixCache &a) const
{
	polyfem::utils::SparseMatrixCache out(a);

	if (a.mapping().empty() || mapping().empty())
	{
		out.mat_ = a.mat_ + mat_;
		if (use_second_cache_)
		{
			const size_t this_e_size = second_cache_entries_.size();
			const size_t a_e_size = a.second_cache_entries_.size();

			out.second_cache_entries_.resize(std::max(this_e_size, a_e_size));
			for (int e = 0; e < std::min(this_e_size, a_e_size); ++e)
			{
				assert(second_cache_entries_[e].size() == 0 || a.second_cache_entries_[e].size() == 0);
				out.second_cache_entries_[e].insert(out.second_cache_entries_[e].end(), second_cache_entries_[e].begin(), second_cache_entries_[e].end());
				out.second_cache_entries_[e].insert(out.second_cache_entries_[e].end(), a.second_cache_entries_[e].begin(), a.second_cache_entries_[e].end());
			}

			for (int e = std::min(this_e_size, a_e_size); e < std::max(this_e_size, a_e_size); ++e)
			{
				if (second_cache_entries_.size() < e)
					out.second_cache_entries_[e].insert(out.second_cache_entries_[e].end(), second_cache_entries_[e].begin(), second_cache_entries_[e].end());
				else
					out.second_cache_entries_[e].insert(out.second_cache_entries_[e].end(), a.second_cache_entries_[e].begin(), a.second_cache_entries_[e].end());
			}
		}
	}
	else
	{
		const auto &outer_index = main_cache_ == nullptr ? outer_index_ : main_cache_->outer_index_;
		const auto &inner_index = main_cache_ == nullptr ? inner_index_ : main_cache_->inner_index_;
		const auto &aouter_index = a.main_cache_ == nullptr ? a.outer_index_ : a.main_cache_->outer_index_;
		const auto &ainner_index = a.main_cache_ == nullptr ? a.inner_index_ : a.main_cache_->inner_index_;
		assert(ainner_index.size() == inner_index.size());
		assert(aouter_index.size() == outer_index.size());
		assert(a.values_.size() == values_.size());

		maybe_parallel_for(a.values_.size(), [&](int start, int end, int thread_id) {
			for (int i = start; i < end; ++i)
			{
				out.values_[i] = a.values_[i] + values_[i];
			}
		});
	}

	return out;
}

void polyfem::utils::SparseMatrixCache::operator+=(const SparseMatrixCache &o)
{
	if (mapping().empty() || o.mapping().empty())
	{
		mat_ += o.mat_;

		if (use_second_cache_)
		{
			const size_t this_e_size = second_cache_entries_.size();
			const size_t o_e_size = o.second_cache_entries_.size();

			second_cache_entries_.resize(std::max(this_e_size, o_e_size));
			for (int e = 0; e < o_e_size; ++e)
			{
				assert(second_cache_entries_[e].size() == 0 || o.second_cache_entries_[e].size() == 0);
				second_cache_entries_[e].insert(second_cache_entries_[e].end(), o.second_cache_entries_[e].begin(), o.second_cache_entries_[e].end());
			}
		}
	}
	else
	{
		const auto &outer_index = main_cache_ == nullptr ? outer_index_ : main_cache_->outer_index_;
		const auto &inner_index = main_cache_ == nullptr ? inner_index_ : main_cache_->inner_index_;
		const auto &oouter_index = o.main_cache_ == nullptr ? o.outer_index_ : o.main_cache_->outer_index_;
		const auto &oinner_index = o.main_cache_ == nullptr ? o.inner_index_ : o.main_cache_->inner_index_;
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

namespace
{
	inline bool nanproof_equals(const double x, const double y)
	{
		return x == y || (std::isnan(x) == std::isnan(y));
	}
} // namespace

// Flatten rowwises
Eigen::VectorXd polyfem::utils::flatten(const Eigen::MatrixXd &X)
{
	if (X.size() == 0)
		return Eigen::VectorXd();

	Eigen::VectorXd x(X.size());
	for (int i = 0; i < X.rows(); ++i)
	{
		for (int j = 0; j < X.cols(); ++j)
		{
			x(i * X.cols() + j) = X(i, j);
		}
	}
	assert(nanproof_equals(X(0, 0), x(0)));
	assert(X.cols() <= 1 || nanproof_equals(X(0, 1), x(1)));
	return x;
}

// Unflatten rowwises, so every dim elements in x become a row.
Eigen::MatrixXd polyfem::utils::unflatten(const Eigen::VectorXd &x, int dim)
{
	if (x.size() == 0)
		return Eigen::MatrixXd(0, dim);

	assert(x.size() % dim == 0);
	Eigen::MatrixXd X(x.size() / dim, dim);
	for (int i = 0; i < x.size(); ++i)
	{
		X(i / dim, i % dim) = x(i);
	}
	assert(nanproof_equals(X(0, 0), x(0)));
	assert(X.cols() <= 1 || nanproof_equals(X(0, 1), x(1)));
	return X;
}

void polyfem::utils::vector2matrix(const Eigen::VectorXd &vec, Eigen::MatrixXd &mat)
{
	int size = sqrt(vec.size());
	assert(size * size == vec.size());

	mat = unflatten(vec, size);
}

Eigen::SparseMatrix<double> polyfem::utils::lump_matrix(const Eigen::SparseMatrix<double> &M)
{
	std::vector<Eigen::Triplet<double>> triplets;

	for (int k = 0; k < M.outerSize(); ++k)
	{
		for (Eigen::SparseMatrix<double>::InnerIterator it(M, k); it; ++it)
		{
			triplets.emplace_back(it.row(), it.row(), it.value());
		}
	}

	Eigen::SparseMatrix<double> lumped(M.rows(), M.rows());
	lumped.setFromTriplets(triplets.begin(), triplets.end());
	lumped.makeCompressed();

	return lumped;
}

void polyfem::utils::full_to_reduced_matrix(
	const int full_size,
	const int reduced_size,
	const std::vector<int> &removed_vars,
	const StiffnessMatrix &full,
	StiffnessMatrix &reduced)
{
	POLYFEM_SCOPED_TIMER("full to reduced matrix");

	if (reduced_size == full_size || reduced_size == full.rows())
	{
		assert(reduced_size == full.rows() && reduced_size == full.cols());
		reduced = full;
		return;
	}
	assert(full.rows() == full_size && full.cols() == full_size);

	Eigen::VectorXi indices(full_size);
	int index = 0;
	size_t kk = 0;
	for (int i = 0; i < full_size; ++i)
	{
		if (kk < removed_vars.size() && removed_vars[kk] == i)
		{
			++kk;
			indices(i) = -1;
		}
		else
		{
			indices(i) = index++;
		}
	}
	assert(index == reduced_size);

	std::vector<Eigen::Triplet<double>> entries;
	entries.reserve(full.nonZeros()); // Conservative estimate
	for (int k = 0; k < full.outerSize(); ++k)
	{
		if (indices(k) < 0)
			continue;

		for (StiffnessMatrix::InnerIterator it(full, k); it; ++it)
		{
			assert(it.col() == k);
			if (indices(it.row()) < 0 || indices(it.col()) < 0)
				continue;

			assert(indices(it.row()) >= 0);
			assert(indices(it.col()) >= 0);

			entries.emplace_back(indices(it.row()), indices(it.col()), it.value());
		}
	}

	reduced.resize(reduced_size, reduced_size);
	reduced.setFromTriplets(entries.begin(), entries.end());
	reduced.makeCompressed();
}

Eigen::MatrixXd polyfem::utils::reorder_matrix(
	const Eigen::MatrixXd &in,
	const Eigen::VectorXi &in_to_out,
	int out_blocks,
	const int block_size)
{
	constexpr double NaN = std::numeric_limits<double>::quiet_NaN();

	assert(in.rows() % block_size == 0);
	assert(in_to_out.size() == in.rows() / block_size);

	if (out_blocks < 0)
		out_blocks = in.rows() / block_size;

	Eigen::MatrixXd out = Eigen::MatrixXd::Constant(
		out_blocks * block_size, in.cols(), NaN);

	const int in_blocks = in.rows() / block_size;
	for (int i = 0; i < in_blocks; ++i)
	{
		const int j = in_to_out[i];
		if (j < 0)
			continue;

		out.middleRows(block_size * j, block_size) =
			in.middleRows(block_size * i, block_size);
	}

	return out;
}

Eigen::MatrixXd polyfem::utils::unreorder_matrix(
	const Eigen::MatrixXd &out,
	const Eigen::VectorXi &in_to_out,
	int in_blocks,
	const int block_size)
{
	constexpr double NaN = std::numeric_limits<double>::quiet_NaN();

	assert(out.rows() % block_size == 0);

	if (in_blocks < 0)
		in_blocks = out.rows() / block_size;
	assert(in_to_out.size() == in_blocks);

	Eigen::MatrixXd in = Eigen::MatrixXd::Constant(
		in_blocks * block_size, out.cols(), NaN);

	for (int i = 0; i < in_blocks; i++)
	{
		const int j = in_to_out[i];
		if (j < 0)
			continue;

		in.middleRows(block_size * i, block_size) =
			out.middleRows(block_size * j, block_size);
	}

	return in;
}

Eigen::MatrixXi polyfem::utils::map_index_matrix(
	const Eigen::MatrixXi &in,
	const Eigen::VectorXi &index_mapping)
{
	Eigen::MatrixXi out(in.rows(), in.cols());
	for (int i = 0; i < in.rows(); i++)
	{
		for (int j = 0; j < in.cols(); j++)
		{
			out(i, j) = index_mapping[in(i, j)];
		}
	}
	return out;
}
