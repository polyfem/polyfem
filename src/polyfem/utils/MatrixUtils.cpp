#include "MatrixUtils.hpp"

#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/Timer.hpp>

#include <vector>

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
