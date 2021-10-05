#include <polyfem/MatrixUtils.hpp>

#include <polyfem/MaybeParallelFor.hpp>
#include <polyfem/Logger.hpp>

#include <igl/list_to_matrix.h>

#include <iostream>
#include <fstream>
#include <iomanip> // setprecision
#include <vector>

void polyfem::show_matrix_stats(const Eigen::MatrixXd &M)
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

template <typename T>
bool polyfem::read_matrix(const std::string &path, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &mat)
{
	std::fstream file;
	file.open(path.c_str());

	if (!file.good())
	{
		logger().error("Failed to open file: {}", path);
		file.close();

		return false;
	}

	std::string s;
	std::vector<std::vector<T>> matrix;

	while (getline(file, s))
	{
		std::stringstream input(s);
		T temp;
		matrix.emplace_back();

		std::vector<T> &currentLine = matrix.back();

		while (input >> temp)
			currentLine.push_back(temp);
	}

	if (!igl::list_to_matrix(matrix, mat))
	{
		logger().error("list to matrix error");
		file.close();

		return false;
	}

	return true;
}

template <typename T>
bool polyfem::read_matrix_binary(const std::string &path, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &mat)
{
	typedef typename Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Index Index;

	std::ifstream in(path, std::ios::in | std::ios::binary);
	if (!in.good())
	{
		logger().error("Failed to open file: {}", path);
		in.close();

		return false;
	}

	Index rows = 0, cols = 0;
	in.read((char *)(&rows), sizeof(Index));
	in.read((char *)(&cols), sizeof(Index));

	mat.resize(rows, cols);
	in.read((char *)mat.data(), rows * cols * sizeof(T));
	in.close();

	return true;
}

template <typename Mat>
bool polyfem::write_matrix_binary(const std::string &path, const Mat &mat)
{
	typedef typename Mat::Index Index;
	typedef typename Mat::Scalar Scalar;
	std::ofstream out(path, std::ios::out | std::ios::binary);

	if (!out.good())
	{
		logger().error("Failed to write to file: {}", path);
		out.close();

		return false;
	}

	const Index rows = mat.rows(), cols = mat.cols();
	out.write((const char *)(&rows), sizeof(Index));
	out.write((const char *)(&cols), sizeof(Index));
	out.write((const char *)mat.data(), rows * cols * sizeof(Scalar));
	out.close();

	return true;
}

bool polyfem::write_sparse_matrix_csv(const std::string &path, const Eigen::SparseMatrix<double> &mat)
{
	std::ofstream csv(path, std::ios::out);

	if (!csv.good())
	{
		logger().error("Failed to write to file: {}", path);
		csv.close();

		return false;
	}

	csv << std::setprecision(std::numeric_limits<long double>::digits10 + 2);

	csv << fmt::format("shape,{},{}\n", mat.rows(), mat.cols());
	csv << "Row,Col,Val\n";
	for (int k = 0; k < mat.outerSize(); ++k)
	{
		for (Eigen::SparseMatrix<double>::InnerIterator it(mat, k); it; ++it)
		{
			csv << it.row() << "," // row index
				<< it.col() << "," // col index (here it is equal to k)
				<< it.value() << "\n";
		}
	}
	csv.close();

	return true;
}

polyfem::SpareMatrixCache::SpareMatrixCache(const size_t size)
	: size_(size)
{
	tmp_.resize(size_, size_);
	mat_.resize(size_, size_);
	mat_.setZero();
}

polyfem::SpareMatrixCache::SpareMatrixCache(const size_t rows, const size_t cols)
	: size_(rows == cols ? rows : 0)
{
	tmp_.resize(rows, cols);
	mat_.resize(rows, cols);
	mat_.setZero();
}

polyfem::SpareMatrixCache::SpareMatrixCache(const SpareMatrixCache &other)
{
	init(other);
}

void polyfem::SpareMatrixCache::init(const size_t size)
{
	assert(mapping().empty() || size_ == size);

	size_ = size;
	tmp_.resize(size_, size_);
	mat_.resize(size_, size_);
	mat_.setZero();
}

void polyfem::SpareMatrixCache::init(const size_t rows, const size_t cols)
{
	assert(mapping().empty());

	size_ = rows == cols ? rows : 0;
	tmp_.resize(rows, cols);
	mat_.resize(rows, cols);
	mat_.setZero();
}

void polyfem::SpareMatrixCache::init(const SpareMatrixCache &other)
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

void polyfem::SpareMatrixCache::set_zero()
{
	tmp_.setZero();
	mat_.setZero();

	std::fill(values_.begin(), values_.end(), 0);
}

void polyfem::SpareMatrixCache::add_value(const int i, const int j, const double value)
{
	if (mapping().empty())
	{
		entries_.emplace_back(i, j, value);
	}
	else
	{
		//mapping()[i].find(j)
		const auto &map = mapping()[i];
		bool found = false;
		for (const auto &p : map)
		{
			if (p.first == j)
			{
				assert(p.second < values_.size());
				values_[p.second] += value;
				found = true;
			}
		}
		assert(found);
	}
}

void polyfem::SpareMatrixCache::prune()
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

polyfem::StiffnessMatrix polyfem::SpareMatrixCache::get_matrix(const bool compute_mapping)
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
		}
	}
	else
	{
		assert(size_ > 0);
		const auto &outer_index = main_cache_ == nullptr ? outer_index_ : main_cache_->outer_index_;
		const auto &inner_index = main_cache_ == nullptr ? inner_index_ : main_cache_->inner_index_;
		mat_ = Eigen::Map<const StiffnessMatrix>(size_, size_, values_.size(), &outer_index[0], &inner_index[0], &values_[0]);
		logger().trace("Using cache");
	}
	std::fill(values_.begin(), values_.end(), 0);
	return mat_;
}

polyfem::SpareMatrixCache polyfem::SpareMatrixCache::operator+(const SpareMatrixCache &a) const
{
	SpareMatrixCache out(a);

	if (a.mapping().empty() || mapping().empty())
	{
		out.mat_ = a.mat_ + mat_;
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

void polyfem::SpareMatrixCache::operator+=(const SpareMatrixCache &o)
{
	if (mapping().empty() || o.mapping().empty())
	{
		mat_ += o.mat_;
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

//template instantiation
template bool polyfem::read_matrix<int>(const std::string &, Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> &);
template bool polyfem::read_matrix<double>(const std::string &, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> &);

template bool polyfem::read_matrix_binary<int>(const std::string &, Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> &);
template bool polyfem::read_matrix_binary<double>(const std::string &, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> &);

template bool polyfem::write_matrix_binary<Eigen::MatrixXd>(const std::string &, const Eigen::MatrixXd &);
template bool polyfem::write_matrix_binary<Eigen::MatrixXf>(const std::string &, const Eigen::MatrixXf &);
template bool polyfem::write_matrix_binary<Eigen::VectorXd>(const std::string &, const Eigen::VectorXd &);
template bool polyfem::write_matrix_binary<Eigen::VectorXf>(const std::string &, const Eigen::VectorXf &);
