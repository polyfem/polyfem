#pragma once

#include <polyfem/Types.hpp>

#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace polyfem
{

	// Show some stats about the matrix M: det, singular values, condition number, etc
	void show_matrix_stats(const Eigen::MatrixXd &M);

	template <typename T>
	T determinant(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> &mat)
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
	bool read_matrix(const std::string &path, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &mat);

	template <typename T>
	bool read_matrix_binary(const std::string &path, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &mat);

	template <typename Mat>
	bool write_matrix_binary(const std::string &path, const Mat &mat);

	class SpareMatrixCache
	{
	public:
		SpareMatrixCache() {}
		SpareMatrixCache(const size_t size);
		SpareMatrixCache(const size_t rows, const size_t cols);
		SpareMatrixCache(const SpareMatrixCache &other);

		void init(const size_t size);
		void init(const size_t rows, const size_t cols);
		void init(const SpareMatrixCache &other);

		void set_zero();

		inline void reserve(const size_t size) { entries_.reserve(size); }
		inline size_t entries_size() const { return entries_.size(); }
		inline size_t capacity() const { return entries_.capacity(); }
		inline size_t non_zeros() const { return mapping_.empty() ? mat_.nonZeros() : values_.size(); }

		void add_value(const int i, const int j, const double value);
		StiffnessMatrix get_matrix(const bool compute_mapping = true);
		void prune();

		SpareMatrixCache operator+(const SpareMatrixCache &a) const;
		void operator+=(const SpareMatrixCache &o);

	private:
		size_t size_;
		StiffnessMatrix tmp_, mat_;
		std::vector<Eigen::Triplet<double>> entries_;
		std::vector<std::vector<std::pair<int, size_t>>> mapping_;
		std::vector<int> inner_index_, outer_index_;
		std::vector<double> values_;
	};

} // namespace polyfem
