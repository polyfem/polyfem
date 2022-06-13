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

		inline Eigen::SparseMatrix<double> sparse_identity(int rows, int cols)
		{
			Eigen::SparseMatrix<double> I(rows, cols);
			I.setIdentity();
			return I;
		}

		/// Reads a matrix from a file. Determines the file format based on the path's extension.
		template <typename T>
		bool read_matrix(const std::string &path, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &mat);

		/// Writes a matrix to a file. Determines the file format based on the path's extension.
		template <typename Mat>
		bool write_matrix(const std::string &path, const Mat &mat);

		template <typename T>
		bool read_matrix_ascii(const std::string &path, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &mat);

		template <typename Mat>
		bool write_matrix_ascii(const std::string &path, const Mat &mat);

		template <typename T>
		bool read_matrix_binary(const std::string &path, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &mat);

		template <typename Mat>
		bool write_matrix_binary(const std::string &path, const Mat &mat);

		bool write_sparse_matrix_csv(const std::string &path, const Eigen::SparseMatrix<double> &mat);

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
			inline size_t mapping_size() const { return mapping_.size(); }

			void add_value(const int e, const int i, const int j, const double value);
			StiffnessMatrix get_matrix(const bool compute_mapping = true);
			void prune();

			SpareMatrixCache operator+(const SpareMatrixCache &a) const;
			void operator+=(const SpareMatrixCache &o);

			const StiffnessMatrix &mat() const { return mat_; }
			const std::vector<Eigen::Triplet<double>> &entries() const { return entries_; }

		private:
			size_t size_;
			StiffnessMatrix tmp_, mat_;
			std::vector<Eigen::Triplet<double>> entries_;
			std::vector<std::vector<std::pair<int, size_t>>> mapping_;
			std::vector<int> inner_index_, outer_index_;
			std::vector<double> values_;
			const SpareMatrixCache *main_cache_ = nullptr;

			std::vector<std::vector<int>> second_cache_;
			std::vector<std::vector<std::pair<int, int>>> second_cache_entries_;
			bool use_second_cache_ = true;
			int current_e_ = -1;
			int current_e_index_ = -1;

			inline const std::vector<std::vector<std::pair<int, size_t>>> &mapping() const
			{
				return main_cache_ == nullptr ? mapping_ : main_cache_->mapping_;
			}

			inline const std::vector<std::vector<int>> &second_cache() const
			{
				return main_cache_ == nullptr ? second_cache_ : main_cache_->second_cache_;
			}
		};

		// Flatten rowwises
		Eigen::VectorXd flatten(const Eigen::MatrixXd &X);

		// Unflatten rowwises, so every dim elements in x become a row.
		Eigen::MatrixXd unflatten(const Eigen::VectorXd &x, int dim);
	} // namespace utils
} // namespace polyfem

namespace std
{
	// https://github.com/ethz-asl/map_api/blob/master/map-api-common/include/map-api-common/eigen-hash.h
	template <typename Scalar, int Rows, int Cols>
	struct hash<Eigen::Matrix<Scalar, Rows, Cols>>
	{
		// https://wjngkoh.wordpress.com/2015/03/04/c-hash-function-for-eigen-matrix-and-vector/
		size_t operator()(const Eigen::Matrix<Scalar, Rows, Cols> &matrix) const
		{
			size_t seed = 0;
			for (size_t i = 0; i < matrix.size(); ++i)
			{
				Scalar elem = *(matrix.data() + i);
				seed ^=
					std::hash<Scalar>()(elem) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
			}
			return seed;
		}
	};
} // namespace std
