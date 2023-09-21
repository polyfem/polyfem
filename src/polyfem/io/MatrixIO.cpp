#include "MatrixIO.hpp"

#include <polyfem/utils/Logger.hpp>

#include <igl/list_to_matrix.h>

#include <iostream>
#include <h5pp/h5pp.h>

#include <fstream>
#include <iomanip> // setprecision
#include <vector>
#include <filesystem>

namespace polyfem::io
{
	template <typename T>
	bool read_matrix(const std::string &path, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &mat)
	{
		std::string extension = std::filesystem::path(path).extension().string();
		std::transform(extension.begin(), extension.end(), extension.begin(),
					   [](unsigned char c) { return std::tolower(c); });

		if (extension == ".txt")
		{
			return read_matrix_ascii(path, mat);
		}
		else if (extension == ".bin")
		{
			return read_matrix_binary(path, mat);
		}
		else
		{
			bool success = read_matrix_ascii(path, mat);
			if (!success)
				success = read_matrix_binary(path, mat); // Try with the binary format reader
			return success;
		}
	}

	template <typename Mat>
	bool write_matrix(const std::string &path, const Mat &mat)
	{
		std::string extension = std::filesystem::path(path).extension().string();
		std::transform(extension.begin(), extension.end(), extension.begin(),
					   [](unsigned char c) { return std::tolower(c); });

		if (extension == ".txt")
		{
			return write_matrix_ascii(path, mat);
		}
		else if (extension == ".bin")
		{
			return write_matrix_binary(path, mat);
		}
		else
		{
			logger().warn("Uknown output matrix format (\"{}\"). Using ASCII format.");
			return write_matrix_ascii(path, mat);
		}
	}

	template <typename Mat>
	bool write_matrix(const std::string &path, const std::string &key, const Mat &mat, const bool replace)
	{
		h5pp::File hdf5_file(path, replace ? h5pp::FileAccess::REPLACE : h5pp::FileAccess::READWRITE);
		hdf5_file.writeDataset(mat, key);

		return true;
	}

	template <typename Mat>
	bool read_matrix(const std::string &path, const std::string &key, Mat &mat)
	{
		h5pp::File hdf5_file(path, h5pp::FileAccess::READONLY);
		mat = hdf5_file.readDataset<Mat>(key);
		return true;
	}

	template <typename T>
	bool read_matrix_ascii(const std::string &path, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &mat)
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
			if (s.empty())
				continue;
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

	template <typename Mat>
	bool write_matrix_ascii(const std::string &path, const Mat &mat)
	{
		std::ofstream out(path);
		if (!out.good())
		{
			logger().error("Failed to write to file: {}", path);
			out.close();

			return false;
		}

		out.precision(15);
		out << mat;
		out.close();

		return true;
	}

	template <typename T>
	bool read_matrix_binary(const std::string &path, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &mat)
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
	bool write_matrix_binary(const std::string &path, const Mat &mat)
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

	bool write_sparse_matrix_csv(const std::string &path, const Eigen::SparseMatrix<double> &mat)
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

	template <typename T>
	bool import_matrix(
		const std::string &path, const json &import, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &mat)
	{
		bool success;
		if (import.contains("offset"))
		{
			const int offset = import["offset"];

			Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> tmp;
			success = read_matrix(path, tmp);
			if (success)
			{
				assert(mat.rows() >= offset && mat.cols() >= 1);
				mat.block(0, 0, offset, 1) = tmp.block(0, 0, offset, 1);
			}
		}
		else
		{
			success = read_matrix(path, mat);
		}
		return success;
	}

	// template instantiation
	template bool read_matrix<int>(const std::string &, Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> &);
	template bool read_matrix<double>(const std::string &, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> &);

	template bool read_matrix<Eigen::MatrixXi>(const std::string &, const std::string &, Eigen::MatrixXi &);
	template bool read_matrix<Eigen::MatrixXd>(const std::string &, const std::string &, Eigen::MatrixXd &);

	template bool write_matrix<Eigen::MatrixXd>(const std::string &, const Eigen::MatrixXd &);
	template bool write_matrix<Eigen::MatrixXf>(const std::string &, const Eigen::MatrixXf &);
	template bool write_matrix<Eigen::VectorXd>(const std::string &, const Eigen::VectorXd &);
	template bool write_matrix<Eigen::VectorXf>(const std::string &, const Eigen::VectorXf &);

	template bool write_matrix<Eigen::MatrixXd>(const std::string &, const std::string &, const Eigen::MatrixXd &, const bool);
	template bool write_matrix<Eigen::MatrixXf>(const std::string &, const std::string &, const Eigen::MatrixXf &, const bool);
	template bool write_matrix<Eigen::VectorXd>(const std::string &, const std::string &, const Eigen::VectorXd &, const bool);
	template bool write_matrix<Eigen::VectorXf>(const std::string &, const std::string &, const Eigen::VectorXf &, const bool);

	template bool read_matrix_ascii<int>(const std::string &, Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> &);
	template bool read_matrix_ascii<double>(const std::string &, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> &);

	template bool write_matrix_ascii<Eigen::MatrixXd>(const std::string &, const Eigen::MatrixXd &);
	template bool write_matrix_ascii<Eigen::MatrixXf>(const std::string &, const Eigen::MatrixXf &);
	template bool write_matrix_ascii<Eigen::VectorXd>(const std::string &, const Eigen::VectorXd &);
	template bool write_matrix_ascii<Eigen::VectorXf>(const std::string &, const Eigen::VectorXf &);

	template bool read_matrix_binary<int>(const std::string &, Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> &);
	template bool read_matrix_binary<double>(const std::string &, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> &);

	template bool write_matrix_binary<Eigen::MatrixXd>(const std::string &, const Eigen::MatrixXd &);
	template bool write_matrix_binary<Eigen::MatrixXf>(const std::string &, const Eigen::MatrixXf &);
	template bool write_matrix_binary<Eigen::VectorXd>(const std::string &, const Eigen::VectorXd &);
	template bool write_matrix_binary<Eigen::VectorXf>(const std::string &, const Eigen::VectorXf &);

	template bool import_matrix<int>(const std::string &path, const json &import, Eigen::MatrixXi &mat);
	template bool import_matrix<double>(const std::string &path, const json &import, Eigen::MatrixXd &mat);

} // namespace polyfem::io