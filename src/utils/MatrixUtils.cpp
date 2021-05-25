#include <polyfem/MatrixUtils.hpp>

#include <polyfem/Logger.hpp>

#include <igl/list_to_matrix.h>

#include <iostream>
#include <fstream>
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
void polyfem::read_matrix(const std::string &path, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &mat)
{
	std::fstream file;
	file.open(path.c_str());

	if (!file.good())
	{
		logger().error("Failed to open file: {}", path);
		file.close();
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
	}
}

//template instantiation
template void polyfem::read_matrix<int>(const std::string &, Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> &);
template void polyfem::read_matrix<double>(const std::string &, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> &);
