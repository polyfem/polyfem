#include "MatrixUtils.hpp"

#include <igl/list_to_matrix.h>

#include <iostream>
#include <fstream>
#include <vector>

void poly_fem::show_matrix_stats(const Eigen::MatrixXd &M) {
	Eigen::FullPivLU<Eigen::MatrixXd> lu(M);
	Eigen::JacobiSVD<Eigen::MatrixXd> svd(M);
	double s1 = svd.singularValues()(0);
	double s2 = svd.singularValues()(svd.singularValues().size()-1);
	double cond = s1 / s2;

	std::cout << "----------------------------------------" << std::endl;
	std::cout << "-- Determinant: " << M.determinant() << std::endl;
	std::cout << "-- Singular values: " << s1 << " " << s2 << std::endl;
	std::cout << "-- Cond: " << cond << std::endl;
	std::cout << "-- Invertible: " << lu.isInvertible() << std::endl;
	std::cout << "----------------------------------------" << std::endl;
	std::cout << lu.solve(M) << std::endl;
}


template<typename T>
void poly_fem::read_matrix(const std::string &path, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &mat)
{
	std::fstream file;
	file.open(path.c_str());

	if (!file.good())
	{
		std::cerr << "Failed to open file: " << path << std::endl;
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
		std::cerr << "list to matrix error" << std::endl;
		file.close();
	}
}

//template instantiation
template void poly_fem::read_matrix<int>(const std::string &, Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> &);
template void poly_fem::read_matrix<double>(const std::string &, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> &);

