#include "MatrixUtils.hpp"
#include <iostream>

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
