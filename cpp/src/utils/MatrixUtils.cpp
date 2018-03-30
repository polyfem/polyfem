#include "MatrixUtils.hpp"

#include <igl/list_to_matrix.h>
#include <MatOp/SparseSymMatProd.h>
#include <MatOp/SparseSymShiftSolve.h>
#include <SymEigsSolver.h>
#include <SymEigsShiftSolver.h>


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

Eigen::Vector4d poly_fem::compute_specturm(const Eigen::SparseMatrix<double> &mat)
{
	typedef Spectra::SparseSymMatProd<double> MatOp;
	typedef Spectra::SparseSymShiftSolve<double> InvMatOp;
	Eigen::Vector4d res;
	res.setConstant(NAN);

	InvMatOp invOpt(mat);
	Spectra::SymEigsShiftSolver< double, Spectra::LARGEST_MAGN, InvMatOp> small_eig(&invOpt, 2, 4, 0);

	small_eig.init();
    const int n_small = small_eig.compute(1000, 1e-8, Spectra::SMALLEST_MAGN);
    if(small_eig.info() == Spectra::SUCCESSFUL){
        res(0) = small_eig.eigenvalues()(0);
        res(1) = small_eig.eigenvalues()(1);
    }


    MatOp op(mat);
    Spectra::SymEigsSolver< double, Spectra::LARGEST_MAGN, MatOp> large_eig(&op, 2, 4);

	large_eig.init();
    const int n_large = large_eig.compute(1000, 1e-8, Spectra::LARGEST_MAGN);
    if(large_eig.info() == Spectra::SUCCESSFUL){
        res(2) = large_eig.eigenvalues()(1);
        res(3) = large_eig.eigenvalues()(0);
    }


	return res;
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

