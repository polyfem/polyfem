#include "MatrixUtils.hpp"

////////////////////////////////////////////////////////////////////////////////
#include <FEMSolver.hpp>
#include <unsupported/Eigen/SparseExtra>
////////////////////////////////////////////////////////////////////////////////

void poly_fem::dirichlet_solve(
	LinearSolver &solver, Eigen::SparseMatrix<double> &A, Eigen::VectorXd &f,
	const std::vector<int> &dirichlet_nodes, Eigen::VectorXd &u, const std::string &save_path)
{
	// Let Γ be the set of Dirichlet dofs.
	// To implement nonzero Dirichlet boundary conditions, we seek to replace
	// the linear system Au = f with a new system Ãx = g, where
	// - Ã is the matrix A with rows and cols of i ∈ Γ set to identity
	// - g[i] = f[i] for i ∈ Γ
	// - g[i] = f[i] - Σ_{j ∈ Γ} a_ij f[j] for i ∉ Γ
	// In matrix terms, if we call N = diag({1 iff i ∈ Γ}), then we have that
	// g = f - (I-N)*A*N*f

	int n = A.outerSize();
	Eigen::VectorXd N(n);
	N.setZero();
	for (int i : dirichlet_nodes) { N(i) = 1; }

	Eigen::VectorXd g = f - ((1.0 - N.array()).matrix()).asDiagonal() * (A * (N.asDiagonal() * f));

	if (0) {
		Eigen::MatrixXd rhs(g.size(), 6);
		rhs.col(0) = N;
		rhs.col(1) = f;
		rhs.col(2) = N.asDiagonal() * f;
		rhs.col(3) = A * (N.asDiagonal() * f);
		rhs.col(4) = ((1.0 - N.array()).matrix()).asDiagonal() * (A * (N.asDiagonal() * f));
		rhs.col(5) = g;
		std::cout << rhs << std::endl;
	}

	std::vector<Eigen::Triplet<double>> coeffs;
	coeffs.reserve(A.nonZeros());
	assert(A.rows() == A.cols());
	for (int k = 0; k < A.outerSize(); ++k) {
		for (Eigen::SparseMatrix<double>::InnerIterator it(A, k); it; ++it) {
			// it.value();
			// it.row();   // row index
			// it.col();   // col index (here it is equal to k)
			// it.index(); // inner index, here it is equal to it.row()
			if (N(it.row()) != 1 && N(it.col()) != 1) {
				coeffs.emplace_back(it.row(), it.col(), it.value());
			}
		}
	}
	// TODO: For numerical stability, we should set diagonal values of the same
	// magnitude than the other entries in the matrix
	for (int k = 0; k < n; ++k) {
		coeffs.emplace_back(k, k, N(k));
	}
	// Eigen::saveMarket(A, "A_before.mat");
	A.setFromTriplets(coeffs.begin(), coeffs.end());
	A.makeCompressed();

	// std::cout << A << std::endl;

	// Eigen::saveMarket(A, "A.mat");
	// Eigen::saveMarketVector(g, "b.mat");

	if (u.size() != n) {
		u.resize(n);
		u.setZero();
	}


	solver.analyzePattern(A);
	solver.factorize(A);
	solver.solve(g, u);
	f = g;

	if(!save_path.empty())
		Eigen::saveMarket(A, save_path);
}

