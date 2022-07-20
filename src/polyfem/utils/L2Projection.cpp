#include "L2Projection.hpp"

#include <polysolve/LinearSolver.hpp>

#include <polyfem/utils/MatrixUtils.hpp>

namespace polyfem::utils
{

	using namespace polysolve;
	using namespace polyfem::basis;
	using namespace polyfem::assembler;

	/// @brief Project the quantities in u on to the space spanned by mesh.bases.
	void L2_projection(
		const bool is_volume,
		const int size,
		const int n_basis_a,
		const std::vector<ElementBases> &bases_a,
		const std::vector<ElementBases> &gbases_a,
		const int n_basis_b,
		const std::vector<ElementBases> &bases_b,
		const std::vector<ElementBases> &gbases_b,
		const Density &density,
		const AssemblyValsCache &cache,
		const Eigen::VectorXd &u,
		Eigen::VectorXd &u_proj)
	{
		MassMatrixAssembler assembler;

		Eigen::SparseMatrix<double> M;
		assembler.assemble(
			is_volume, size, n_basis_b, density, bases_b, gbases_b, cache, M);

		write_sparse_matrix_csv("M.csv", M);

		Eigen::SparseMatrix<double> A;
		assembler.assemble_cross(
			is_volume, size,
			n_basis_a, bases_a, gbases_a,
			n_basis_b, bases_b, gbases_b,
			cache, A);

		write_sparse_matrix_csv("A.csv", A);

		std::unique_ptr<LinearSolver> linear_solver;
		linear_solver = LinearSolver::create("Eigen::SparseLU", LinearSolver::defaultPrecond());
		// linear_solver->setParameters(solver_params);

		Eigen::SparseMatrix<double> LHS = M.transpose() * M;
		Eigen::VectorXd rhs = M * A * u;

		linear_solver->analyzePattern(LHS, LHS.rows());
		linear_solver->factorize(LHS);
		linear_solver->solve(rhs, u_proj);
		assert((LHS * u_proj - rhs).norm() < 1e-12);
	}

} // namespace polyfem::utils