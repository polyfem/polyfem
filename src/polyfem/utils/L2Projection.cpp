#include "L2Projection.hpp"

#include <polyfem/utils/MatrixUtils.hpp>
#include <polyfem/utils/Logger.hpp>

#include <Eigen/SparseLU>

namespace polyfem::utils
{
	using namespace polyfem::basis;
	using namespace polyfem::assembler;

	/// @brief Project the quantities in u on to the space spanned by mesh.bases.
	void L2_projection(
		const bool is_volume,
		const int size,
		const int n_from_basis,
		const std::vector<ElementBases> &from_bases,
		const std::vector<ElementBases> &from_gbases,
		const int n_to_basis,
		const std::vector<ElementBases> &to_bases,
		const std::vector<ElementBases> &to_gbases,
		const AssemblyValsCache &cache,
		const Eigen::MatrixXd &y,
		Eigen::MatrixXd &x)
	{
		// solve M x = A y for x where M is the mass matrix and A is the cross mass matrix.
		Eigen::SparseMatrix<double> M, A;
		{
			MassMatrixAssembler assembler;
			Density no_density; // Density of one (i.e., no scaling of mass matrix)
			assembler.assemble(
				is_volume, size,
				n_to_basis, no_density, to_bases, to_gbases,
				cache, M);

			assembler.assemble_cross(
				is_volume, size,
				n_from_basis, from_bases, from_gbases,
				n_to_basis, to_bases, to_gbases,
				cache, A);

			// write_sparse_matrix_csv("M.csv", M);
			// write_sparse_matrix_csv("A.csv", A);
			// logger().critical("M =\n{}", Eigen::MatrixXd(M));
			// logger().critical("A =\n{}", Eigen::MatrixXd(A));
		}

		// Construct a linear solver for M
		Eigen::SparseLU<decltype(M)> solver;
		// linear_solver->setParameters(solver_params);
		const Eigen::SparseMatrix<double> &LHS = M; // NOTE: remove & if you want to have a more complicated LHS
		solver.analyzePattern(LHS);
		solver.factorize(LHS);

		const Eigen::MatrixXd rhs = A * y;
		x.resize(rhs.rows(), rhs.cols());
		x = solver.solve(rhs);
		double residual_error = (LHS * x - rhs).norm();
		logger().critical("residual error in L2 projection: {}", residual_error);
		assert(residual_error < 1e-12);
	}

} // namespace polyfem::utils