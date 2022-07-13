#include <polyfem/utils/L2Projection.hpp>
#include <polyfem/assembler/MassMatrixAssembler.hpp>

#include <Eigen/LU>
#include <Eigen/SparseCore>

namespace polyfem::utils
{

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
		const AssemblyValsCache &cache,
		const Eigen::VectorXd &u,
		const Eigen::VectorXd &u_proj)
	{
		MassMatrixAssembler assembler;

		Eigen::SparseMatrix<double> M;
		assembler.assemble(
			is_volume, size, n_basis_b, density, bases_b, gbases_b, cache, M);

		Eigen::SparseMatrix<double> A;
		assembler.assemble_cross(
			is_volume, size,
			n_basis_a, bases_a, gbases_a,
			n_basis_b, bases_b, gbases_b,
			cache, A);

		u_proj = M.ldlt().solve(A * u);
		assert(np.linalg.norm(M * u_proj - A * u) < 1e-12);

		return u_proj;
	}

	class L2Projection
	{
		void build();
		void project();
	}

} // namespace polyfem::utils