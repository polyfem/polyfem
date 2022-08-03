#pragma once

#include <polyfem/assembler/MassMatrixAssembler.hpp>

namespace polyfem::utils
{

	/// @brief Project the quantities in u on to the space spanned by mesh.bases.
	void L2_projection(
		const bool is_volume,
		const int size,
		const int n_basis_a,
		const std::vector<polyfem::basis::ElementBases> &bases_a,
		const std::vector<polyfem::basis::ElementBases> &gbases_a,
		const int n_basis_b,
		const std::vector<polyfem::basis::ElementBases> &bases_b,
		const std::vector<polyfem::basis::ElementBases> &gbases_b,
		const polyfem::assembler::AssemblyValsCache &cache,
		const Eigen::MatrixXd &y,
		Eigen::MatrixXd &x,
		const bool lump_mass_matrix = false);

} // namespace polyfem::utils