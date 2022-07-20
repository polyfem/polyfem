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
		const polyfem::Density &density,
		const polyfem::assembler::AssemblyValsCache &cache,
		const Eigen::VectorXd &u,
		Eigen::VectorXd &u_proj);

} // namespace polyfem::utils