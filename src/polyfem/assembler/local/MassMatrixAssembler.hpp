#pragma once

#include <polyfem/assembler/ElementAssemblyValues.hpp>
#include <polyfem/assembler/AssemblyValsCache.hpp>
#include <polyfem/assembler/MatParams.hpp>

#include <Eigen/Sparse>
#include <vector>
#include <iostream>
#include <cmath>
#include <memory>

// mass matrix assembler
namespace polyfem::assembler
{
	class MassMatrixAssembler
	{
	public:
		/// @brief Assembles the cross mass matrix between to function spaces.
		/// @param[in]  is_volume    True if the mesh is volumetric.
		/// @param[in]  size         Size of the problem (e.g., 1 for Laplace).
		/// @param[in]  n_from_basis Number of basis functions in the initial function space.
		/// @param[in]  from_bases   Finite element bases to map from.
		/// @param[in]  from_gbases  Geometric bases to map from.
		/// @param[in]  n_from_basis Number of basis functions in the resulting function space.
		/// @param[in]  from_bases   Finite element bases to map to.
		/// @param[in]  from_gbases  Geometric bases to map to.
		/// @param[in]  cache        Assembly values cache.
		/// @param[out] mass         Output constructed mass matrix.
		void assemble_cross(
			const bool is_volume,
			const int size,
			const int n_from_basis,
			const std::vector<basis::ElementBases> &from_bases,
			const std::vector<basis::ElementBases> &from_gbases,
			const int n_to_basis,
			const std::vector<basis::ElementBases> &to_bases,
			const std::vector<basis::ElementBases> &to_gbases,
			const AssemblyValsCache &cache,
			StiffnessMatrix &mass) const;
	};
} // namespace polyfem::assembler
