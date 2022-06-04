#pragma once

#include <polyfem/assembler/ElementAssemblyValues.hpp>
#include <polyfem/assembler/AssemblyValsCache.hpp>
#include <polyfem/utils/ElasticityUtils.hpp>

#include <Eigen/Sparse>
#include <vector>
#include <iostream>
#include <cmath>
#include <memory>

//mass matrix assembler
namespace polyfem
{
	namespace assembler
	{
		class MassMatrixAssembler
		{
		public:
			//assembles the mass matrix
			//mesh is volumetric
			//size of the problem (eg, 1 for laplace)
			//number of bases, bases and geom bases
			//density, class that can evaluate per point density
			void assemble(
				const bool is_volume,
				const int size,
				const int n_basis,
				const Density &density,
				const std::vector<basis::ElementBases> &bases,
				const std::vector<basis::ElementBases> &gbases,
				const AssemblyValsCache &cache,
				StiffnessMatrix &mass) const;
		};
	} // namespace assembler
} // namespace polyfem
