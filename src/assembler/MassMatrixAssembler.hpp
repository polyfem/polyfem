#pragma once

#include <polyfem/ElementAssemblyValues.hpp>
#include <polyfem/AssemblyValsCache.hpp>
#include <polyfem/ElasticityUtils.hpp>

#include <Eigen/Sparse>
#include <vector>
#include <iostream>
#include <cmath>
#include <memory>

//mass matrix assembler
namespace polyfem
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
			const std::vector<ElementBases> &bases,
			const std::vector<ElementBases> &gbases,
			const AssemblyValsCache &cache,
			StiffnessMatrix &mass) const;
	};
} // namespace polyfem
