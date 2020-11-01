#pragma once

#include <polyfem/ElementAssemblyValues.hpp>
#include <polyfem/ElasticityUtils.hpp>

#include <Eigen/Sparse>
#include <vector>
#include <iostream>
#include <cmath>
#include <memory>

namespace polyfem
{
	class MassMatrixAssembler
	{
	public:
		void assemble(
			const bool is_volume,
			const int size,
			const int n_basis,
			const Density &density,
			const std::vector<ElementBases> &bases,
			const std::vector<ElementBases> &gbases,
			StiffnessMatrix &mass) const;
	};
}
