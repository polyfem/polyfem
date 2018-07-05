#pragma once

#include <polyfem/ElementAssemblyValues.hpp>

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
			const std::vector< ElementBases > &bases,
			const std::vector< ElementBases > &gbases,
			Eigen::SparseMatrix<double> &mass);
	};
}
