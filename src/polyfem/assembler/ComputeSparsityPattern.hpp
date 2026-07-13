#pragma once

#include <polyfem/assembler/AssemblyData.hpp>
#include <polyfem/utils/BlockCSRMatrix.hpp>

namespace polyfem::assembler
{

	/// @brief Compute sparsity pattern from basis info.
	/// @param data Assemby data.
	/// @param node_num Total node number.
	/// @param block_dim Block dimension. 1/2/3.
	BSRSparsityPattern compute_sparsity_pattern(
		const AssemblyDataView &data,
		int node_num,
		int block_dim);

} // namespace polyfem::assembler
