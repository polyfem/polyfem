#include <polyfem/assembler/AssembleOnHost.hpp>

#include <polyfem/assembler/AssemblyCache.hpp>
#include <polyfem/assembler/ComputeAssemblyCache.hpp>
#include <polyfem/assembler/AssemblyData.hpp>
#include <polyfem/utils/AtomicAdd.hpp>
#include <polyfem/utils/BlockCSRMatrix.hpp>
#include <polyfem/utils/Span.hpp>

#include <Eigen/Core>

#include <cassert>

namespace polyfem::assembler::detail
{
	ElementAssemblyCacheView get_element_cache(
		const AssemblyDataView &data,
		const AssemblyDataView &geom_data,
		const AssemblyCacheView &cache,
		int element_id,
		bool is_mass,
		AssemblyTempStorage &temp,
		AssemblyCache &temp_cache)
	{
		// Get pre-computed element cache is possible.
		if (!cache.desc.empty() && !cache.desc[element_id].is_empty)
		{
			ElementAssemblyCacheView view = cache.slice(element_id);
			assert(view.is_mass == is_mass);
			return view;
		}

		// Compute assembly cache on the fly.
		temp_cache.clear();
		int dim = data.element_desc[element_id].basis_desc.dim;
		switch (dim)
		{
		case 1:
			compute_assembly_cache_single<1>(data, geom_data, element_id, is_mass, temp);
			break;
		case 2:
			compute_assembly_cache_single<2>(data, geom_data, element_id, is_mass, temp);
			break;
		case 3:
			compute_assembly_cache_single<3>(data, geom_data, element_id, is_mass, temp);
			break;
		default:
			assert(false);
		}
		int cache_element_id = temp_cache.append(is_mass, temp);
		assert(cache_element_id == 0);
		return temp_cache.view().slice(cache_element_id);
	}

	void scatter_element_vector(
		const ElementDesc &element_desc,
		DofMappingStoreView mapping,
		Span<const double> local_vector,
		int basis_num,
		int value_dim,
		Span<double> global_vector)
	{
		assert(local_vector.size() == basis_num * value_dim);
		for (int basis_id = 0; basis_id < basis_num; ++basis_id)
		{
			int mapping_id = element_desc.dof_mapping_range.offset + basis_id;
			auto node_ids = mapping.get_node_ids(mapping_id);
			auto node_weights = mapping.get_weights(mapping_id);
			assert(node_ids.size() == node_weights.size());

			for (int d = 0; d < value_dim; ++d)
			{
				double local_value = local_vector[basis_id * value_dim + d];
				if (local_value == 0.0)
				{
					continue;
				}

				for (int node_id = 0; node_id < node_ids.size(); ++node_id)
				{
					int offset = node_ids[node_id] * value_dim + d;
					assert(offset >= 0 && offset < global_vector.size());
					global_vector[offset] += node_weights[node_id] * local_value;
				}
			}
		}
	}

	void scatter_element_matrix(
		const ElementDesc &element_desc,
		DofMappingStoreView mapping,
		const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &local_matrix,
		int basis_num,
		int value_dim,
		BSRMatrixMutableView global_matrix)
	{
		int local_dof_num = basis_num * value_dim;
		assert(local_matrix.rows() == local_dof_num);
		assert(local_matrix.cols() == local_dof_num);

		// Loop element basis node ij.
		for (int bi = 0; bi < basis_num; ++bi)
		{
			int row_mapping_id = element_desc.dof_mapping_range.offset + bi;
			auto row_node_ids = mapping.get_node_ids(row_mapping_id);
			auto row_weights = mapping.get_weights(row_mapping_id);
			assert(row_node_ids.size() == row_weights.size());

			for (int bj = 0; bj < basis_num; ++bj)
			{
				// ij component of element local matrix M.
				// Represents contribution form element local basis node i and j.
				auto local_block = local_matrix.block(
					bi * value_dim,
					bj * value_dim,
					value_dim,
					value_dim);
				if (local_block.isZero())
				{
					continue;
				}

				int col_mapping_id = element_desc.dof_mapping_range.offset + bj;
				auto col_node_ids = mapping.get_node_ids(col_mapping_id);
				auto col_weights = mapping.get_weights(col_mapping_id);
				assert(col_node_ids.size() == col_weights.size());

				// Loop basis node i local to global node mappings.
				for (int row_node_id = 0; row_node_id < row_node_ids.size(); ++row_node_id)
				{
					// Loop basis node j local to global node mappings.
					for (int col_node_id = 0; col_node_id < col_node_ids.size(); ++col_node_id)
					{
						double weight = row_weights[row_node_id] * col_weights[col_node_id];

						for (int r = 0; r < value_dim; ++r)
						{
							for (int c = 0; c < value_dim; ++c)
							{
								double value = local_block(r, c);
								if (value != 0.0)
								{
									int row = row_node_ids[row_node_id] * value_dim + r;
									int col = col_node_ids[col_node_id] * value_dim + c;
									double *dst = global_matrix.get_entry(row, col);
									assert(dst);
									utils::atomic_add(*dst, weight * value);
								}
							}
						}
					}
				}
			}
		}
	}

} // namespace polyfem::assembler::detail
