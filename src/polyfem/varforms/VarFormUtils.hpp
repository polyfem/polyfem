#pragma once

#include <polyfem/Common.hpp>

#include <polyfem/basis/ElementBases.hpp>
#include <polyfem/io/Evaluator.hpp>
#include <polyfem/io/OutputData.hpp>
#include <polyfem/mesh/LocalBoundary.hpp>

#include <unsupported/Eigen/SparseExtra>

#include <algorithm>
#include <map>
#include <unordered_map>
#include <vector>

namespace polyfem::varform::internal
{
	inline void copy_local_boundaries(
		const std::vector<mesh::LocalBoundary> &from,
		std::vector<mesh::LocalBoundary> &to)
	{
		to.clear();
		to.reserve(from.size());
		for (const auto &lb : from)
			to.emplace_back(lb);
	}

	inline void rebuild_node_positions(
		const std::vector<basis::ElementBases> &bases,
		const std::vector<int> &node_ids,
		std::vector<RowVectorNd> &positions)
	{
		positions.resize(node_ids.size());
		for (int n = 0; n < int(node_ids.size()); ++n)
		{
			const int node_id = node_ids[n];
			bool found = false;
			for (const auto &bs : bases)
			{
				for (const auto &b : bs.bases)
				{
					for (const auto &lg : b.global())
					{
						if (lg.index == node_id)
						{
							positions[n] = lg.node;
							found = true;
							break;
						}
					}

					if (found)
						break;
				}

				if (found)
					break;
			}
			assert(found);
		}
	}

	inline bool sample_scalar_field(
		const mesh::Mesh &mesh,
		const std::vector<basis::ElementBases> &field_bases,
		const std::vector<basis::ElementBases> &gbases,
		const io::OutputSample &sample,
		const Eigen::MatrixXd &dof_values,
		Eigen::MatrixXd &values)
	{
		if (dof_values.size() <= 0)
			return false;

		const bool has_element_samples =
			sample.local_points.rows() > 0
			&& sample.local_points.rows() == sample.element_ids.size();

		if (has_element_samples)
		{
			values.resize(sample.local_points.rows(), 1);
			for (int i = 0; i < sample.local_points.rows(); ++i)
			{
				const int element_id = sample.element_ids(i);
				if (element_id < 0)
				{
					values(i) = 0;
					continue;
				}

				Eigen::MatrixXd local_sol, local_grad;
				io::Evaluator::interpolate_at_local_vals(
					mesh, 1, field_bases, gbases,
					element_id, sample.local_points.row(i), dof_values, local_sol, local_grad);
				values(i) = local_sol(0);
			}
			return true;
		}

		if (sample.node_ids.size() > 0)
		{
			values.resize(sample.node_ids.size(), 1);
			for (int i = 0; i < sample.node_ids.size(); ++i)
			{
				const int node_id = sample.node_ids(i);
				if (node_id < 0 || node_id >= dof_values.rows())
					return false;
				values(i) = dof_values(node_id);
			}
			return true;
		}

		return false;
	}

	inline void expand_primary_matrix(
		const int full_size,
		const StiffnessMatrix &primary,
		StiffnessMatrix &expanded)
	{
		std::vector<Eigen::Triplet<double>> blocks;
		blocks.reserve(primary.nonZeros());
		for (int k = 0; k < primary.outerSize(); ++k)
		{
			for (StiffnessMatrix::InnerIterator it(primary, k); it; ++it)
				blocks.emplace_back(it.row(), it.col(), it.value());
		}

		expanded.resize(full_size, full_size);
		expanded.setFromTriplets(blocks.begin(), blocks.end());
		expanded.makeCompressed();
	}

	inline bool write_matrix_market(const json &args, const StiffnessMatrix &stiffness)
	{
		const std::string full_mat_path = args["output"]["data"]["full_mat"];
		if (full_mat_path.empty())
			return false;

		Eigen::saveMarket(stiffness, full_mat_path);
		return true;
	}
} // namespace polyfem::varform::internal
