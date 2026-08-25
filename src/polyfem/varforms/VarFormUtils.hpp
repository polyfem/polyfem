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

	inline bool sample_scalar_field(
		const mesh::Mesh &mesh,
		const std::vector<basis::ElementBases> &field_bases,
		const std::vector<basis::ElementBases> &gbases,
		const io::OutputSample &sample,
		const Eigen::MatrixXd &dof_values,
		Eigen::MatrixXd &values,
		Eigen::MatrixXd *gradients = nullptr)
	{
		if (dof_values.size() <= 0)
			return false;

		const bool has_element_samples =
			sample.local_points.rows() > 0
			&& sample.local_points.rows() == sample.element_ids.size();

		if (has_element_samples)
		{
			values.resize(sample.local_points.rows(), 1);
			if (gradients)
				gradients->resize(sample.local_points.rows(), mesh.dimension());
			for (int i = 0; i < sample.local_points.rows(); ++i)
			{
				const int element_id = sample.element_ids(i);
				if (element_id < 0)
				{
					values(i) = 0;
					if (gradients)
						gradients->row(i).setZero();
					continue;
				}

				Eigen::MatrixXd local_sol, local_grad;
				io::Evaluator::interpolate_at_local_vals(
					mesh, 1, field_bases, gbases,
					element_id, sample.local_points.row(i), dof_values, local_sol, local_grad);
				values(i) = local_sol(0);
				if (gradients)
					gradients->row(i) = local_grad;
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
