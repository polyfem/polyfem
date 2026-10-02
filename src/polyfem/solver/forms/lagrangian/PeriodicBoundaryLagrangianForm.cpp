#include "PeriodicBoundaryLagrangianForm.hpp"

#include <polyfem/utils/Logger.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <vector>

namespace polyfem::solver
{
	namespace
	{
		struct BoundaryNode
		{
			RowVectorNd point;
			std::vector<std::pair<int, double>> weights;
		};

		using BoundaryNodes = std::vector<BoundaryNode>;

		bool same_weights(
			const std::vector<std::pair<int, double>> &lhs,
			const std::vector<std::pair<int, double>> &rhs)
		{
			if (lhs.size() != rhs.size())
				return false;
			for (int i = 0; i < int(lhs.size()); ++i)
			{
				if (lhs[i].first != rhs[i].first || std::abs(lhs[i].second - rhs[i].second) > 1e-12)
					return false;
			}
			return true;
		}

		BoundaryNodes collect_boundary_nodes(
			const int boundary_id,
			const mesh::Mesh &mesh,
			const std::vector<basis::ElementBases> &bases,
			const std::vector<mesh::LocalBoundary> &local_boundary)
		{
			BoundaryNodes result;
			for (const mesh::LocalBoundary &lb : local_boundary)
			{
				const basis::ElementBases &element_bases = bases.at(lb.element_id());
				for (int i = 0; i < lb.size(); ++i)
				{
					const int primitive_id = lb.global_primitive_id(i);
					if (mesh.get_boundary_id(primitive_id) != boundary_id)
						continue;

					const Eigen::VectorXi local_nodes = element_bases.local_nodes_for_primitive(primitive_id, mesh);
					for (const int local_node : local_nodes)
					{
						std::map<int, double> accumulated_weights;
						RowVectorNd point = RowVectorNd::Zero(mesh.dimension());
						for (const basis::Local2Global &global : element_bases.bases.at(local_node).global())
						{
							accumulated_weights[global.index] += global.val;
							point += global.val * global.node;
						}

						BoundaryNode node;
						node.point = point;
						for (const auto &[index, weight] : accumulated_weights)
						{
							if (std::abs(weight) > 1e-14)
								node.weights.emplace_back(index, weight);
						}
						if (node.weights.empty())
							log_and_throw_error("Unable to assemble a DoF on periodic boundary {}", boundary_id);

						const auto duplicate = std::find_if(
							result.begin(), result.end(),
							[&](const BoundaryNode &other) { return same_weights(node.weights, other.weights); });
						if (duplicate == result.end())
							result.emplace_back(std::move(node));
						else if ((duplicate->point - point).norm() > 1e-12)
							log_and_throw_error("Inconsistent position for a DoF on periodic boundary {}", boundary_id);
					}
				}
			}
			return result;
		}
	} // namespace

	PeriodicBoundaryLagrangianForm::PeriodicBoundaryLagrangianForm(
		const int ndof,
		const int value_dim,
		const mesh::Mesh &mesh,
		const std::vector<basis::ElementBases> &bases,
		const std::vector<mesh::LocalBoundary> &local_boundary,
		const std::array<int, 2> &boundary_ids,
		const double relative_tolerance)
		: PeriodicBoundaryLagrangianForm(build_mapping(
			  ndof, value_dim, mesh, bases, local_boundary, boundary_ids, relative_tolerance))
	{
	}

	PeriodicBoundaryLagrangianForm::PeriodicBoundaryLagrangianForm(Mapping mapping)
		: MatrixLagrangianForm(mapping.A, mapping.b)
	{
	}

	PeriodicBoundaryLagrangianForm::Mapping PeriodicBoundaryLagrangianForm::build_mapping(
		const int ndof,
		const int value_dim,
		const mesh::Mesh &mesh,
		const std::vector<basis::ElementBases> &bases,
		const std::vector<mesh::LocalBoundary> &local_boundary,
		const std::array<int, 2> &boundary_ids,
		const double relative_tolerance)
	{
		if (boundary_ids[0] < 0 || boundary_ids[1] < 0 || boundary_ids[0] == boundary_ids[1])
			log_and_throw_error("Periodic boundary IDs must be distinct non-negative integers");
		if (relative_tolerance <= 0)
			log_and_throw_error("Periodic boundary tolerance must be positive");

		const BoundaryNodes first = collect_boundary_nodes(boundary_ids[0], mesh, bases, local_boundary);
		const BoundaryNodes second = collect_boundary_nodes(boundary_ids[1], mesh, bases, local_boundary);
		if (first.empty() || second.empty())
			log_and_throw_error(
				"Unable to find DoFs for periodic boundary pair ({}, {}): found {} and {} DoFs",
				boundary_ids[0], boundary_ids[1], first.size(), second.size());
		if (first.size() != second.size())
			log_and_throw_error(
				"Periodic boundary pair ({}, {}) has different DoF counts: {} and {}",
				boundary_ids[0], boundary_ids[1], first.size(), second.size());

		RowVectorNd first_centroid = RowVectorNd::Zero(mesh.dimension());
		RowVectorNd second_centroid = RowVectorNd::Zero(mesh.dimension());
		for (const BoundaryNode &node : first)
			first_centroid += node.point;
		for (const BoundaryNode &node : second)
			second_centroid += node.point;
		first_centroid /= double(first.size());
		second_centroid /= double(second.size());

		// The paired boundaries are assumed to be translated copies. Use their
		// trace-DoF centroids to estimate the translation, then match the
		// translated DoF positions below. The tolerance is used only to find the
		// correspondence; the assembled periodic constraints are exact equalities.
		const RowVectorNd translation = second_centroid - first_centroid;

		RowVectorNd bbox_min, bbox_max;
		mesh.bounding_box(bbox_min, bbox_max);
		const double tolerance = relative_tolerance * (bbox_max - bbox_min).maxCoeff();

		std::vector<std::pair<int, int>> pairs;
		pairs.reserve(first.size());
		std::set<int> used_second;
		for (int first_index = 0; first_index < int(first.size()); ++first_index)
		{
			int matched_index = -1;
			double matched_distance = std::numeric_limits<double>::infinity();
			for (int second_index = 0; second_index < int(second.size()); ++second_index)
			{
				if (used_second.count(second_index) > 0)
					continue;

				const double distance = (first[first_index].point + translation - second[second_index].point).norm();
				if (distance < matched_distance)
				{
					matched_distance = distance;
					matched_index = second_index;
				}
			}

			if (matched_index < 0 || matched_distance > tolerance)
				log_and_throw_error(
					"No matching DoF found on periodic boundary {} for trace DoF {} on boundary {} (distance {}, tolerance {})",
					boundary_ids[1], first_index, boundary_ids[0], matched_distance, tolerance);
			if (!used_second.insert(matched_index).second)
				log_and_throw_error(
					"Periodic boundary pair ({}, {}) does not have a bijective DoF correspondence; DoF {} was matched more than once",
					boundary_ids[0], boundary_ids[1], matched_index);
			pairs.emplace_back(first_index, matched_index);
		}

		std::vector<Eigen::Triplet<double>> entries;
		for (int i = 0; i < int(pairs.size()); ++i)
		{
			for (int d = 0; d < value_dim; ++d)
			{
				const int row = i * value_dim + d;
				for (const auto &[index, weight] : first[pairs[i].first].weights)
				{
					const int dof = index * value_dim + d;
					if (dof < 0 || dof >= ndof)
						log_and_throw_error("Periodic boundary DoF index exceeds the problem size");
					entries.emplace_back(row, dof, weight);
				}
				for (const auto &[index, weight] : second[pairs[i].second].weights)
				{
					const int dof = index * value_dim + d;
					if (dof < 0 || dof >= ndof)
						log_and_throw_error("Periodic boundary DoF index exceeds the problem size");
					entries.emplace_back(row, dof, -weight);
				}
			}
		}

		Mapping data;
		data.translation = translation;
		std::set<int> boundary_dofs;
		for (const BoundaryNode &node : first)
			for (const auto &[index, weight] : node.weights)
				boundary_dofs.insert(index);
		for (const BoundaryNode &node : second)
			for (const auto &[index, weight] : node.weights)
				boundary_dofs.insert(index);
		data.boundary_dofs.assign(boundary_dofs.begin(), boundary_dofs.end());
		data.A.resize(int(pairs.size()) * value_dim, ndof);
		data.A.setFromTriplets(entries.begin(), entries.end());
		data.A.makeCompressed();
		data.b.setZero(data.A.rows(), 1);
		return data;
	}
} // namespace polyfem::solver
