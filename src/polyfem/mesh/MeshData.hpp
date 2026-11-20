#pragma once

#include <Eigen/Core>

#include <optional>
#include <utility>
#include <vector>

namespace polyfem::mesh
{
	/// Primitive NC state. Indices refer to full storage, including inactive entities.
	struct NCMeshData
	{
		Eigen::MatrixXd vertices;
		Eigen::MatrixXi cells, ordered_cells, cell_edges, cell_faces, children;
		/// Columns: level, parent, refined, ghost, body ID, geometry ID.
		Eigen::MatrixXi element_state;
		/// Connectivity followed by the boundary label.
		Eigen::MatrixXi edges, faces;
		/// Sorted edge endpoints followed by their midpoint vertex.
		Eigen::MatrixXi midpoints;
		std::vector<int> node_ids, refinement_history;
		int label_flags = 0; // body=1, geometry=2, node=4, boundary=8
		void validate(const class MeshData &active) const;
		/// Merge full storage, remapping all references into this payload.
		void append(const NCMeshData &other);
	};

	/// Format-independent input used to construct a runtime Mesh.
	class MeshData
	{
	public:
		MeshData(Eigen::MatrixXd vertices, Eigen::MatrixXi elements)
			: vertices(std::move(vertices)), elements(std::move(elements)) {}

		void validate() const;
		int dimension() const { return vertices.cols(); }
		bool has_polyhedral_topology() const { return !faces.empty(); }

		std::optional<NCMeshData> nc;
		Eigen::MatrixXd vertices;
		Eigen::MatrixXi elements;
		/// Whether each element row carries a format-defined local vertex order.
		/// Arbitrary HYBRID polyhedra only provide an unordered vertex set.
		bool elements_are_ordered = true;

		std::vector<int> body_ids;
		std::vector<int> geometry_ids;
		std::vector<int> node_ids;
		std::vector<std::vector<int>> boundary_elements;
		std::vector<int> boundary_ids;

		Eigen::MatrixXd higher_order_nodes;
		std::vector<std::vector<int>> higher_order_connectivity;
		std::vector<std::vector<double>> higher_order_weights;

		/// Optional topology for arbitrary polyhedral cells.
		std::vector<std::vector<int>> faces;
		std::vector<std::vector<int>> cell_faces;
		std::vector<std::vector<int>> cell_face_orientations;
		std::vector<bool> cell_is_hex;
		Eigen::MatrixXd cell_kernel_points;
	};
} // namespace polyfem::mesh
