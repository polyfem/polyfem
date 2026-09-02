#pragma once

#include <Eigen/Core>

#include <utility>
#include <vector>

namespace polyfem::mesh
{
	/// Format-independent input used to construct a runtime Mesh.
	class MeshData
	{
	public:
		MeshData(Eigen::MatrixXd vertices, Eigen::MatrixXi elements)
			: vertices(std::move(vertices)), elements(std::move(elements)) {}

		void validate() const;
		int dimension() const { return vertices.cols(); }
		bool has_polyhedral_topology() const { return !faces.empty(); }

		Eigen::MatrixXd vertices;
		Eigen::MatrixXi elements;

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
