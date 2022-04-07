#pragma once

#include <polyfem/Mesh.hpp>
#include <polyfem/Types.hpp>

#include <polyfem/Navigation.hpp>
#include <polyfem/Navigation3D.hpp>

#include <Eigen/Dense>

namespace polyfem
{

	// Wrapper for lazy assignment of node ids
	class MeshNodes
	{
	public:
		MeshNodes(const Mesh &mesh, const bool has_poly, const bool connect_nodes, const int max_nodes_per_edge, const int max_nodes_per_face, const int max_nodes_per_cell = 0);

		// Number of currently assigned nodes
		int n_nodes() const { return node_to_primitive_.size(); }

		// Lazy retrieval of node ids
		int node_id_from_vertex(int v);
		int node_id_from_edge(int e);
		int node_id_from_face(int f);
		int node_id_from_cell(int c);
		int node_id_from_primitive(int primitive_id);

		std::vector<int> node_ids_from_edge(const Navigation::Index &index, const int n_new_nodes);
		std::vector<int> node_ids_from_face(const Navigation::Index &index, const int n_new_nodes);

		std::vector<int> node_ids_from_edge(const Navigation3D::Index &index, const int n_new_nodes);
		std::vector<int> node_ids_from_face(const Navigation3D::Index &index, const int n_new_nodes);
		std::vector<int> node_ids_from_cell(const Navigation3D::Index &index, const int n_new_nodes);

		// Packed id from primitive
		int primitive_from_vertex(int v) const { return v; }
		int primitive_from_edge(int e) const { return edge_offset_ + e; }
		int primitive_from_face(int f) const { return face_offset_ + f; }
		int primitive_from_cell(int c) const { return cell_offset_ + c; }

		// Primitive id from node id
		int vertex_from_node_id(int node_id) const;
		int edge_from_node_id(int node_id) const;
		int face_from_node_id(int node_id) const;
		int cell_from_node_id(int node_id) const;

		const std::vector<int> &primitive_to_node() const { return primitive_to_node_; }

		// Node position from node id
		RowVectorNd node_position(int node_id) const { return nodes_.row(node_to_primitive_[node_id]); }

		// Whether a node is on the mesh boundary or not
		bool is_boundary(int node_id) const { return is_boundary_[node_to_primitive_[node_id]]; }
		bool is_primitive_boundary(int primitive) const { return is_boundary_[primitive]; }

		// Whether an edge node (in 2D) or face node (in 3D) is at the interface with a polytope
		bool is_interface(int node_id) const { return is_interface_[node_to_primitive_[node_id]]; }
		bool is_primitive_interface(int primitive) const { return is_interface_[primitive]; }

		// Either boundary or interface
		bool is_boundary_or_interface(const int node_id) const { return is_boundary(node_id) || is_interface(node_id); }

		// Retrieve a list of nodes which are marked as boundary
		std::vector<int> boundary_nodes() const;

	private:
		const Mesh &mesh_;
		const bool connect_nodes_;
		// Offset to pack primitives ids into a single vector
		const int edge_offset_;
		const int face_offset_;
		const int cell_offset_;

		const int max_nodes_per_edge_;
		const int max_nodes_per_face_;
		const int max_nodes_per_cell_;

		// Map primitives to nodes back and forth
		std::vector<int> primitive_to_node_; // #v + #e + #f + #c
		std::vector<int> node_to_primitive_; // #assigned nodes

		// Precomputed node data (#v + #e + #f + #c)
		Eigen::MatrixXd nodes_;
		std::vector<bool> is_boundary_;
		std::vector<bool> is_interface_;
	};

} // namespace polyfem
