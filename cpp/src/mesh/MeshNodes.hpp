#pragma once

#include "Mesh.hpp"
#include "Types.hpp"
#include <Eigen/Dense>

namespace poly_fem {

// Wrapper for lazy assignment of node ids
class MeshNodes {
public:
	MeshNodes(const Mesh &mesh, bool vertices_only = false);

	// Number of currently assigned nodes
	int n_nodes() const { return node_to_primitive_.size(); }

	// Lazy retrieval of node ids
	int node_id_from_vertex(int v);
	int node_id_from_edge(int e);
	int node_id_from_face(int f);
	int node_id_from_cell(int c);
	int node_id_from_primitive(int primitive_id);

	int primitive_from_vertex(int v) const { return v; }
	int primitive_from_edge(int e) const { return edge_offset_ + e; }
	int primitive_from_face(int f) const { return face_offset_ + f; }
	int primitive_from_cell(int c) const { return cell_offset_ + c; }

	int vertex_from_node_id(int node_id) const
	{
		const int res = node_to_primitive_[node_id];
		assert(res >= 0 && res < edge_offset_);
		return res;
	}
	int edge_from_node_id(int node_id) const {
		const int res = node_to_primitive_[node_id];
		assert(res >= edge_offset_ && res < face_offset_);

		return res-edge_offset_;
	}
	int face_from_node_id(int node_id) const {
		const int res = node_to_primitive_[node_id];
		assert(res >= face_offset_ && res < cell_offset_);

		return res-face_offset_;
	}
	int cell_from_node_id(int node_id) const {
		const int res = node_to_primitive_[node_id];
		assert(res >= cell_offset_ && res < n_nodes());

		return res-cell_offset_;
	}

	// Node position from node id
	RowVectorNd node_position(int node_id) const { return nodes_.row(node_to_primitive_[node_id]); }

	// Whether a node is on the mesh boundary or not
	bool is_boundary(int node_id) const { return is_boundary_[node_to_primitive_[node_id]]; }

	// Whether an edge node (in 2D) or face node (in 3D) is at the interface with a polytope
	bool is_interface(int node_id) const { return is_interface_[node_to_primitive_[node_id]]; }
	bool is_primitive_interface(int primitive) const { return is_interface_[primitive]; }



	// Either boundary or interface
	bool is_boundary_or_interface(const int node_id) const { return is_boundary(node_id) || is_interface(node_id); }

	// Retrieve a list of nodes which are marked as boundary
	std::vector<int> boundary_nodes() const;

private:
	// Offset to pack primitives ids into a single vector
	const int edge_offset_;
	const int face_offset_;
	const int cell_offset_;

	// Map primitives to nodes back and forth
	std::vector<int> primitive_to_node_; // #v + #e + #f + #c
	std::vector<int> node_to_primitive_; // #assigned nodes

	// Precomputed node data (#v + #e + #f + #c)
	Eigen::MatrixXd nodes_;
	std::vector<bool> is_boundary_;
	std::vector<bool> is_interface_;
};

} // namespace poly_fem
