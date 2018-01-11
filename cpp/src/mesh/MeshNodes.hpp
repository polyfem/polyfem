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
	int n_nodes() const { return node_to_simplex_.size(); }

	// Lazy retrieval of node ids
	int node_id_from_vertex(int v);
	int node_id_from_edge(int e);
	int node_id_from_face(int f);
	int node_id_from_cell(int c);

	// Node position from node id
	RowVectorNd node_position(int node_id) const { return nodes_.row(node_id); }

	// Whether a node is on the mesh boundary or not
	bool is_boundary(int node_id) const { return is_boundary_[(size_t) node_id]; }

	// Whether an edge node (in 2D) or face node (in 3D) is at the interface with a polytope
	bool is_interface(int node_id) const { return is_interface_[(size_t) node_id]; }

	// Retrieve a list of nodes which are marked as boundary
	std::vector<int> boundary_nodes() const;

private:
	// Lazy assignment of a node id from the offset simplex id (vertex, edge, face or cell)
	int node_id_from_simplex(int packed_simplex_id);

private:
	// Offset to pack simplices ids into a single vector
	int edge_offset_;
	int face_offset_;
	int cell_offset_;

	// Map simplices to nodes back and forth
	std::vector<int> simplex_to_node_; // #v + #e + #f + #c
	std::vector<int> node_to_simplex_; // #assigned nodes

	// Precomputed node data (#v + #e + #f + #c)
	Eigen::MatrixXd nodes_;
	std::vector<bool> is_boundary_;
	std::vector<bool> is_interface_;
};

} // namespace poly_fem
