////////////////////////////////////////////////////////////////////////////////
#include "MeshNodes.hpp"
////////////////////////////////////////////////////////////////////////////////

poly_fem::MeshNodes::MeshNodes(const Mesh &mesh, bool vertices_only)
	: edge_offset_(mesh.n_vertices())
	, face_offset_(edge_offset_ + (vertices_only ? 0 : mesh.n_edges()))
	, cell_offset_(face_offset_ + (vertices_only ? 0 : mesh.n_faces()))
{
	// Initialization
	int n_nodes = cell_offset_ + (vertices_only ? 0 : mesh.n_cells());
	simplex_to_node_.assign(n_nodes, -1); // #v + #e + #f + #c
	nodes_.resize(n_nodes, mesh.is_volume() ? 3 : 2);
	is_boundary_.assign(n_nodes, false);
	is_interface_.assign(n_nodes, false);


	// Vertex nodes
	for (int v = 0; v < mesh.n_vertices(); ++v) {
		nodes_.row(v) = mesh.point(v);
		is_boundary_[v] = mesh.is_boundary_vertex(v);
	}
	if (!vertices_only) {
		Eigen::MatrixXd bary;
		// Edge nodes
		mesh.edge_barycenters(bary);
		for (int e = 0; e < mesh.n_edges(); ++e) {
			nodes_.row(edge_offset_ + e) = bary.row(e);
			is_boundary_[edge_offset_ + e] = mesh.is_boundary_edge(e);
		}
		// Face nodes
		mesh.face_barycenters(bary);
		for (int f = 0; f < mesh.n_faces(); ++f) {
			nodes_.row(face_offset_ + f) = bary.row(f);
			if (mesh.is_volume()) {
				is_boundary_[face_offset_ + f] = mesh.is_boundary_face(f);
			} else {
				is_boundary_[face_offset_ + f] = false;
			}
		}
		// Cell nodes
		mesh.cell_barycenters(bary);
		for (int c = 0; c < mesh.n_cells(); ++c) {
			nodes_.row(cell_offset_ + c) = bary.row(c);
			is_boundary_[cell_offset_ + c] = false;
		}
	}
}

////////////////////////////////////////////////////////////////////////////////

int poly_fem::MeshNodes::node_id_from_simplex(int simplex_id) {
	if (simplex_to_node_[simplex_id] < 0) {
		simplex_to_node_[simplex_id] = n_nodes();
		node_to_simplex_.push_back(simplex_id);
	}
	return simplex_to_node_[simplex_id];
}

// -----------------------------------------------------------------------------
int poly_fem::MeshNodes::node_id_from_vertex(int v) {
	return node_id_from_simplex(v);
}

int poly_fem::MeshNodes::node_id_from_edge(int e) {
	return node_id_from_simplex(edge_offset_ + e);
}

int poly_fem::MeshNodes::node_id_from_face(int f) {
	return node_id_from_simplex(face_offset_ + f);
}

int poly_fem::MeshNodes::node_id_from_cell(int c) {
	return node_id_from_simplex(cell_offset_ + c);
}

////////////////////////////////////////////////////////////////////////////////

std::vector<int> poly_fem::MeshNodes::boundary_nodes() const {
	std::vector<int> res;
	res.reserve(n_nodes());
	for (int node_id : simplex_to_node_) {
		if (node_id >= 0) {
			res.push_back(node_id);
		}
	}
	return res;
}
