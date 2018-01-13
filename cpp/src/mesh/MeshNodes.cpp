////////////////////////////////////////////////////////////////////////////////
#include "MeshNodes.hpp"
#include "Mesh2D.hpp"
#include "Mesh3D.hpp"
////////////////////////////////////////////////////////////////////////////////

using namespace poly_fem;

// -----------------------------------------------------------------------------

namespace {

	std::vector<bool> interface_edges(const Mesh2D &mesh) {
		std::vector<bool> is_interface(mesh.n_edges(), false);
		for (int f = 0; f < mesh.n_faces(); ++f) {
			if (mesh.is_cube(f)) { continue; } // Skip quads

			auto index = mesh.get_index_from_face(f, 0);
			for (int lv = 0; lv < mesh.n_face_vertices(f); ++lv) {
				auto index2 = mesh.switch_face(index);
				if (index2.face >= 0) {
					is_interface[index.edge] = true;
				}
				index = mesh.next_around_face(index);
			}
		}
		return is_interface;
	}

	std::vector<bool> interface_faces(const Mesh3D &mesh) {
		std::vector<bool> is_interface(mesh.n_faces(), false);
		for (int c = 0; c < mesh.n_cells(); ++c) {
			if (mesh.is_cube(c)) { continue; } // Skip hexes

			for (int lf = 0; lf < mesh.n_cell_faces(c); ++lf) {
				auto index = mesh.get_index_from_element(c, lf, 0);
				auto index2 = mesh.switch_element(index);
				if (index2.element >= 0) {
					is_interface[index.face] = true;
				}
			}
		}
		return is_interface;
	}


} // anonymous namespace

// -----------------------------------------------------------------------------

poly_fem::MeshNodes::MeshNodes(const Mesh &mesh, bool vertices_only)
	: edge_offset_(mesh.n_vertices())
	, face_offset_(edge_offset_ + (vertices_only ? 0 : mesh.n_edges()))
	, cell_offset_(face_offset_ + (vertices_only ? 0 : mesh.n_faces()))
{
	// Initialization
	int n_nodes = cell_offset_ + (vertices_only ? 0 : mesh.n_cells());
	primitive_to_node_.assign(n_nodes, -1); // #v + #e + #f + #c
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

	// Vertices only, no need to compute interface marker
	if (vertices_only) { return; }

	// Compute edges/faces that are at interface with a polytope
	if (mesh.is_volume()) {
		const Mesh3D * mesh3d = dynamic_cast<const Mesh3D *>(&mesh);
		assert(mesh3d);
		auto is_interface_face = interface_faces(*mesh3d);
		for (int f = 0; f < mesh.n_faces(); ++f) {
			is_interface_[face_offset_ + f] = is_interface_face[f];
		}
	} else {
		const Mesh2D * mesh2d = dynamic_cast<const Mesh2D *>(&mesh);
		assert(mesh2d);
		auto is_interface_edge = interface_edges(*mesh2d);
		for (int e = 0; e < mesh.n_edges(); ++e) {
			is_interface_[edge_offset_ + e] = is_interface_edge[e];
		}
	}

}

////////////////////////////////////////////////////////////////////////////////

int poly_fem::MeshNodes::node_id_from_primitive(int primitive_id) {
	if (primitive_to_node_[primitive_id] < 0) {
		primitive_to_node_[primitive_id] = n_nodes();
		node_to_primitive_.push_back(primitive_id);
	}
	return primitive_to_node_[primitive_id];
}

int poly_fem::MeshNodes::node_id_from_vertex(int v) {
	return node_id_from_primitive(v);
}

int poly_fem::MeshNodes::node_id_from_edge(int e) {
	return node_id_from_primitive(edge_offset_ + e);
}

int poly_fem::MeshNodes::node_id_from_face(int f) {
	return node_id_from_primitive(face_offset_ + f);
}

int poly_fem::MeshNodes::node_id_from_cell(int c) {
	return node_id_from_primitive(cell_offset_ + c);
}

////////////////////////////////////////////////////////////////////////////////

int poly_fem::MeshNodes::vertex_from_node_id(int node_id) const {
	const int res = node_to_primitive_[node_id];
	assert(res >= 0 && res < edge_offset_);
	return res;
}

int poly_fem::MeshNodes::edge_from_node_id(int node_id) const {
	const int res = node_to_primitive_[node_id];
	assert(res >= edge_offset_ && res < face_offset_);
	return res-edge_offset_;
}

int poly_fem::MeshNodes::face_from_node_id(int node_id) const {
	const int res = node_to_primitive_[node_id];
	assert(res >= face_offset_ && res < cell_offset_);
	return res-face_offset_;
}

int poly_fem::MeshNodes::cell_from_node_id(int node_id) const {
	const int res = node_to_primitive_[node_id];
	assert(res >= cell_offset_ && res < n_nodes());
	return res-cell_offset_;
}

////////////////////////////////////////////////////////////////////////////////

std::vector<int> poly_fem::MeshNodes::boundary_nodes() const {
	std::vector<int> res;
	res.reserve(n_nodes());
	for (size_t prim_id = 0; prim_id < primitive_to_node_.size(); ++prim_id) {
		int node_id = primitive_to_node_[prim_id];
		if (node_id >= 0 && is_boundary_[prim_id]) {
			res.push_back(node_id);
		}
	}
	return res;
}
