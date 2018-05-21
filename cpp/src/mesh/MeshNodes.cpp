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

poly_fem::MeshNodes::MeshNodes(const Mesh &mesh, const int max_nodes_per_edge, const int max_nodes_per_face, const int max_nodes_per_cell)
: mesh_(mesh)
, edge_offset_(mesh.n_vertices())
, face_offset_(edge_offset_ + mesh.n_edges() * max_nodes_per_edge)
, cell_offset_(face_offset_ + mesh.n_faces() * max_nodes_per_face)

, max_nodes_per_edge_(max_nodes_per_edge)
, max_nodes_per_face_(max_nodes_per_face)
, max_nodes_per_cell_(max_nodes_per_cell)
{
	// Initialization
	int n_nodes = cell_offset_ + mesh.n_cells() * max_nodes_per_cell;
	primitive_to_node_.assign(n_nodes, -1); // #v + #e + #f + #c
	nodes_.resize(n_nodes, mesh.dimension());
	is_boundary_.assign(n_nodes, false);
	is_interface_.assign(n_nodes, false);

	// Vertex nodes
	for (int v = 0; v < mesh.n_vertices(); ++v) {
		// nodes_.row(v) = mesh.point(v);
		is_boundary_[v] = mesh.is_boundary_vertex(v);
	}
	// if (!vertices_only) {
		// Eigen::MatrixXd bary;
		// Edge nodes
		// mesh.edge_barycenters(bary);

	for (int e = 0; e < mesh.n_edges(); ++e) {
		// nodes_.row(edge_offset_ + e) = bary.row(e);
		const bool is_boundary = mesh.is_boundary_edge(e);
		for(int tmp = 0; tmp < max_nodes_per_edge; ++tmp)
			is_boundary_[edge_offset_ + max_nodes_per_edge * e + tmp] = is_boundary;
	}
		// Face nodes
		// mesh.face_barycenters(bary);
	for (int f = 0; f < mesh.n_faces(); ++f) {
		// nodes_.row(face_offset_ + f) = bary.row(f);
		for(int tmp = 0; tmp < max_nodes_per_face; ++tmp)
		{
			if (mesh.is_volume()) {
				is_boundary_[face_offset_ + max_nodes_per_face * f + tmp] = mesh.is_boundary_face(f);
			} else {
				is_boundary_[face_offset_ + max_nodes_per_face * f + tmp] = false;
			}
		}
	}
		// Cell nodes
		// mesh.cell_barycenters(bary);
	for (int c = 0; c < mesh.n_cells(); ++c) {
		for(int tmp = 0; tmp < max_nodes_per_cell; ++tmp)
			is_boundary_[cell_offset_ + max_nodes_per_cell * c + tmp] = false;
	}
	// }

	// Vertices only, no need to compute interface marker
	// if (vertices_only) { return; }

	// Compute edges/faces that are at interface with a polytope
	if (mesh.is_volume()) {
		const Mesh3D * mesh3d = dynamic_cast<const Mesh3D *>(&mesh);
		assert(mesh3d);
		auto is_interface_face = interface_faces(*mesh3d);
		for (int f = 0; f < mesh.n_faces(); ++f) {
			for(int tmp = 0; tmp < max_nodes_per_face; ++tmp)
				is_interface_[face_offset_ + max_nodes_per_face * f + tmp] = is_interface_face[f];
		}
	} else {
		const Mesh2D * mesh2d = dynamic_cast<const Mesh2D *>(&mesh);
		assert(mesh2d);
		auto is_interface_edge = interface_edges(*mesh2d);
		for (int e = 0; e < mesh.n_edges(); ++e) {
			for(int tmp = 0; tmp < max_nodes_per_edge; ++tmp)
				is_interface_[edge_offset_ + max_nodes_per_edge * e + tmp] = is_interface_edge[e];
		}
	}

}

////////////////////////////////////////////////////////////////////////////////

int poly_fem::MeshNodes::node_id_from_primitive(int primitive_id) {
	if (primitive_to_node_[primitive_id] < 0) {
		primitive_to_node_[primitive_id] = n_nodes();
		node_to_primitive_.push_back(primitive_id);

		if(primitive_id < edge_offset_)
			nodes_.row(primitive_id) = mesh_.point(primitive_id);
		else if(primitive_id < face_offset_)
			nodes_.row(primitive_id) = mesh_.edge_barycenter(primitive_id - edge_offset_);
		else if(primitive_id < cell_offset_)
			nodes_.row(primitive_id) = mesh_.face_barycenter(primitive_id - face_offset_);
		else
			nodes_.row(primitive_id) = mesh_.cell_barycenter(primitive_id - cell_offset_);
	}
	return primitive_to_node_[primitive_id];
}

std::vector<int> poly_fem::MeshNodes::node_ids_from_edge(const Navigation::Index &index, const int n_new_nodes)
{
	std::vector<int> res;
	if(n_new_nodes <= 0)
		return res;

	const int start = edge_offset_ + index.edge * max_nodes_per_edge_;

	const Mesh2D * mesh2d = dynamic_cast<const Mesh2D *>(&mesh_);

	const auto v1 = mesh2d->point(index.vertex);
	const auto v2 = mesh2d->point(mesh2d->switch_vertex(index).vertex);

	const int start_node_id = primitive_to_node_[start];
	if (start_node_id < 0) {
		for(int i = 1; i <= n_new_nodes; ++i)
		{
			const double t = i/(n_new_nodes + 1.0);

			const int primitive_id = start + i - 1;
			primitive_to_node_[primitive_id] = n_nodes();
			node_to_primitive_.push_back(primitive_id);

			nodes_.row(primitive_id) = (1 - t) * v1 + t * v2;

			res.push_back(primitive_to_node_[primitive_id]);
		}
	}
	else
	{
		const double t = 1/(n_new_nodes + 1.0);
		const RowVectorNd v = (1 - t) * v1 + t * v2;
		if((node_position(start_node_id) - v).squaredNorm() < 1e-8)
		{
			for(int i = 0; i < n_new_nodes; ++i)
			{
				const int primitive_id = start + i;
				res.push_back(primitive_to_node_[primitive_id]);
			}
		}
		else
		{
			for(int i = n_new_nodes - 1; i >= 0; --i)
			{
				const int primitive_id = start + i;
				res.push_back(primitive_to_node_[primitive_id]);
			}
		}
	}

	assert(res.size() == n_new_nodes);
	return res;
}

std::vector<int> poly_fem::MeshNodes::node_ids_from_edge(const Navigation3D::Index &index, const int n_new_nodes)
{
	std::vector<int> res;
	if(n_new_nodes <= 0)
		return res;

	const int start = edge_offset_ + index.edge * max_nodes_per_edge_;

	const Mesh3D * mesh3d = dynamic_cast<const Mesh3D *>(&mesh_);

	const auto v1 = mesh3d->point(index.vertex);
	const auto v2 = mesh3d->point(mesh3d->switch_vertex(index).vertex);

	const int start_node_id = primitive_to_node_[start];
	if (start_node_id < 0) {
		for(int i = 1; i <= n_new_nodes; ++i)
		{
			const double t = i/(n_new_nodes + 1.0);

			const int primitive_id = start + i - 1;
			primitive_to_node_[primitive_id] = n_nodes();
			node_to_primitive_.push_back(primitive_id);

			nodes_.row(primitive_id) = (1 - t) * v1 + t * v2;

			res.push_back(primitive_to_node_[primitive_id]);
		}
	}
	else
	{
		const double t = 1/(n_new_nodes + 1.0);
		const RowVectorNd v = (1 - t) * v1 + t * v2;
		if((node_position(start_node_id) - v).squaredNorm() < 1e-8)
		{
			for(int i = 0; i < n_new_nodes; ++i)
			{
				const int primitive_id = start + i;
				res.push_back(primitive_to_node_[primitive_id]);
			}
		}
		else
		{
			for(int i = n_new_nodes - 1; i >= 0; --i)
			{
				const int primitive_id = start + i;
				res.push_back(primitive_to_node_[primitive_id]);
			}
		}
	}

	assert(res.size() == n_new_nodes);
	return res;
}

std::vector<int> poly_fem::MeshNodes::node_ids_from_face(const Navigation::Index &index, const int n_new_nodes)
{
	std::vector<int> res;
	if(n_new_nodes <= 0)
		return res;

	assert(mesh_.is_simplex(index.face));
	const int start = face_offset_ + index.face * max_nodes_per_face_;
	const int start_node_id = primitive_to_node_[start];

	const Mesh2D * mesh2d = dynamic_cast<const Mesh2D *>(&mesh_);

	const auto v1 = mesh2d->point(index.vertex);
	const auto v2 = mesh2d->point(mesh2d->switch_vertex(index).vertex);
	const auto v3 = mesh2d->point(mesh2d->switch_vertex(mesh2d->switch_edge(index)).vertex);

	if(start_node_id < 0)
	{
		int loc_index = 0;
		for(int i = 1; i <= n_new_nodes; ++i)
		{
			const double b2 = i/(n_new_nodes + 2.0);
			for(int j = 1; j <= n_new_nodes - i + 1; ++j)
			{
				const double b3 = j/(n_new_nodes + 2.0);
				const double b1 = 1 - b3 - b2;
				assert(b3 < 1);
				assert(b3 > 0);

				const int primitive_id = start + loc_index;
				primitive_to_node_[primitive_id] = n_nodes();
				node_to_primitive_.push_back(primitive_id);

				nodes_.row(primitive_id) = b1 * v1 + b2 * v2 + b3 * v3;

				res.push_back(primitive_to_node_[primitive_id]);

				++loc_index;
			}
		}
	}
	else
	{
		assert(false);
	}
	assert(res.size() == n_new_nodes *(n_new_nodes+1) / 2);
	return res;
}


std::vector<int> poly_fem::MeshNodes::node_ids_from_face(const Navigation3D::Index &index, const int n_new_nodes)
{
	std::vector<int> res;
	if(n_new_nodes <= 0)
		return res;

	assert(mesh_.is_simplex(index.element));
	const int start = face_offset_ + index.face * max_nodes_per_face_;
	const int start_node_id = primitive_to_node_[start];

	const Mesh3D * mesh3d = dynamic_cast<const Mesh3D *>(&mesh_);

	const auto v1 = mesh3d->point(index.vertex);
	const auto v2 = mesh3d->point(mesh3d->switch_vertex(index).vertex);
	const auto v3 = mesh3d->point(mesh3d->switch_vertex(mesh3d->switch_edge(index)).vertex);

	if(start_node_id < 0)
	{
		int loc_index = 0;
		for(int i = 1; i <= n_new_nodes; ++i)
		{
			const double b2 = i/(n_new_nodes + 2.0);
			for(int j = 1; j <= n_new_nodes - i + 1; ++j)
			{
				const double b3 = j/(n_new_nodes + 2.0);
				const double b1 = 1 - b3 - b2;
				assert(b3 < 1);
				assert(b3 > 0);

				const int primitive_id = start + loc_index;
				primitive_to_node_[primitive_id] = n_nodes();
				node_to_primitive_.push_back(primitive_id);

				nodes_.row(primitive_id) = b1 * v1 + b2 * v2 + b3 * v3;

				res.push_back(primitive_to_node_[primitive_id]);

				++loc_index;
			}
		}
	}
	else
	{
		//TODO
		assert(false);
	}
	assert(res.size() == n_new_nodes *(n_new_nodes+1) / 2);
	return res;
}


std::vector<int> poly_fem::MeshNodes::node_ids_from_cell(const Navigation3D::Index &index, const int n_new_nodes)
{
	assert(n_new_nodes == 0); //P4 only
	const int idx = node_id_from_cell(index.element);

	return {idx};
}

int poly_fem::MeshNodes::node_id_from_vertex(int v) {
	return node_id_from_primitive(v);
}

int poly_fem::MeshNodes::node_id_from_edge(int e) {
	return node_id_from_primitive(edge_offset_ + e * max_nodes_per_edge_);
}

int poly_fem::MeshNodes::node_id_from_face(int f) {
	return node_id_from_primitive(face_offset_ + f * max_nodes_per_face_);
}

int poly_fem::MeshNodes::node_id_from_cell(int c) {
	return node_id_from_primitive(cell_offset_ + c * max_nodes_per_cell_);
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
