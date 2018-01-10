////////////////////////////////////////////////////////////////////////////////
#include "FEBasis3d.hpp"
#include "HexQuadrature.hpp"
#include <igl/viewer/Viewer.h>
#include <cassert>
#include <array>
////////////////////////////////////////////////////////////////////////////////

/*

Axes:
  z y
  │╱
  o──x

Boundaries:
X axis: left/right
Y axis: front/back
Z axis: bottom/top

Corner nodes:
      v7──────x─────v6
      ╱┆     ╱      ╱│
     ╱ ┆    ╱      ╱ │
    x┄┄┼┄┄┄x┄┄┄┄┄┄x  │
   ╱   x  ╱      ╱┆  x
  ╱    ┆ ╱      ╱ ┆ ╱│
v4─────┼x─────v5  ┆╱ │
 │     ┆┆      │  x  │
 │    v3┼┄┄┄┄┄x┼┄⌿┼┄v2
 │    ╱ ┆      │╱ ┆ ╱
 x┄┄┄⌿┄┄x┄┄┄┄┄┄x  ┆╱
 │  x   ┆      │  x
 │ ╱    ┆      │ ╱
 │╱     ┆      │╱
v0──────x─────v1

v0 = (0, 0, 0)
v1 = (1, 0, 0)
v2 = (1, 1, 0)
v3 = (0, 1, 0)
v4 = (0, 0, 1)
v5 = (1, 0, 1)
v6 = (1, 1, 1)
v7 = (0, 1, 1)

Edge nodes:
       x─────e10─────x
      ╱┆     ╱      ╱│
     ╱ ┆    ╱      ╱ │
   e11┄┼┄┄┄x┄┄┄┄┄e9  │
   ╱  e7  ╱      ╱┆ e6
  ╱    ┆ ╱      ╱ ┆ ╱│
 x─────e8──────x  ┆╱ │
 │     ┆┆      │  x  │
 │     x┼┄┄┄┄┄e2┄⌿┼┄┄x
 │    ╱ ┆      │╱ ┆ ╱
e4┄┄┄⌿┄┄x┄┄┄┄┄e5  ┆╱
 │ e3   ┆      │ e1
 │ ╱    ┆      │ ╱
 │╱     ┆      │╱
 x─────e0──────x

e0  = (0.5,   0,   0)
e1  = (  1, 0.5,   0)
e2  = (0.5,   1,   0)
e3  = (  0, 0.5,   0)
e4  = (  0,   0, 0.5)
e5  = (  1,   0, 0.5)
e6  = (  1,   1, 0.5)
e7  = (  0,   1, 0.5)
e8  = (0.5,   0,   1)
e9  = (  1, 0.5,   1)
e10 = (0.5,   1,   1)
e11 = (  0, 0.5,   1)

Face nodes:
      v7──────x─────v6
      ╱┆     ╱      ╱│
     ╱ ┆    ╱      ╱ │
    x┄┄┼┄┄f5┄┄┄┄┄┄x  │
   ╱   x  ╱  f3  ╱┆  x
  ╱    ┆ ╱      ╱ ┆ ╱│
v4─────┼x─────v5  ┆╱ │
 │ f0  ┆┆      │ f1  │
 │    v3┼┄┄┄┄┄x┼┄⌿┼┄v2
 │    ╱ ┆      │╱ ┆ ╱
 x┄┄┄⌿┄f2┄┄┄┄┄┄x  ┆╱
 │  x   ┆ f4   │  x
 │ ╱    ┆      │ ╱
 │╱     ┆      │╱
v0──────x─────v1

f0  = (  0, 0.5, 0.5)
f1  = (  1, 0.5, 0.5)
f2  = (0.5,   0, 0.5)
f3  = (0.5,   1, 0.5)
f4  = (0.5, 0.5,   0)
f5  = (0.5, 0.5,   1)

*/

namespace {

template<typename T>
Eigen::MatrixXd alpha(int i, T &t) {
	switch (i) {
		case 0: return (1-t);
		case 1: return t;
		default: assert(false);
	}
	throw std::runtime_error("Invalid index");
}

template<typename T>
Eigen::MatrixXd dalpha(int i, T &t) {
	switch (i) {
		case 0: return -1+0*t;
		case 1: return 1+0*t;;
		default: assert(false);
	}
	throw std::runtime_error("Invalid index");
}

template<typename T>
Eigen::MatrixXd theta(int i, T &t) {
	switch (i) {
		case 0: return (1 - t) * (1 - 2 * t);
		case 1: return 4 * t * (1 - t);
		case 2: return t * (2 * t - 1);
		default: assert(false);
	}
	throw std::runtime_error("Invalid index");
}

template<typename T>
Eigen::MatrixXd dtheta(int i, T &t) {
	switch (i) {
		case 0: return -3+4*t;
		case 1: return 4-8*t;
		case 2: return -1+4*t;
		default: assert(false);
	}
	throw std::runtime_error("Invalid index");
}

// -----------------------------------------------------------------------------

constexpr std::array<std::array<int, 3>, 8> linear_hex_local_node = {{
	{{0, 0, 0}}, // v0  = (0, 0, 0)
	{{1, 0, 0}}, // v1  = (1, 0, 0)
	{{1, 1, 0}}, // v2  = (1, 1, 0)
	{{0, 1, 0}}, // v3  = (0, 1, 0)
	{{0, 0, 1}}, // v4  = (0, 0, 1)
	{{1, 0, 1}}, // v5  = (1, 0, 1)
	{{1, 1, 1}}, // v6  = (1, 1, 1)
	{{0, 1, 1}}, // v7  = (0, 1, 1)
}};

void linear_hex_basis_value(const int local_index, const Eigen::MatrixXd &xne, Eigen::MatrixXd &val) {
	auto x=xne.col(0).array();
	auto n=xne.col(1).array();
	auto e=xne.col(2).array();

	std::array<int, 3> idx = linear_hex_local_node[local_index];
	val = alpha(idx[0], x).array() * alpha(idx[1], n).array() * alpha(idx[2], e).array();
}

void linear_hex_basis_grad(const int local_index, const Eigen::MatrixXd &xne, Eigen::MatrixXd &val) {
	auto x=xne.col(0).array();
	auto n=xne.col(1).array();
	auto e=xne.col(2).array();

	std::array<int, 3> idx = linear_hex_local_node[local_index];

	val.resize(xne.rows(), 3);
	val.col(0) = dalpha(idx[0], x).array() * alpha(idx[1], n).array() * alpha(idx[2], e).array();
	val.col(1) = alpha(idx[0], x).array() * dalpha(idx[1], n).array() * alpha(idx[2], e).array();
	val.col(2) = alpha(idx[0], x).array() * alpha(idx[1], n).array() * dalpha(idx[2], e).array();
}

// -----------------------------------------------------------------------------

constexpr std::array<std::array<int, 3>, 27> quadr_hex_local_node = {{
	{{0, 0, 0}}, // v0  = (  0,   0,   0)
	{{2, 0, 0}}, // v1  = (  1,   0,   0)
	{{2, 2, 0}}, // v2  = (  1,   1,   0)
	{{0, 2, 0}}, // v3  = (  0,   1,   0)
	{{0, 0, 2}}, // v4  = (  0,   0,   1)
	{{2, 0, 2}}, // v5  = (  1,   0,   1)
	{{2, 2, 2}}, // v6  = (  1,   1,   1)
	{{0, 2, 2}}, // v7  = (  0,   1,   1)
	{{1, 0, 0}}, // e0  = (0.5,   0,   0)
	{{2, 1, 0}}, // e1  = (  1, 0.5,   0)
	{{1, 2, 0}}, // e2  = (0.5,   1,   0)
	{{0, 1, 0}}, // e3  = (  0, 0.5,   0)
	{{0, 0, 1}}, // e4  = (  0,   0, 0.5)
	{{2, 0, 1}}, // e5  = (  1,   0, 0.5)
	{{2, 2, 1}}, // e6  = (  1,   1, 0.5)
	{{0, 2, 1}}, // e7  = (  0,   1, 0.5)
	{{1, 0, 2}}, // e8  = (0.5,   0,   1)
	{{2, 1, 2}}, // e9  = (  1, 0.5,   1)
	{{1, 2, 2}}, // e10 = (0.5,   1,   1)
	{{0, 1, 2}}, // e11 = (  0, 0.5,   1)
	{{0, 1, 1}}, // f0  = (  0, 0.5, 0.5)
	{{2, 1, 1}}, // f1  = (  1, 0.5, 0.5)
	{{1, 0, 1}}, // f2  = (0.5,   0, 0.5)
	{{1, 2, 1}}, // f3  = (0.5,   1, 0.5)
	{{1, 1, 0}}, // f4  = (0.5, 0.5,   0)
	{{1, 1, 2}}, // f5  = (0.5, 0.5,   1)
	{{1, 1, 1}}, // c0  = (0.5, 0.5, 0.5)
}};

// -----------------------------------------------------------------------------

int find_edge(const poly_fem::Mesh3D &mesh, int c, int v1, int v2) {
	std::array<int, 2> v = {{v1, v2}};
	std::sort(v.begin(), v.end());
	for (int lf = 0; lf < mesh.n_element_faces(c); ++lf) {
		auto idx = mesh.get_index_from_element(c, lf, 0);
		for (int lv = 0; lv < mesh.n_face_vertices(idx.face); ++lv) {
			std::array<int, 2> u;
			u[0] = idx.vertex;
			u[1] = mesh.switch_vertex(idx).vertex;
			std::sort(u.begin(), u.end());
			if (u == v) {
				return idx.edge;
			}
			idx = mesh.next_around_face_of_element(idx);
		}
	}
	throw std::runtime_error("Edge not found");
}

int find_face(const poly_fem::Mesh3D &mesh, int c, int v1, int v2, int v3, int v4) {
	std::array<int, 4> v = {{v1, v2, v3, v4}};
	std::sort(v.begin(), v.end());
	for (int lf = 0; lf < mesh.n_element_faces(c); ++lf) {
		auto idx = mesh.get_index_from_element(c, lf, 0);
		assert(mesh.n_face_vertices(idx.face) == 4);
		std::array<int, 4> u;
		for (int lv = 0; lv < mesh.n_face_vertices(idx.face); ++lv) {
			u[lv] = idx.vertex;
			idx = mesh.next_around_face_of_element(idx);
		}
		std::sort(u.begin(), u.end());
		if (u == v) {
			return idx.face;
		}
	}
	return 0;
}

// -----------------------------------------------------------------------------

constexpr std::array<int, 4> local_edge_to_interface_flag = {{
	poly_fem::InterfaceData::BOTTOM_FLAG,
	poly_fem::InterfaceData::RIGHT_FLAG,
	poly_fem::InterfaceData::TOP_FLAG,
	poly_fem::InterfaceData::LEFT_FLAG,
}};

// -----------------------------------------------------------------------------

std::array<int, 8> linear_hex_local_to_global(const poly_fem::Mesh3D &mesh, int c) {
	assert(mesh.n_element_vertices(c) == 8);
	assert(mesh.n_element_faces(c) == 6);

	// Vertex nodes
	std::array<int, 8> l2g;
	int lv = 0;
	for (int vi : mesh.get_ordered_vertices_from_hex(c)) {
		l2g[lv++] = vi;
	}

	return l2g;
}

// -----------------------------------------------------------------------------

std::array<int, 27> quadr_hex_local_to_global(const poly_fem::Mesh3D &mesh, int c) {
	assert(mesh.n_element_vertices(c) == 8);
	assert(mesh.n_element_faces(c) == 6);

	int e_offset = mesh.n_pts();
	int f_offset = e_offset + mesh.n_edges();
	int c_offset = f_offset + mesh.n_faces();

	// Vertex nodes
	auto v = linear_hex_local_to_global(mesh, c);

	// Edge nodes
	Eigen::Matrix<int, 12, 1> e;
	Eigen::Matrix<int, 12, 2> ev;
	ev.row(0)  << v[0], v[1];
	ev.row(1)  << v[1], v[2];
	ev.row(2)  << v[2], v[3];
	ev.row(3)  << v[3], v[0];
	ev.row(4)  << v[0], v[4];
	ev.row(5)  << v[1], v[5];
	ev.row(6)  << v[2], v[6];
	ev.row(7)  << v[3], v[7];
	ev.row(8)  << v[4], v[5];
	ev.row(9)  << v[5], v[6];
	ev.row(10) << v[6], v[7];
	ev.row(11) << v[7], v[4];
	for (int le = 0; le < e.rows(); ++le) {
		e[le] = find_edge(mesh, c, ev(le, 0), ev(le, 1));
	}

	// Face nodes
	Eigen::Matrix<int, 6, 1> f;
	Eigen::Matrix<int, 6, 4> fv;
	fv.row(0) << v[0], v[3], v[4], v[7];
	fv.row(1) << v[1], v[2], v[5], v[6];
	fv.row(2) << v[0], v[1], v[5], v[4];
	fv.row(3) << v[3], v[2], v[6], v[7];
	fv.row(4) << v[0], v[1], v[2], v[3];
	fv.row(5) << v[4], v[5], v[6], v[7];
	for (int lf = 0; lf < f.rows(); ++lf) {
		f[lf] = find_face(mesh, c, fv(lf, 0), fv(lf, 1), fv(lf, 2), fv(lf, 3));
	}

	// Local to global mapping of node indices
	std::array<int, 27> l2g;

	// Assign global ids to local nodes
	{
		int i = 0;
		for (size_t lv = 0; lv < v.size(); ++lv) {
			l2g[i++] = v[lv];
		}
		for (int le = 0; le < e.rows(); ++le) {
			l2g[i++] = e_offset + e[le];
		}
		for (int lf = 0; lf < f.rows(); ++lf) {
			l2g[i++] = f_offset + f[lf];
		}
		l2g[i++] = c_offset + c;
	}

	return l2g;
}

// -----------------------------------------------------------------------------

///
/// @brief      Compute the list of global nodes for the mesh. If discr_order is 1 then this is the
///             same as the vertices of the input mesh. If discr_order is 2, then nodes are inserted
///             in the middle of each simplex (edge, facet, cell), and node per elements are
///             numbered accordingly.
///
/// @param[in]  mesh               The input mesh
/// @param[in]  discr_order        The discretization order
/// @param[out] nodes              The node positions
/// @param[out] boundary_nodes     List of boundary node indices
/// @param[out] element_nodes_id   List of node indices per element
/// @param[out] local_boundary     Which facet of the element are on the boundary
/// @param[out] poly_face_to_data  Data for faces at the interface with a polyhedra
///
void compute_nodes(
	const poly_fem::Mesh3D &mesh,
	const int discr_order,
	std::vector<Eigen::RowVector3d> &nodes,
	std::vector<int> &boundary_nodes,
	std::vector<std::vector<int> > &element_nodes_id,
	std::vector<poly_fem::LocalBoundary> &local_boundary,
	std::map<int, poly_fem::InterfaceData> &poly_face_to_data)
{
	const int n_nodes = mesh.n_pts() + (discr_order > 1 ? mesh.n_edges() + mesh.n_faces() + mesh.n_elements() : 0);
	const int e_offset = mesh.n_pts();
	const int f_offset = e_offset + mesh.n_edges();
	const int c_offset = f_offset + mesh.n_faces();
	Eigen::MatrixXd all_nodes(n_nodes, 3);
	std::vector<bool> is_boundary(n_nodes, false);
	std::vector<int> remapped_node(n_nodes, -1);

	// Step 1: Compute all node positions + node boundary tag
	{
		for (int v = 0; v < mesh.n_pts(); ++v) {
			all_nodes.row(v) = mesh.point(v);
			is_boundary[v] = mesh.is_boundary_vertex(v);
		}
		if (discr_order > 1) {
			Eigen::MatrixXd bary;
			mesh.edge_barycenters(bary);
			for (int e = 0; e < mesh.n_edges(); ++e) {
				all_nodes.row(e_offset + e) = bary.row(e);
				is_boundary[e_offset + e] = mesh.is_boundary_edge(e);
			}
			mesh.face_barycenters(bary);
			for (int f = 0; f < mesh.n_faces(); ++f) {
				all_nodes.row(f_offset + f) = bary.row(f);
				is_boundary[f_offset + f] = mesh.is_boundary_face(f);
			}
			mesh.cell_barycenters(bary);
			for (int c = 0; c < mesh.n_elements(); ++c) {
				all_nodes.row(c_offset + c) = bary.row(c);
				is_boundary[c_offset + c] = false;
			}
		}
	}

	nodes.clear();
	local_boundary.clear();
	local_boundary.resize(mesh.n_elements());
	element_nodes_id.resize(mesh.n_elements());

	// Step 2: Keep only read real nodes + compute parametric boundary tag
	for (int c = 0; c < mesh.n_elements(); ++c) {
		bool is_hex = (mesh.n_element_vertices(c) == 8) && (mesh.n_element_faces(c) == 6);
		if (!is_hex) { continue; } // Skip polytopes

		// Create remapped node array for element
		auto remap_nodes = [&] (auto l2g) {
			std::vector<int> res;
			res.reserve(l2g.size());
			for (int id : l2g) {
				if (remapped_node[id] < 0) {
					remapped_node[id] = nodes.size();
					Eigen::RowVector3d pos = all_nodes.row(id);
					nodes.push_back(pos);
					if (is_boundary[id]) {
						boundary_nodes.push_back(remapped_node[id]);
					}
				}
				res.push_back(remapped_node[id]);
			}
			return res;
		};
		if (discr_order == 1) {
			element_nodes_id[c] = remap_nodes(linear_hex_local_to_global(mesh, c));
		} else {
			element_nodes_id[c] = remap_nodes(quadr_hex_local_to_global(mesh, c));
		}

		// List of faces around the quad
		std::array<int, 6> f;
		{
			auto l2g = quadr_hex_local_to_global(mesh, c);
			for (int lf = 0; lf < 6; ++lf) {
				f[lf] = l2g[8+12+lf] - f_offset;
			}
		}

		// Set boundary faces
		if (mesh.is_boundary_face(f[0])) {
			local_boundary[c].set_left_edge_id(f[0]);
			local_boundary[c].set_left_boundary();
		}
		if (mesh.is_boundary_face(f[1])) {
			local_boundary[c].set_right_edge_id(f[1]);
			local_boundary[c].set_right_boundary();
		}
		if (mesh.is_boundary_face(f[2])) {
			local_boundary[c].set_front_edge_id(f[2]);
			local_boundary[c].set_front_boundary();
		}
		if (mesh.is_boundary_face(f[3])) {
			local_boundary[c].set_back_edge_id(f[3]);
			local_boundary[c].set_back_boundary();
		}
		if (mesh.is_boundary_face(f[4])) {
			local_boundary[c].set_bottom_edge_id(f[4]);
			local_boundary[c].set_bottom_boundary();
		}
		if (mesh.is_boundary_face(f[5])) {
			local_boundary[c].set_top_edge_id(f[5]);
			local_boundary[c].set_top_boundary();
		}
	}

	// Step 3: Iterate over edges of polygons and compute interface weights
	for (int f = 0; f < mesh.n_elements(); ++f) {
		if (mesh.n_element_vertices(f) == 4) { continue; } // Skip quads

		// auto index = mesh.get_index_from_face(f, 0);
		// for (int lv = 0; lv < mesh.n_element_vertices(f); ++lv) {
		// 	auto index2 = mesh.switch_face(index);
		// 	if (index2.face >= 0) {
		// 		// Opposite face is a quad, we need to set interface data
		// 		int f2 = index2.face;
		// 		assert(mesh.n_element_vertices(f2) == 4);
		// 		auto abc = poly_fem::FEBasis3d::quadr_quad_edge_local_nodes(mesh, index2);
		// 		poly_fem::InterfaceData data;
		// 		data.element_id = index2.face;
		// 		data.flag = local_edge_to_interface_flag[abc[1] - 4];
		// 		if (discr_order == 2) {
		// 			for (auto local_node : abc) {
		// 				data.node_id.push_back(element_nodes_id[f2][local_node]);
		// 			}
		// 			data.local_indices.assign(abc.begin(), abc.end());
		// 		} else {
		// 			assert(discr_order == 1);
		// 			auto ab = poly_fem::FEBasis2d::linear_quad_edge_local_nodes(mesh, index2);
		// 			for (auto local_node : ab) {
		// 				data.node_id.push_back(element_nodes_id[f2][local_node]);
		// 			}
		// 			data.local_indices.assign(ab.begin(), ab.end());
		// 		}
		// 		data.vals.assign(data.local_indices.size(), 1);
		// 		poly_face_to_data[index2.edge] = data;
		// 	}
		// 	index = mesh.next_around_face(index);
		// }
	}
}

/*
Axes:
  z
  │
  o──x
 ╱
y

Boundaries:
X axis: left/right
Y axis: back/front
Z axis: bottom/top

Face nodes:
      v7──────x─────v6
      ╱┆     ╱      ╱│
     ╱ ┆    ╱      ╱ │
    x┄┄┼┄┄f5┄┄┄┄┄┄x  │
   ╱   x  ╱  f3  ╱┆  x
  ╱    ┆ ╱      ╱ ┆ ╱│
v4─────┼x─────v5  ┆╱ │
 │ f0  ┆┆      │ f1  │
 │    v3┼┄┄┄┄┄x┼┄⌿┼┄v2
 │    ╱ ┆      │╱ ┆ ╱
 x┄┄┄⌿┄f2┄┄┄┄┄┄x  ┆╱
 │  x   ┆ f4   │  x
 │ ╱    ┆      │ ╱
 │╱     ┆      │╱
v0──────x─────v1
*/

// -----------------------------------------------------------------------------

template<class InputIterator, class T>
	int find_index(InputIterator first, InputIterator last, const T& val)
{
	return std::distance(first, std::find(first, last, val));
}

Eigen::RowVector3d linear_hex_local_node_coordinates(int local_index) {
	auto p = linear_hex_local_node[local_index];
	return Eigen::RowVector3d(p[0], p[1], p[2]);
}

Eigen::RowVector3d quadr_hex_local_node_coordinates(int local_index) {
	auto p = quadr_hex_local_node[local_index];
	return Eigen::RowVector3d(p[0], p[1], p[2]) / 2.0;
}

} // anonymous namespace

////////////////////////////////////////////////////////////////////////////////

std::array<int, 4> poly_fem::FEBasis3d::linear_hex_face_local_nodes(
	const Mesh3D &mesh, Navigation3D::Index index)
{
	int c = index.element;
	assert(mesh.n_face_vertices(index.face) == 4);
	assert(mesh.n_element_vertices(c) == 8);
	assert(mesh.n_element_faces(c) == 6);

	// Local to global mapping of node indices
	auto l2g = linear_hex_local_to_global(mesh, c);

	// Extract requested interface
	std::array<int, 4> result;
	for (int lv = 0, i = 0; lv < 4; ++lv) {
		result[i++] = find_index(l2g.begin(), l2g.end(), index.vertex);
		index = mesh.next_around_face_of_element(index);
	}
	return result;
}

Eigen::MatrixXd poly_fem::FEBasis3d::linear_hex_face_local_nodes_coordinates(
	const Mesh3D &mesh, Navigation3D::Index index)
{
	auto idx = linear_hex_face_local_nodes(mesh, index);
	Eigen::MatrixXd res(idx.size(), 3);
	int cnt = 0;
	for (int i : idx) {
		res.row(cnt++) = linear_hex_local_node_coordinates(i);
	}
	return res;
}

// -----------------------------------------------------------------------------

std::array<int, 9> poly_fem::FEBasis3d::quadr_hex_face_local_nodes(
	const Mesh3D &mesh, Navigation3D::Index index)
{
	int c = index.element;
	assert(mesh.n_face_vertices(index.face) == 4);
	assert(mesh.n_element_vertices(c) == 8);
	assert(mesh.n_element_faces(c) == 6);
	int e_offset = mesh.n_pts();
	int f_offset = e_offset + mesh.n_edges();

	// Local to global mapping of node indices
	auto l2g = quadr_hex_local_to_global(mesh, c);

	// Extract requested interface
	std::array<int, 9> result;
	for (int lv = 0, i = 0; lv < 4; ++lv) {
		result[i++] = find_index(l2g.begin(), l2g.end(), index.vertex);
		result[i++] = find_index(l2g.begin(), l2g.end(), e_offset + index.edge);
		index = mesh.next_around_face_of_element(index);
	}
	result[8] = find_index(l2g.begin(), l2g.end(), f_offset + index.face);
	return result;
}

Eigen::MatrixXd poly_fem::FEBasis3d::quadr_hex_face_local_nodes_coordinates(
	const Mesh3D &mesh, Navigation3D::Index index)
{
	auto idx = quadr_hex_face_local_nodes(mesh, index);
	Eigen::MatrixXd res(idx.size(), 3);
	int cnt = 0;
	for (int i : idx) {
		res.row(cnt++) = quadr_hex_local_node_coordinates(i);
	}
	return res;
}

////////////////////////////////////////////////////////////////////////////////

void poly_fem::FEBasis3d::quadr_hex_basis_value(const int local_index, const Eigen::MatrixXd &xne, Eigen::MatrixXd &val) {
	auto x=xne.col(0).array();
	auto n=xne.col(1).array();
	auto e=xne.col(2).array();

	std::array<int, 3> idx = quadr_hex_local_node[local_index];
	val = theta(idx[0], x).array() * theta(idx[1], n).array() * theta(idx[2], e).array();
}

void poly_fem::FEBasis3d::quadr_hex_basis_grad(const int local_index, const Eigen::MatrixXd &xne, Eigen::MatrixXd &val) {
	auto x=xne.col(0).array();
	auto n=xne.col(1).array();
	auto e=xne.col(2).array();

	std::array<int, 3> idx = quadr_hex_local_node[local_index];

	val.resize(xne.rows(), 3);
	val.col(0) = dtheta(idx[0], x).array() * theta(idx[1], n).array() * theta(idx[2], e).array();
	val.col(1) = theta(idx[0], x).array() * dtheta(idx[1], n).array() * theta(idx[2], e).array();
	val.col(2) = theta(idx[0], x).array() * theta(idx[1], n).array() * dtheta(idx[2], e).array();
}

int poly_fem::FEBasis3d::build_bases(
	const Mesh3D &mesh,
	const int quadrature_order,
	const int discr_order,
	std::vector< ElementBases > &bases,
	std::vector< LocalBoundary > &local_boundary,
	std::vector< int > &boundary_nodes,
	std::map<int, InterfaceData> &poly_face_to_data)
{
	assert(mesh.is_volume());

	std::vector<Eigen::RowVector3d> nodes;
	std::vector<std::vector<int>> element_nodes_id;
	compute_nodes(mesh, discr_order, nodes, boundary_nodes, element_nodes_id, local_boundary, poly_face_to_data);

	HexQuadrature hex_quadrature;
	bases.resize(mesh.n_elements());
	for (int e = 0; e < mesh.n_elements(); ++e) {
		ElementBases &b = bases[e];
		const int n_el_vertices = mesh.n_element_vertices(e);
		const int n_el_bases = (int) element_nodes_id[e].size();
		b.bases.resize(n_el_bases);

		if (n_el_vertices == 8) {
			hex_quadrature.get_quadrature(quadrature_order, b.quadrature);
			b.bases.resize(n_el_bases);

			for (int j = 0; j < n_el_bases; ++j) {
				const int global_index = element_nodes_id[e][j];

				b.bases[j].init(global_index, j, nodes[global_index]);

				if (discr_order == 1) {
					b.bases[j].set_basis([j](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
						{ linear_hex_basis_value(j, uv, val); });
					b.bases[j].set_grad([j](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
						{ linear_hex_basis_grad(j, uv, val); });
				} else if (discr_order == 2) {
					b.bases[j].set_basis([j](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
						{ quadr_hex_basis_value(j, uv, val); });
					b.bases[j].set_grad([j](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
						{ quadr_hex_basis_grad(j, uv, val); });
				} else {
					assert(false);
				}
			}
		} else {
			// Polyhedra bases are built later on
			// assert(false);
		}
	}

	return (int) nodes.size();
}
