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

v0 = (0, 1, 0)
v1 = (1, 1, 0)
v2 = (1, 0, 0)
v3 = (0, 0, 0)
v4 = (0, 1, 1)
v5 = (1, 1, 1)
v6 = (1, 0, 1)
v7 = (0, 0, 1)

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

e0  = (0.5,   1,   0)
e1  = (  1, 0.5,   0)
e2  = (0.5,   0,   0)
e3  = (  0, 0.5,   0)
e4  = (  0,   1, 0.5)
e5  = (  1,   1, 0.5)
e6  = (  1,   0, 0.5)
e7  = (  0,   0, 0.5)
e8  = (0.5,   1,   1)
e9  = (  1, 0.5,   1)
e10 = (0.5,   0,   1)
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
f2  = (0.5,   1, 0.5)
f3  = (0.5,   0, 0.5)
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

constexpr std::array<std::array<int, 3>, 8> linear_hex_dofs = {{
	{{0, 0, 0}}, // v0  = (0, 1, 0)
	{{1, 0, 0}}, // v1  = (1, 1, 0)
	{{1, 1, 0}}, // v2  = (1, 0, 0)
	{{0, 1, 0}}, // v3  = (0, 0, 0)
	{{0, 0, 1}}, // v4  = (0, 1, 1)
	{{1, 0, 1}}, // v5  = (1, 1, 1)
	{{1, 1, 1}}, // v6  = (1, 0, 1)
	{{0, 1, 1}}, // v7  = (0, 0, 1)
}};

void linear_hex_basis_value(const int local_index, const Eigen::MatrixXd &xne, Eigen::MatrixXd &val) {
	auto x=xne.col(0).array();
	auto n=xne.col(1).array();
	auto e=xne.col(2).array();

	std::array<int, 3> idx = linear_hex_dofs[local_index];
	val = alpha(idx[0], x).array() * alpha(idx[1], n).array() * alpha(idx[2], e).array();
}

void linear_hex_basis_grad(const int local_index, const Eigen::MatrixXd &xne, Eigen::MatrixXd &val) {
	auto x=xne.col(0).array();
	auto n=xne.col(1).array();
	auto e=xne.col(2).array();

	std::array<int, 3> idx = linear_hex_dofs[local_index];

	val.resize(xne.rows(), 3);
	val.col(0) = dalpha(idx[0], x).array() * alpha(idx[1], n).array() * alpha(idx[2], e).array();
	val.col(1) = alpha(idx[0], x).array() * dalpha(idx[1], n).array() * alpha(idx[2], e).array();
	val.col(2) = alpha(idx[0], x).array() * alpha(idx[1], n).array() * dalpha(idx[2], e).array();
}

// -----------------------------------------------------------------------------

constexpr std::array<std::array<int, 3>, 27> quadr_hex_dofs = {{
	{{0, 0, 0}}, // v0  = (0, 1, 0)
	{{2, 0, 0}}, // v1  = (1, 1, 0)
	{{2, 2, 0}}, // v2  = (1, 0, 0)
	{{0, 2, 0}}, // v3  = (0, 0, 0)
	{{0, 0, 2}}, // v4  = (0, 1, 1)
	{{2, 0, 2}}, // v5  = (1, 1, 1)
	{{2, 2, 2}}, // v6  = (1, 0, 1)
	{{0, 2, 2}}, // v7  = (0, 0, 1)
	{{1, 0, 0}}, // e0  = (0.5,   1,   0)
	{{2, 1, 0}}, // e1  = (  1, 0.5,   0)
	{{1, 2, 0}}, // e2  = (0.5,   0,   0)
	{{0, 1, 0}}, // e3  = (  0, 0.5,   0)
	{{0, 0, 1}}, // e4  = (  0,   1, 0.5)
	{{2, 0, 1}}, // e5  = (  1,   1, 0.5)
	{{2, 2, 1}}, // e6  = (  1,   0, 0.5)
	{{0, 2, 1}}, // e7  = (  0,   0, 0.5)
	{{1, 0, 2}}, // e8  = (0.5,   1,   1)
	{{2, 1, 2}}, // e9  = (  1, 0.5,   1)
	{{1, 2, 2}}, // e10 = (0.5,   0,   1)
	{{0, 1, 2}}, // e11 = (  0, 0.5,   1)
	{{0, 1, 1}}, // f0  = (  0, 0.5, 0.5)
	{{2, 1, 1}}, // f1  = (  1, 0.5, 0.5)
	{{1, 0, 1}}, // f2  = (0.5,   1, 0.5)
	{{1, 2, 1}}, // f3  = (0.5,   0, 0.5)
	{{1, 1, 0}}, // f4  = (0.5, 0.5,   0)
	{{1, 1, 2}}, // f5  = (0.5, 0.5,   1)
	{{1, 1, 1}}, // c0  = (0.5, 0.5, 0.5)
}};

void quadr_hex_basis_value(const int local_index, const Eigen::MatrixXd &xne, Eigen::MatrixXd &val) {
	auto x=xne.col(0).array();
	auto n=xne.col(1).array();
	auto e=xne.col(2).array();

	std::array<int, 3> idx = quadr_hex_dofs[local_index];
	val = theta(idx[0], x).array() * theta(idx[1], n).array() * theta(idx[2], e).array();
}

void quadr_hex_basis_grad(const int local_index, const Eigen::MatrixXd &xne, Eigen::MatrixXd &val) {
	auto x=xne.col(0).array();
	auto n=xne.col(1).array();
	auto e=xne.col(2).array();

	std::array<int, 3> idx = quadr_hex_dofs[local_index];

	val.resize(xne.rows(), 3);
	val.col(0) = dtheta(idx[0], x).array() * theta(idx[1], n).array() * theta(idx[2], e).array();
	val.col(1) = theta(idx[0], x).array() * dtheta(idx[1], n).array() * theta(idx[2], e).array();
	val.col(2) = theta(idx[0], x).array() * theta(idx[1], n).array() * dtheta(idx[2], e).array();
}

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

Eigen::Vector3d barycenter(const Eigen::MatrixXd &nodes, const std::vector<int> ids) {
	Eigen::Vector3d p;
	p.setZero();
	for (int i : ids) {
		p += nodes.row(i);
	}
	return p / ids.size();
}

// -----------------------------------------------------------------------------

///
/// @brief      Compute the list of global dofs for the mesh. If discr_order is
///             1 then this is the same as the vertices of the input mesh. If
///             discr_order is 2, then dofs are inserted in the middle of each
///             simplex (edge, facet, cell), and dofs per elements are numbered
///             accordingly.
///
/// @param[in]  mesh            The input mesh
/// @param[in]  discr_order     The discretization order
/// @param[out] nodes           The dofs positions
/// @param[out] boundary_nodes  List of boundary dof indices
/// @param[out] element_dofs    List of dof indices per element
/// @param[out] local_boundary  Which facet of the element are on the boundary
///
void compute_dofs(
	const poly_fem::Mesh3D &mesh,
	const int discr_order,
	Eigen::MatrixXd &nodes,
	std::vector<int> &boundary_nodes,
	std::vector<std::vector<int> > &element_dofs,
	std::vector<poly_fem::LocalBoundary> &local_boundary)
{
	if (discr_order == 1) {
		// Compute dofs positions + whether it is a boundary dof
		nodes.resize(mesh.n_pts(), 3);
		Eigen::MatrixXd tmp(1, 3);
		for (int v = 0; v < mesh.n_pts(); ++v) {
			if (mesh.is_boundary_vertex(v)) {
				boundary_nodes.push_back(v);
			}
			mesh.point(v, tmp);
			nodes.row(v) = tmp;
		}
		// Assign global ids to dofs
		element_dofs.reserve(mesh.n_elements());
		for (int c = 0; c < mesh.n_elements(); ++c) {
			element_dofs.emplace_back();
			assert(mesh.n_element_vertices(c) == 8);
			assert(mesh.n_element_faces(c) == 6);
			for (int lv = 0; lv < mesh.n_element_vertices(c); ++lv) {
				element_dofs.back().push_back(mesh.vertex_global_index(c, lv));
			}
		}

		// Compute boundary facets
		local_boundary.clear();
		local_boundary.resize(mesh.n_elements());
		for (int c = 0; c < mesh.n_elements(); ++c) {
			// Vertices
			Eigen::Matrix<int, 8, 1> v;
			{
				int lv = 0;
				for (int vi : mesh.get_ordered_vertices_from_hex(c)) {
					v[lv++] = vi;
				}
			}

			// Faces
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
	} else if (discr_order == 2) {
		int e_offset = mesh.n_pts();
		int f_offset = e_offset + mesh.n_edges();
		int c_offset = f_offset + mesh.n_faces();
		int ndofs = c_offset + mesh.n_elements();
		nodes.resize(ndofs, 3);
		Eigen::MatrixXd tmp(1, 3);
		element_dofs.reserve(mesh.n_elements());
		local_boundary.clear();
		local_boundary.resize(mesh.n_elements());
		for (int c = 0; c < mesh.n_elements(); ++c) {
			assert(mesh.n_element_vertices(c) == 8);
			assert(mesh.n_element_faces(c) == 6);

			// Corner dofs position + is boundary
			Eigen::Matrix<int, 8, 1> v;
			{
				int lv = 0;
				for (int vi : mesh.get_ordered_vertices_from_hex(c)) {
					v[lv++] = vi;
				}
			}
			for (int lv = 0; lv < v.rows(); ++lv) {
				mesh.point(v[lv], tmp);
				nodes.row(v[lv]) = tmp;
				if (mesh.is_boundary_vertex(v[lv])) {
					boundary_nodes.push_back(v[lv]);
				}
			}

			// Edge dofs position + is boundary
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
				nodes.row(e_offset + e[le]) = barycenter(nodes, {{ ev(le, 0), ev(le, 1) }});
				bool boundary = true;
				for (int k = 0; k < ev.cols(); ++k) {
					if (!mesh.is_boundary_vertex(ev(le, k))) {
						boundary = false;
					}
				}
				if (boundary) {
					boundary_nodes.push_back(e_offset + e[le]);
				}
			}

			// Face dofs position + is boundary
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
				nodes.row(f_offset + f[lf]) = barycenter(nodes, {{ fv(lf, 0), fv(lf, 1), fv(lf, 2), fv(lf, 3) }});
				bool boundary = true;
				for (int k = 0; k < fv.cols(); ++k) {
					if (!mesh.is_boundary_vertex(fv(lf, k))) {
						boundary = false;
					}
				}
				if (boundary) {
					boundary_nodes.push_back(f_offset + f[lf]);
				}
			}

			// Cell dofs position
			nodes.row(c_offset + c) = barycenter(nodes, {{ v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7] }});

			// Assign global ids to dofs
			element_dofs.emplace_back();
			for (int lv = 0; lv < v.rows(); ++lv) {
				element_dofs.back().push_back(v[lv]);
			}
			for (int le = 0; le < e.rows(); ++le) {
				element_dofs.back().push_back(e_offset + e[le]);
			}
			for (int lf = 0; lf < f.rows(); ++lf) {
				element_dofs.back().push_back(f_offset + f[lf]);
			}
			element_dofs.back().push_back(c_offset + c);

			// Eigen::MatrixXd vv(27, 3);
			// int cnt = 0;
			// igl::viewer::Viewer viewer;
			// for (int i : element_dofs.back()) {
			// 	viewer.data.add_label(nodes.row(i), std::to_string(cnt));
			// 	vv.row(cnt++) = nodes.row(i);
			// }
			// viewer.data.set_points(vv, Eigen::RowVector3d(0, 0, 0));
			// viewer.core.align_camera_center(vv);
			// viewer.core.set_rotation_type(igl::viewer::ViewerCore::RotationType::ROTATION_TYPE_TRACKBALL);
			// viewer.launch();

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
	} else {
		throw std::runtime_error("Not implemented");
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

Face dofs:
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


} // anonymous namespace

////////////////////////////////////////////////////////////////////////////////

int poly_fem::FEBasis3d::build_bases(
	const Mesh3D &mesh,
	const int quadrature_order,
	const int discr_order,
	std::vector< ElementBases > &bases,
	std::vector< LocalBoundary > &local_boundary,
	std::vector< int > &boundary_nodes)
{
	assert(mesh.is_volume());

	Eigen::MatrixXd nodes;
	std::vector<std::vector<int>> element_dofs;
	compute_dofs(mesh, discr_order, nodes, boundary_nodes, element_dofs, local_boundary);

	HexQuadrature hex_quadrature;
	bases.resize(mesh.n_elements());
	for (int e = 0; e < mesh.n_elements(); ++e) {
		ElementBases &b = bases[e];
		const int n_el_vertices = mesh.n_element_vertices(e);
		const int n_el_bases = (int) element_dofs[e].size();
		b.bases.resize(n_el_bases);

		if (n_el_vertices == 8) {
			hex_quadrature.get_quadrature(quadrature_order, b.quadrature);
			b.bases.resize(n_el_bases);

			for (int j = 0; j < n_el_bases; ++j) {
				const int global_index = element_dofs[e][j];

				b.bases[j].init(global_index, j, nodes.row(global_index));

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
			assert(false);
		}
	}

	return (int) nodes.rows();
}
