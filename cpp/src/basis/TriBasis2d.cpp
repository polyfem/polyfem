////////////////////////////////////////////////////////////////////////////////
#include "TriBasis2d.hpp"
#include "MeshNodes.hpp"
#include "TriQuadrature.hpp"
#include <igl/viewer/Viewer.h>
#include <cassert>
#include <array>
////////////////////////////////////////////////////////////////////////////////

using namespace poly_fem;

/*

Axes:
 y
 |
 o──x

Boundaries:
X axis: left/right
Y axis: bottom/top

Corner nodes:
  v2
 │  \
 │   \
 │    \
 x      x
 │         \
 │          \
 │           \
v0──────x─────v1

v0 = (0, 0)
v1 = (1, 0)
v2 = (0, 1)

Edge nodes:
  x
 │  \
 │   \
 │    \
e2      e1
 │        \
 │         \
 │          \
 x──────e0───x

e0  = (0.5,   0)
e1  = (0.5, 0.5)
e2  = (  0, 0.5)

*/

namespace {

// -----------------------------------------------------------------------------

constexpr std::array<std::array<int, 2>, 8> linear_tri_local_node = {{
	{{0, 0}}, // v0  = (0, 0)
	{{1, 0}}, // v1  = (1, 0)
	{{0, 1}}, // v2  = (0, 1)
}};

void linear_tri_basis_value(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) {
	auto u=uv.col(0).array();
	auto v=uv.col(1).array();
	switch(local_index)
	{
		case 0: val = 1 - u - v; break;
		case 1: val = u; break;
		case 2: val = v; break;
		default: assert(false);
	}
}

void linear_tri_basis_grad(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) {
	val.resize(uv.rows(), uv.cols());
	switch(local_index)
	{
		case 0: val.col(0).setConstant(-1); val.col(1).setConstant(-1); break;
		case 1: val.col(0).setConstant( 1); val.col(1).setConstant( 0); break;
		case 2: val.col(0).setConstant( 0); val.col(1).setConstant( 1); break;
		default: assert(false);
	}
}

// -----------------------------------------------------------------------------

constexpr std::array<std::array<int, 2>, 6> quadr_tri_local_node = {{
	{{0, 0}}, // v0  = (0, 0)
	{{2, 0}}, // v1  = (1, 0)
	{{0, 2}}, // v2  = (0, 1)
	{{1, 0}}, // e0  = (0.5,   0)
	{{1, 1}}, // e1  = (0.5, 0.5)
	{{0, 1}}, // e3  = (  0, 0.5)
}};

void quadr_tri_basis_value(
	const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
{
	auto u=uv.col(0).array();
	auto v=uv.col(1).array();
	switch(local_index)
	{
		case 0: val = (1 - u - v)*(1-2*u-2*v); break;
		case 1: val = u*(2*u-1); break;
		case 2: val = v*(2*v-1); break;

		case 3: val = 4*u*(1-u-v); break;
		case 4: val = 4*u*v; break;
		case 5: val = 4*v*(1-u-v); break;
		default: assert(false);
	}
}

void quadr_tri_basis_grad(
	const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
{
	auto u=uv.col(0).array();
	auto v=uv.col(1).array();
	val.resize(uv.rows(), uv.cols());
	switch(local_index)
	{
		case 0:
		val.col(0) = 4*u+4*v-3;
		val.col(1) = 4*u+4*v-3;
		break;

		case 1:
		val.col(0) = 4*u -1;
		val.col(1).setZero();
		break;

		case 2:
		val.col(0).setZero();
		val.col(1) = 4*v - 1;
		break;


		case 3:
		val.col(0) = 4 - 8*u - 4*v;
		val.col(1) = -4*u;
		break;

		case 4:
		val.col(0) = 4*v;
		val.col(1) = 4*u;
		break;

		case 5:
		val.col(0) = -4*v;
		val.col(1) = 4 - 4*u - 8*v;
		break;
		default: assert(false);
	}
}

// -----------------------------------------------------------------------------

poly_fem::Navigation::Index find_edge(const poly_fem::Mesh2D &mesh, int f, int v1, int v2) {
	std::array<int, 2> v = {{v1, v2}};
	std::sort(v.begin(), v.end());
	auto idx = mesh.get_index_from_face(f, 0);
	for (int lv = 0; lv < mesh.n_face_vertices(idx.face); ++lv) {
		std::array<int, 2> u;
		u[0] = idx.vertex;
		u[1] = mesh.switch_vertex(idx).vertex;
		std::sort(u.begin(), u.end());
		if (u == v) {
			return idx;
		}
		idx = mesh.next_around_face(idx);
	}
	throw std::runtime_error("Edge not found");
}

// -----------------------------------------------------------------------------

std::array<int, 3> linear_tri_local_to_global(const Mesh2D &mesh, int f) {
	// assert(mesh.is_cube(f));

	// Vertex nodes
	std::array<int, 3> l2g;
	for (int lv = 0; lv < 3; ++lv) {
		l2g[lv] = mesh.face_vertex(f, lv);
	}

	return l2g;
}

// -----------------------------------------------------------------------------

std::array<int, 6> quadr_tri_local_to_global(const Mesh2D &mesh, int f) {
	// assert(mesh.is_cube(f));

	int edge_offset = mesh.n_vertices();
	int face_offset = edge_offset + mesh.n_edges();

	// Vertex nodes
	auto v = linear_tri_local_to_global(mesh, f);

	// Edge nodes
	Eigen::Matrix<int, 3, 1> e;
	Eigen::Matrix<int, 3, 2> ev;
	ev.row(0) << v[0], v[1];
	ev.row(1) << v[1], v[2];
	ev.row(2) << v[2], v[0];
	for (int le = 0; le < e.rows(); ++le) {
		e[le] = find_edge(mesh, f, ev(le, 0), ev(le, 1)).edge;
	}

	// Local to global mapping of node indices
	std::array<int, 6> l2g;

	// Assign global ids to local nodes
	{
		int i = 0;
		for (size_t lv = 0; lv < v.size(); ++lv) {
			l2g[i++] = v[lv];
		}
		for (int le = 0; le < e.rows(); ++le) {
			l2g[i++] = edge_offset + e[le];
		}
	}

	return l2g;
}

// -----------------------------------------------------------------------------

///
/// @brief      Compute the list of global nodes for the mesh. If discr_order is 1 then this is the
///             same as the vertices of the input mesh. If discr_order is 2, then nodes are inserted
///             in the middle of each simplex (edge, facet, cell), and nodes per elements are
///             numbered accordingly.
///
/// @param[in]  mesh               The input mesh
/// @param[in]  discr_order        The discretization order
/// @param[in]  nodes              Lazy evaluator for node ids
/// @param[out] element_nodes_id   List of node indices per element
/// @param[out] local_boundary     Which facet of the element are on the boundary
/// @param[out] poly_edge_to_data  Data for edges at the interface with a polygon
///
void compute_nodes(
	const poly_fem::Mesh2D &mesh,
	const int discr_order,
	MeshNodes &nodes,
	std::vector<std::vector<int> > &element_nodes_id,
	std::vector<poly_fem::LocalBoundary> &local_boundary,
	std::map<int, poly_fem::InterfaceData> &poly_edge_to_data)
{
	// Step 1: Assign global node ids for each tri
	local_boundary.clear();
	local_boundary.resize(mesh.n_faces());
	element_nodes_id.resize(mesh.n_faces());
	for (int f = 0; f < mesh.n_faces(); ++f) {
		// if (mesh.is_polytope(f)) { continue; } // Skip polygons

		if (discr_order == 1) {
			for (int id : linear_tri_local_to_global(mesh, f)) {
				element_nodes_id[f].push_back(nodes.node_id_from_primitive(id));
			}
		} else {
			for (int id : quadr_tri_local_to_global(mesh, f)) {
				element_nodes_id[f].push_back(nodes.node_id_from_primitive(id));
			}
		}

		// List of edges around the tri
		std::array<int, 3> e;
		{
			auto l2g = quadr_tri_local_to_global(mesh, f);
			for (int le = 0; le < 3; ++le) {
				e[le] = l2g[3+le] - mesh.n_vertices();
			}
		}

		if (mesh.is_boundary_edge(e[1])) {
			local_boundary[f].set_right_edge_id(e[1]);
			local_boundary[f].set_right_boundary();
		}
		if (mesh.is_boundary_edge(e[0])) {
			local_boundary[f].set_bottom_edge_id(e[0]);
			local_boundary[f].set_bottom_boundary();
		}
		if (mesh.is_boundary_edge(e[2])) {
			local_boundary[f].set_top_edge_id(e[2]);
			local_boundary[f].set_top_boundary();
		}
	}
}



// -----------------------------------------------------------------------------

template<class InputIterator, class T>
	int find_index(InputIterator first, InputIterator last, const T& val)
{
	return std::distance(first, std::find(first, last, val));
}

// -----------------------------------------------------------------------------

Eigen::RowVector2d linear_tri_local_node_coordinates(int local_index) {
	auto p = linear_tri_local_node[local_index];
	return Eigen::RowVector2d(p[0], p[1]);
}

Eigen::RowVector2d quadr_tri_local_node_coordinates(int local_index) {
	auto p = quadr_tri_local_node[local_index];
	return Eigen::RowVector2d(p[0], p[1]) / 2.0;
}

} // anonymous namespace

////////////////////////////////////////////////////////////////////////////////

std::array<int, 2> poly_fem::TriBasis2d::linear_tri_edge_local_nodes(
	const Mesh2D &mesh, Navigation::Index index)
{
	int f = index.face;
	assert(mesh.is_cube(f));

	// Local to global mapping of node indices
	auto l2g = linear_tri_local_to_global(mesh, f);

	// Extract requested interface
	std::array<int, 2> result;
	result[0] = find_index(l2g.begin(), l2g.end(), index.vertex);
	result[1] = find_index(l2g.begin(), l2g.end(), mesh.switch_vertex(index).vertex);
	return result;
}

Eigen::MatrixXd poly_fem::TriBasis2d::linear_tri_edge_local_nodes_coordinates(
	const Mesh2D &mesh, Navigation::Index index)
{
	auto idx = linear_tri_edge_local_nodes(mesh, index);
	Eigen::MatrixXd res(idx.size(), 2);
	int cnt = 0;
	for (int i : idx) {
		res.row(cnt++) = linear_tri_local_node_coordinates(i);
	}
	return res;
}

// -----------------------------------------------------------------------------

std::array<int, 3> poly_fem::TriBasis2d::quadr_tri_edge_local_nodes(
	const Mesh2D &mesh, Navigation::Index index)
{
	int f = index.face;
	assert(mesh.is_cube(f));
	int e_offset = mesh.n_vertices();

	// Local to global mapping of node indices
	auto l2g = quadr_tri_local_to_global(mesh, f);

	// Extract requested interface
	std::array<int, 3> result;
	result[0] = find_index(l2g.begin(), l2g.end(), index.vertex);
	result[1] = find_index(l2g.begin(), l2g.end(), e_offset + index.edge);
	result[2] = find_index(l2g.begin(), l2g.end(), mesh.switch_vertex(index).vertex);
	return result;
}

Eigen::MatrixXd poly_fem::TriBasis2d::quadr_tri_edge_local_nodes_coordinates(
	const Mesh2D &mesh, Navigation::Index index)
{
	auto idx = quadr_tri_edge_local_nodes(mesh, index);
	Eigen::MatrixXd res(idx.size(), 2);
	int cnt = 0;
	for (int i : idx) {
		res.row(cnt++) = quadr_tri_local_node_coordinates(i);
	}
	return res;
}

////////////////////////////////////////////////////////////////////////////////

// -----------------------------------------------------------------------------

int poly_fem::TriBasis2d::build_bases(
	const Mesh2D &mesh,
	const int quadrature_order,
	const int discr_order,
	std::vector<ElementBases> &bases,
	std::vector<LocalBoundary> &local_boundary,
	std::vector<int> &boundary_nodes,
	std::map<int, InterfaceData> &poly_edge_to_data)
{
	assert(!mesh.is_volume());

	MeshNodes nodes(mesh, discr_order == 1);
	std::vector<std::vector<int>> element_nodes_id;
	compute_nodes(mesh, discr_order, nodes, element_nodes_id, local_boundary, poly_edge_to_data);
	boundary_nodes = nodes.boundary_nodes();


	bases.resize(mesh.n_faces());
	for (int e = 0; e < mesh.n_faces(); ++e) {
		ElementBases &b = bases[e];
		const int n_el_bases = (int) element_nodes_id[e].size();
		b.bases.resize(n_el_bases);

		b.set_quadrature([quadrature_order](Quadrature &quad){
			TriQuadrature tri_quadrature;
			tri_quadrature.get_quadrature(quadrature_order, quad);
		});

		b.bases.resize(n_el_bases);

		for (int j = 0; j < n_el_bases; ++j) {
			const int global_index = element_nodes_id[e][j];

			b.bases[j].init(global_index, j, nodes.node_position(global_index));

			if (discr_order == 1) {
				b.bases[j].set_basis([j](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
					{ linear_tri_basis_value(j, uv, val); });
				b.bases[j].set_grad([j](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
					{ linear_tri_basis_grad(j, uv, val); });
			} else if (discr_order == 2) {
				b.bases[j].set_basis([j](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
					{ quadr_tri_basis_value(j, uv, val); });
				b.bases[j].set_grad([j](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
					{ quadr_tri_basis_grad(j, uv, val); });
			} else {
				assert(false);
			}
		}
	}

	return nodes.n_nodes();
}
