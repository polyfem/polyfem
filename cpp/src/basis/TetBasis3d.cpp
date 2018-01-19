////////////////////////////////////////////////////////////////////////////////
#include "TetBasis3d.hpp"
#include "MeshNodes.hpp"
#include "TetQuadrature.hpp"
#include <igl/viewer/Viewer.h>
#include <cassert>
#include <array>
////////////////////////////////////////////////////////////////////////////////

using namespace poly_fem;

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

v3
 │ \
 │ x\ v2
 │   \╱  \
 x   / x  \
 │  x   \  x
 │ ╱      \ \
 │╱         \\
v0──────x─────v1

v0 = (0, 0, 0)
v1 = (1, 0, 0)
v2 = (0, 1, 0)
v3 = (0, 0, 1)

Edge nodes:

e0  = (0.5,   0,   0)
e1  = (0.5, 0.5,   0)
e2  = (  0, 0.5,   0)
e3  = (  0, 0,   0.5)
e4  = (0.5,   0, 0.5)
e5  = (  0, 0.5, 0.5)

*/

namespace {

// -----------------------------------------------------------------------------

	constexpr std::array<std::array<int, 3>, 4> linear_tet_local_node = {{
	{{0, 0, 0}}, // v0  = (0, 0, 0)
	{{1, 0, 0}}, // v1  = (1, 0, 0)
	{{0, 1, 0}}, // v2  = (0, 1, 0)
	{{0, 0, 1}}, // v3  = (0, 0, 1)
}};

void linear_tet_basis_value(const int local_index, const Eigen::MatrixXd &xne, Eigen::MatrixXd &val) {
	auto x=xne.col(0).array();
	auto n=xne.col(1).array();
	auto e=xne.col(2).array();

	switch(local_index)
	{
		case 0: val = 1 - x - n - e; break;
		case 1: val = x; break;
		case 2: val = n; break;
		case 3: val = e; break;
		default: assert(false);
	}
}

void linear_tet_basis_grad(const int local_index, const Eigen::MatrixXd &xne, Eigen::MatrixXd &val) {
	val.resize(xne.rows(), xne.cols());

	switch(local_index)
	{
		case 0:
		val.col(0).setConstant(-1);
		val.col(1).setConstant(-1);
		val.col(2).setConstant(-1);
		break;

		case 1:
		val.col(0).setConstant( 1);
		val.col(1).setConstant( 0);
		val.col(2).setConstant( 0);
		break;

		case 2:
		val.col(0).setConstant( 0);
		val.col(1).setConstant( 1);
		val.col(2).setConstant( 0);
		break;

		case 3:
		val.col(0).setConstant( 0);
		val.col(1).setConstant( 0);
		val.col(2).setConstant( 1);
		break;

		default: assert(false);
	}
}

// -----------------------------------------------------------------------------

constexpr std::array<std::array<int, 3>, 10> quadr_tet_local_node = {{
	{{0, 0, 0}}, // v0  = (  0,   0,   0)
	{{2, 0, 0}}, // v1  = (  1,   0,   0)
	{{0, 2, 0}}, // v2  = (  0,   1,   0)
	{{0, 0, 2}}, // v3  = (  0,   0,   1)
	{{1, 0, 0}}, // e0  = (0.5,   0,   0)
	{{1, 1, 0}}, // e1  = (0.5, 0.5,   0)
	{{0, 1, 0}}, // e2  = (  0, 0.5,   0)
	{{0, 0, 1}}, // e3  = (  0,   0, 0.5)
	{{1, 0, 1}}, // e4  = (0.5,   0, 0.5)
	{{0, 1, 1}}, // e5  = (  0, 0.5, 0.5)
}};

void quadr_tet_basis_value(const int local_index, const Eigen::MatrixXd &xne, Eigen::MatrixXd &val) {
	auto x=xne.col(0).array();
	auto n=xne.col(1).array();
	auto e=xne.col(2).array();

	switch(local_index)
	{
		case 0: val = (1 - 2*x - 2*n - 2*e)*(1 - x - n - e); break;
		case 1: val = (2*x-1)*x; break;
		case 2: val = (2*n-1)*n; break;
		case 3: val = (2*e-1)*e; break;

		case 4: val = 4*x * (1 - x - n - e); break;
		case 5: val = 4 * x * n; break;
		case 6: val = 4 * (1 - x - n - e) * n; break;

		case 7: val = 4*(1 - x - n - e)*e; break;
		case 8: val = 4*x*e; break;
		case 9: val = 4*n*e; break;
		default: assert(false);
	}
}

void quadr_tet_basis_grad(const int local_index, const Eigen::MatrixXd &xne, Eigen::MatrixXd &val) {
	auto x=xne.col(0).array();
	auto n=xne.col(1).array();
	auto e=xne.col(2).array();

	val.resize(xne.rows(), xne.cols());

	switch(local_index)
	{
		case 0:
		val.col(0) = -3+4*x+4*n+4*e;
		val.col(1) = -3+4*x+4*n+4*e;
		val.col(2) = -3+4*x+4*n+4*e;
		break;
		case 1:
		val.col(0) = 4*x-1;
		val.col(1).setZero();
		val.col(2).setZero();
		break;
		case 2:
		val.col(0).setZero();
		val.col(1) = 4*n-1;
		val.col(2).setZero();
		break;
		case 3:
		val.col(0).setZero();
		val.col(1).setZero();
		val.col(2) = 4*e-1;
		break;

		case 4:
		val.col(0) = 4-8*x-4*n-4*e;
		val.col(1) = -4*x;
		val.col(2) = -4*x;
		break;
		case 5:
		val.col(0) = 4*n;
		val.col(1) = 4*x;
		val.col(2).setZero();
		break;
		case 6:
		val.col(0) = -4*n;
		val.col(1) = -8*n+4-4*x-4*e;
		val.col(2) = -4*n;
		break;

		case 7:
		val.col(0) = -4*e;
		val.col(1) = -4*e;
		val.col(2) = -8*e+4-4*x-4*n;
		break;
		case 8:
		val.col(0) = 4*e;
		val.col(1).setZero();
		val.col(2) = 4*x;
		break;
		case 9:
		val.col(0).setZero();
		val.col(1) = 4*e;
		val.col(2) = 4*n;
		break;
		default: assert(false);
	}
}

// -----------------------------------------------------------------------------

int find_edge(const poly_fem::Mesh3D &mesh, int c, int v1, int v2) {
	std::array<int, 2> v = {{v1, v2}};
	std::sort(v.begin(), v.end());
	for (int lf = 0; lf < mesh.n_cell_faces(c); ++lf) {
		auto idx = mesh.get_index_from_element(c, lf, 0);
		for (int lv = 0; lv < mesh.n_face_vertices(idx.face); ++lv) {
			std::array<int, 2> u;
			u[0] = idx.vertex;
			u[1] = mesh.switch_vertex(idx).vertex;
			std::sort(u.begin(), u.end());
			if (u == v) {
				return idx.edge;
			}
			idx = mesh.next_around_face(idx);
		}
	}
	throw std::runtime_error("Edge not found");
}

int find_face(const poly_fem::Mesh3D &mesh, int c, int v1, int v2, int v3) {
	std::array<int, 3> v = {{v1, v2, v3}};
	std::sort(v.begin(), v.end());
	for (int lf = 0; lf < mesh.n_cell_faces(c); ++lf) {
		auto idx = mesh.get_index_from_element(c, lf, 0);
		assert(mesh.n_face_vertices(idx.face) == 3);
		std::array<int, 3> u;
		for (int lv = 0; lv < mesh.n_face_vertices(idx.face); ++lv) {
			u[lv] = idx.vertex;
			idx = mesh.next_around_face(idx);
		}
		std::sort(u.begin(), u.end());
		if (u == v) {
			return idx.face;
		}
	}
	return 0;
}


// -----------------------------------------------------------------------------

std::array<int, 4> linear_tet_local_to_global(const Mesh3D &mesh, int c) {
	// Vertex nodes
	std::array<int, 4> l2g;
	int lv = 0;
	for (int vi : mesh.get_ordered_vertices_from_tet(c)) {
		l2g[lv++] = vi;
	}

	return l2g;
}

// -----------------------------------------------------------------------------

std::array<int, 10> quadr_tet_local_to_global(const Mesh3D &mesh, int c) {
	int edge_offset = mesh.n_vertices();
	int face_offset = edge_offset + mesh.n_edges();
	int cell_offset = face_offset + mesh.n_faces();

	// Vertex nodes
	auto v = linear_tet_local_to_global(mesh, c);

	// Edge nodes
	Eigen::Matrix<int, 6, 1> e;
	Eigen::Matrix<int, 6, 2> ev;
	ev.row(0)  << v[0], v[1];
	ev.row(1)  << v[1], v[2];
	ev.row(2)  << v[2], v[0];

	ev.row(3)  << v[0], v[3];
	ev.row(4)  << v[1], v[3];
	ev.row(5)  << v[2], v[3];

	for (int le = 0; le < e.rows(); ++le) {
		e[le] = find_edge(mesh, c, ev(le, 0), ev(le, 1));
	}

	// Local to global mapping of node indices
	std::array<int, 10> l2g;

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
///             in the middle of each simplex (edge, facet, cell), and node per elements are
///             numbered accordingly.
///
/// @param[in]  mesh               The input mesh
/// @param[in]  discr_order        The discretization order
/// @param[in]  nodes              Lazy evaluator for node ids
/// @param[out] element_nodes_id   List of node indices per element
/// @param[out] local_boundary     Which facet of the element are on the boundary
/// @param[out] poly_face_to_data  Data for faces at the interface with a polyhedra
///
void compute_nodes(
	const poly_fem::Mesh3D &mesh,
	const int discr_order,
	MeshNodes &nodes,
	std::vector<std::vector<int> > &element_nodes_id,
	std::vector<poly_fem::LocalBoundary> &local_boundary,
	std::map<int, poly_fem::InterfaceData> &poly_face_to_data)
{
	// Step 1: Assign global node ids for each quads
	local_boundary.clear();

	element_nodes_id.resize(mesh.n_faces());
	for (int c = 0; c < mesh.n_cells(); ++c) {

		if (discr_order == 1) {
			for (int id : linear_tet_local_to_global(mesh, c)) {
				element_nodes_id[c].push_back(nodes.node_id_from_primitive(id));
			}
		} else {
			for (int id : quadr_tet_local_to_global(mesh, c)) {
				element_nodes_id[c].push_back(nodes.node_id_from_primitive(id));
			}
		}

		auto v = linear_tet_local_to_global(mesh, c);
		Eigen::Matrix<int, 4, 3> fv;
		fv.row(0) << v[0], v[1], v[2];
		fv.row(1) << v[0], v[1], v[3];
		fv.row(2) << v[1], v[2], v[3];
		fv.row(3) << v[2], v[0], v[3];

		LocalBoundary lb(c, BoundaryType::Tri);
		for(long i = 0; i < fv.rows(); ++i)
		{
			int f = find_face(mesh, c, fv(i,0), fv(i,1), fv(i,2));

			if(mesh.is_boundary_face(f)){
				lb.add_boundary_primitive(f, i);
			}
		}

		if(!lb.empty())
			local_boundary.emplace_back(lb);
	}
}


// -----------------------------------------------------------------------------

template<class InputIterator, class T>
int find_index(InputIterator first, InputIterator last, const T& val)
{
	return std::distance(first, std::find(first, last, val));
}

Eigen::RowVector3d linear_tet_local_node_coordinates(int local_index) {
	auto p = linear_tet_local_node[local_index];
	return Eigen::RowVector3d(p[0], p[1], p[2]);
}

Eigen::RowVector3d quadr_tet_local_node_coordinates(int local_index) {
	auto p = quadr_tet_local_node[local_index];
	return Eigen::RowVector3d(p[0], p[1], p[2]) / 2.0;
}

} // anonymous namespace

////////////////////////////////////////////////////////////////////////////////

Eigen::MatrixXd poly_fem::TetBasis3d::tet_local_node_coordinates_from_face(int lf)
{
	Eigen::Matrix<int, 4, 3> fv;
	fv.row(0) << 0, 1, 2;
	fv.row(1) << 0, 1, 3;
	fv.row(2) << 1, 2, 3;
	fv.row(3) << 2, 0, 3;

	Eigen::MatrixXd res(3,3);
	for(int i = 0; i < 3; ++i)
		res.row(i) = linear_tet_local_node_coordinates(fv(lf, i));

	return res;
}

std::array<int, 3> poly_fem::TetBasis3d::linear_tet_face_local_nodes(
	const Mesh3D &mesh, Navigation3D::Index index)
{
	int c = index.element;

	// Local to global mapping of node indices
	auto l2g = linear_tet_local_to_global(mesh, c);

	// Extract requested interface
	std::array<int, 3> result;
	for (int lv = 0, i = 0; lv < 3; ++lv) {
		result[i++] = find_index(l2g.begin(), l2g.end(), index.vertex);
		index = mesh.next_around_face(index);
	}
	return result;
}

Eigen::MatrixXd poly_fem::TetBasis3d::linear_tet_face_local_nodes_coordinates(
	const Mesh3D &mesh, Navigation3D::Index index)
{
	auto idx = linear_tet_face_local_nodes(mesh, index);
	Eigen::MatrixXd res(idx.size(), 3);
	int cnt = 0;
	for (int i : idx) {
		res.row(cnt++) = linear_tet_local_node_coordinates(i);
	}
	return res;
}

// -----------------------------------------------------------------------------

std::array<int, 6> poly_fem::TetBasis3d::quadr_tet_face_local_nodes(
	const Mesh3D &mesh, Navigation3D::Index index)
{
	int c = index.element;

	int edge_offset = mesh.n_vertices();
	int face_offset = edge_offset + mesh.n_edges();

	// Local to global mapping of node indices
	auto l2g = quadr_tet_local_to_global(mesh, c);

	// Extract requested interface
	std::array<int, 6> result;
	for (int lv = 0, i = 0; lv < 3; ++lv) {
		result[i++] = find_index(l2g.begin(), l2g.end(), index.vertex);
		result[i++] = find_index(l2g.begin(), l2g.end(), edge_offset + index.edge);
		index = mesh.next_around_face(index);
	}
	return result;
}

Eigen::MatrixXd poly_fem::TetBasis3d::quadr_tet_face_local_nodes_coordinates(
	const Mesh3D &mesh, Navigation3D::Index index)
{
	auto idx = quadr_tet_face_local_nodes(mesh, index);
	Eigen::MatrixXd res(idx.size(), 3);
	int cnt = 0;
	for (int i : idx) {
		res.row(cnt++) = quadr_tet_local_node_coordinates(i);
	}
	return res;
}

////////////////////////////////////////////////////////////////////////////////

int poly_fem::TetBasis3d::build_bases(
	const Mesh3D &mesh,
	const int quadrature_order,
	const int discr_order,
	std::vector< ElementBases > &bases,
	std::vector< LocalBoundary > &local_boundary,
	std::vector< int > &boundary_nodes,
	std::map<int, InterfaceData> &poly_face_to_data)
{
	assert(mesh.is_volume());

	MeshNodes nodes(mesh, discr_order == 1);
	std::vector<std::vector<int>> element_nodes_id;
	compute_nodes(mesh, discr_order, nodes, element_nodes_id, local_boundary, poly_face_to_data);
	boundary_nodes = nodes.boundary_nodes();

	bases.resize(mesh.n_cells());
	for (int e = 0; e < mesh.n_cells(); ++e) {
		ElementBases &b = bases[e];
		const int n_el_vertices = mesh.n_cell_vertices(e);
		const int n_el_bases = (int) element_nodes_id[e].size();
		b.bases.resize(n_el_bases);

		b.set_quadrature([quadrature_order](Quadrature &quad){
			TetQuadrature tet_quadrature;
			tet_quadrature.get_quadrature(quadrature_order, quad);
		});
		b.bases.resize(n_el_bases);


		b.set_local_node_from_primitive_func([discr_order, e](const int primitive_id, const Mesh &mesh)
			{
				const auto &mesh3d = dynamic_cast<const Mesh3D &>(mesh);
				Navigation3D::Index index;

				for(int lf = 0; lf < mesh3d.n_cell_faces(e); ++lf)
				{
					index = mesh3d.get_index_from_element(e, lf, 0);
					if(index.face == primitive_id)
						break;
				}
				assert(index.face == primitive_id);

				Eigen::VectorXi res;
				if(discr_order == 1)
				{
					const auto indices = linear_tet_face_local_nodes(mesh3d, index);
					res.resize(indices.size());

					for(size_t i = 0; i< indices.size(); ++i)
						res(i)=indices[i];
				}
				else if(discr_order == 2)
				{
					const auto indices = quadr_tet_face_local_nodes(mesh3d, index);
					res.resize(indices.size());

					for(size_t i = 0; i< indices.size(); ++i)
						res(i)=indices[i];
				}
				else
					assert(false);

				return res;
			});

		for (int j = 0; j < n_el_bases; ++j) {
			const int global_index = element_nodes_id[e][j];

			b.bases[j].init(global_index, j, nodes.node_position(global_index));

			if (discr_order == 1) {
				b.bases[j].set_basis([j](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
					{ linear_tet_basis_value(j, uv, val); });
				b.bases[j].set_grad([j](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
					{ linear_tet_basis_grad(j, uv, val); });
			} else if (discr_order == 2) {
				b.bases[j].set_basis([j](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
					{ quadr_tet_basis_value(j, uv, val); });
				b.bases[j].set_grad([j](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
					{ quadr_tet_basis_grad(j, uv, val); });
			} else {
				assert(false);
			}
		}
		
	}

	return nodes.n_nodes();
}
