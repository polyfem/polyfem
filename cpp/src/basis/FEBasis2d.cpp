////////////////////////////////////////////////////////////////////////////////
#include "FEBasis2d.hpp"
#include "TriQuadrature.hpp"
#include "QuadQuadrature.hpp"
#include "auto_bases.hpp"
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









Corner nodes:
v3──────x─────v2
 │      ┆      │
 │      ┆      │
 │      ┆      │
 x┄┄┄┄┄┄x┄┄┄┄┄┄x
 │      ┆      │
 │      ┆      │
 │      ┆      │
v0──────x─────v1

v0 = (0, 0)
v1 = (1, 0)
v2 = (1, 1)
v3 = (0, 1)

Edge nodes:
 x─────e2──────x
 │      ┆      │
 │      ┆      │
 │      ┆      │
e3┄┄┄┄┄┄x┄┄┄┄┄e1
 │      ┆      │
 │      ┆      │
 │      ┆      │
 x─────e0──────x

e0  = (0.5,   0)
e1  = (  1, 0.5)
e2  = (0.5,   1)
e3  = (  0, 0.5)

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

	constexpr std::array<std::array<int, 2>, 8> linear_tri_local_node = {{
		{{0, 0}}, // v0  = (0, 0)
		{{1, 0}}, // v1  = (1, 0)
		{{0, 1}}, // v2  = (0, 1)
	}};

// -----------------------------------------------------------------------------

	constexpr std::array<std::array<int, 2>, 6> quadr_tri_local_node = {{
		{{0, 0}}, // v0  = (0, 0)
		{{2, 0}}, // v1  = (1, 0)
		{{0, 2}}, // v2  = (0, 1)
		{{1, 0}}, // e0  = (0.5,   0)
		{{1, 1}}, // e1  = (0.5, 0.5)
		{{0, 1}}, // e3  = (  0, 0.5)
	}};



// -----------------------------------------------------------------------------

	constexpr std::array<std::array<int, 2>, 4> linear_quad_local_node = {{
		{{0, 0}}, // v0  = (0, 0)
		{{1, 0}}, // v1  = (1, 0)
		{{1, 1}}, // v2  = (1, 1)
		{{0, 1}}, // v3  = (0, 1)
	}};

	void linear_quad_basis_value(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) {
		auto u=uv.col(0).array();
		auto v=uv.col(1).array();

		std::array<int, 2> idx = linear_quad_local_node[local_index];
		val = alpha(idx[0], u).array() * alpha(idx[1], v).array();
	}

	void linear_quad_basis_grad(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) {
		auto u=uv.col(0).array();
		auto v=uv.col(1).array();

		std::array<int, 2> idx = linear_quad_local_node[local_index];

		val.resize(uv.rows(), 2);
		val.col(0) = dalpha(idx[0], u).array() * alpha(idx[1], v).array();
		val.col(1) = alpha(idx[0], u).array() * dalpha(idx[1], v).array();
	}

// -----------------------------------------------------------------------------

	constexpr std::array<std::array<int, 2>, 9> quadr_quad_local_node = {{
		{{0, 0}}, // v0  = (  0,   0)
		{{2, 0}}, // v1  = (  1,   0)
		{{2, 2}}, // v2  = (  1,   1)
		{{0, 2}}, // v3  = (  0,   1)
		{{1, 0}}, // e0  = (0.5,   0)
		{{2, 1}}, // e1  = (  1, 0.5)
		{{1, 2}}, // e2  = (0.5,   1)
		{{0, 1}}, // e3  = (  0, 0.5)
		{{1, 1}}, // f0  = (0.5, 0.5)
	}};

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
				if(idx.vertex != v1)
					idx = mesh.switch_vertex(idx);
				assert(idx.vertex == v1);
				return idx;
			}
			idx = mesh.next_around_face(idx);
		}
		throw std::runtime_error("Edge not found");
	}

// -----------------------------------------------------------------------------


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
		const Eigen::VectorXi &discr_orders,
		MeshNodes &nodes,
		std::vector<std::vector<int> > &element_nodes_id,
		std::vector<poly_fem::LocalBoundary> &local_boundary,
		std::map<int, poly_fem::InterfaceData> &poly_edge_to_data)
	{
		// Step 1: Assign global node ids for each quads
		local_boundary.clear();

		element_nodes_id.resize(mesh.n_faces());
		
		for (int f = 0; f < mesh.n_faces(); ++f) {
			const int discr_order = discr_orders(f);
			if(mesh.is_cube(f))
			{
				if (discr_order == 1) {
					for (int id : poly_fem::FEBasis2d::linear_quad_local_to_global(mesh, f)) {
						element_nodes_id[f].push_back(nodes.node_id_from_primitive(id));
					}
				} else if (discr_order == 2) {//TODO
					const auto tmp = poly_fem::FEBasis2d::quadr_quad_local_to_global(mesh, f);
					for (int id : tmp) {
						if(id < 0)
							element_nodes_id[f].push_back(id);
						else
							element_nodes_id[f].push_back(nodes.node_id_from_primitive(id));
					}
				}
				else
				{
					assert(false);
				}

				// List of edges around the quad
				std::array<int, 4> e;
				{
					auto l2g = poly_fem::FEBasis2d::quadr_quad_local_to_global(mesh, f);
					for (int le = 0; le < 4; ++le) {
						e[le] = l2g[4+le] - mesh.n_vertices();
					}
				}

				LocalBoundary lb(f, BoundaryType::QuadLine);

				for(int i = 0; i < e.size(); ++i)
				{
					if (mesh.is_boundary_edge(e[i])){
						lb.add_boundary_primitive(e[i], i);
					}
				}

				if(!lb.empty())
					local_boundary.emplace_back(lb);
			} else if(mesh.is_simplex(f)) {
				element_nodes_id[f] = poly_fem::FEBasis2d::tri_local_to_global(discr_order, mesh, f, discr_orders, nodes);

				// List of edges around the tri
				std::array<int, 3> e;
				{
					auto l2g = poly_fem::FEBasis2d::quadr_tri_local_to_global(mesh, f);
					for (int le = 0; le < 3; ++le) {
						e[le] = l2g[3+le] - mesh.n_vertices();
					}
				}

				LocalBoundary lb(f, BoundaryType::TriLine);

				for(int i = 0; i < e.size(); ++i)
				{
					if (mesh.is_boundary_edge(e[i])){
						lb.add_boundary_primitive(e[i], i);
					}
				}

				if(!lb.empty())
					local_boundary.emplace_back(lb);
			}
		}

		// Step 2: Iterate over edges of polygons and compute interface weights
		// TODO p-ref
		for (int f = 0; f < mesh.n_faces(); ++f) {
			const int discr_order = discr_orders(f);
			// Skip non-polytopes
			if (!mesh.is_polytope(f)) { continue; }

			auto index = mesh.get_index_from_face(f, 0);
			for (int lv = 0; lv < mesh.n_face_vertices(f); ++lv) {
				auto index2 = mesh.switch_face(index);
				if (index2.face >= 0) {
					// Opposite face is a quad, we need to set interface data
					int f2 = index2.face;
					if(mesh.is_cube(f2))
					{
						poly_fem::InterfaceData data;
						if (discr_order == 2) {
							auto abc = poly_fem::FEBasis2d::quadr_quad_edge_local_nodes(mesh, index2);
							data.local_indices.assign(abc.begin(), abc.end());
						} else if(discr_order == 1) {
							auto ab = poly_fem::FEBasis2d::linear_quad_edge_local_nodes(mesh, index2);
							data.local_indices.assign(ab.begin(), ab.end());
						}
						else
							assert(false);

						poly_edge_to_data[index2.edge] = data;
					} else if(mesh.is_simplex(f2)) {
						//TODO, might not work!!!!!
						poly_fem::InterfaceData data;
						auto abc = poly_fem::FEBasis2d::tri_edge_local_nodes(discr_order, mesh, index);
						data.local_indices.assign(abc.size(), abc.data()[0]);
						// if (discr_order == 2) {
						// 	auto abc = poly_fem::FEBasis2d::quadr_tri_edge_local_nodes(mesh, index2);
						// 	data.local_indices.assign(abc.begin(), abc.end());
						// } else  if(discr_order == 1) {
						// 	auto ab = poly_fem::FEBasis2d::linear_tri_edge_local_nodes(mesh, index2);
						// 	data.local_indices.assign(ab.begin(), ab.end());
						// }
						// else
						// 	assert(false);
						poly_edge_to_data[index2.edge] = data;
					}
				}
				index = mesh.next_around_face(index);
			}
		}
	}

/*
Axes:
 y
 |
 o──x

Boundaries:
X axis: left/right
Y axis: bottom/top

Edge nodes:
 x─────e2──────x
 │      ┆      │
 │      ┆      │
 │      ┆      │
e3┄┄┄┄┄┄x┄┄┄┄┄e1
 │      ┆      │
 │      ┆      │
 │      ┆      │
 x─────e0──────x
*/

// -----------------------------------------------------------------------------

template<class InputIterator, class T>
	int find_index(InputIterator first, InputIterator last, const T& val)
	{
		return std::distance(first, std::find(first, last, val));
	}

} // anonymous namespace

void poly_fem::FEBasis2d::linear_tri_basis_value(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) {
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

void poly_fem::FEBasis2d::linear_tri_basis_grad(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) {
	val.resize(uv.rows(), uv.cols());
	switch(local_index)
	{
		case 0: val.col(0).setConstant(-1); val.col(1).setConstant(-1); break;
		case 1: val.col(0).setConstant( 1); val.col(1).setConstant( 0); break;
		case 2: val.col(0).setConstant( 0); val.col(1).setConstant( 1); break;
		default: assert(false);
	}
}
void poly_fem::FEBasis2d::quadr_tri_basis_value(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
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

void poly_fem::FEBasis2d::quadr_tri_basis_grad(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
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

Eigen::RowVector2d poly_fem::FEBasis2d::linear_tri_local_node_coordinates(int local_index) {
	auto p = linear_tri_local_node[local_index];
	return Eigen::RowVector2d(p[0], p[1]);
}

Eigen::RowVector2d poly_fem::FEBasis2d::quadr_tri_local_node_coordinates(int local_index) {
	auto p = quadr_tri_local_node[local_index];
	return Eigen::RowVector2d(p[0], p[1]) / 2.0;
}


std::array<int, 3> poly_fem::FEBasis2d::linear_tri_local_to_global(const Mesh2D &mesh, int f) {
	// assert(mesh.is_cube(f));

	// Vertex nodes
	std::array<int, 3> l2g;
	for (int lv = 0; lv < 3; ++lv) {
		l2g[lv] = mesh.face_vertex(f, lv);
	}

	return l2g;
}

std::vector<int> poly_fem::FEBasis2d::tri_local_to_global(const int p, const Mesh2D &mesh, int f, const Eigen::VectorXi &discr_order, poly_fem::MeshNodes &nodes) {
	int edge_offset = mesh.n_vertices();
	int face_offset = edge_offset + mesh.n_edges();

	const int n_edge_nodes = (p-1)*3;
	const int nn = p > 2 ? (p - 2) : 0;
	const int n_face_nodes = nn * (nn + 1) / 2;

	std::vector<int> res; res.reserve(3 + n_edge_nodes + n_face_nodes);

	// Vertex nodes
	auto v = linear_tri_local_to_global(mesh, f);

	// Edge nodes
	Eigen::Matrix<Navigation::Index, 3, 1> e;
	Eigen::Matrix<int, 3, 2> ev;
	ev.row(0) << v[0], v[1];
	ev.row(1) << v[1], v[2];
	ev.row(2) << v[2], v[0];
	for (int le = 0; le < e.rows(); ++le) {
		const auto index = find_edge(mesh, f, ev(le, 0), ev(le, 1));
		e[le] = index;
	}

	for (size_t lv = 0; lv < v.size(); ++lv) {
		const auto index = e[lv];
		const auto other_face = mesh.switch_face(index).face;

		if(discr_order.size() > 0 && other_face >= 0 && discr_order(f) >  discr_order(other_face))
			res.push_back(-lv - 1);
		else
			res.push_back(nodes.node_id_from_primitive(v[lv]));
	}

	for (int le = 0; le < e.rows(); ++le) {
		const auto index = e[le];
		const auto other_face = mesh.switch_face(index).face;

		bool skip_other = discr_order.size() > 0 && other_face >= 0 && discr_order(f) >  discr_order(other_face);

		if(discr_order.size() > 0 && other_face >= 0 && discr_order(f) >  discr_order(other_face))
		{
			for(int tmp = 0; tmp < p - 1; ++tmp)
				res.push_back(-le - 1);
		}
		else
		{
			auto node_ids = nodes.node_ids_from_edge(index, p - 1);
			res.insert(res.end(), node_ids.begin(), node_ids.end());
		}
	}

	if (n_face_nodes > 0) {
		const auto index = e[0];

		auto node_ids = nodes.node_ids_from_face(index, p - 2);
		res.insert(res.end(), node_ids.begin(), node_ids.end());
	}

	assert(res.size() == 3 + n_edge_nodes + n_face_nodes);
	return res;
}

std::array<int, 6> poly_fem::FEBasis2d::quadr_tri_local_to_global(const Mesh2D &mesh, int f, const Eigen::VectorXi &discr_order) {
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
		const auto index = find_edge(mesh, f, ev(le, 0), ev(le, 1));
		const auto other_face = mesh.switch_face(index).face;
		if(discr_order.size() > 0 && other_face >= 0 && discr_order(f) >  discr_order(other_face))
			e[le] = -le - 1;
		else
			e[le] = index.edge;
	}

	// Local to global mapping of node indices
	std::array<int, 6> l2g;

	// Assign global ids to local nodes
	{
		int i = 0;
		for (size_t lv = 0; lv < v.size(); ++lv) {
			l2g[i++] = e[lv] >= 0 ? v[lv] : e[lv];
		}
		for (int le = 0; le < e.rows(); ++le) {
			l2g[i++] = e[le] >= 0 ? (edge_offset + e[le]) : e[le];
		}
	}

	return l2g;
}



Eigen::MatrixXd poly_fem::FEBasis2d::tri_local_node_coordinates_from_edge(int le)
{
	Eigen::MatrixXd res(2,2);
	res.row(0) = linear_tri_local_node_coordinates(le);
	res.row(1) = linear_tri_local_node_coordinates((le+1)%3);

	return res;
}

std::array<int, 2> poly_fem::FEBasis2d::linear_tri_edge_local_nodes(
	const Mesh2D &mesh, Navigation::Index index)
{
	int f = index.face;
	assert(mesh.is_simplex(f));

	// Local to global mapping of node indices
	auto l2g = linear_tri_local_to_global(mesh, f);

	// Extract requested interface
	std::array<int, 2> result;
	result[0] = find_index(l2g.begin(), l2g.end(), index.vertex);
	result[1] = find_index(l2g.begin(), l2g.end(), mesh.switch_vertex(index).vertex);
	return result;
}

Eigen::MatrixXd poly_fem::FEBasis2d::linear_tri_edge_local_nodes_coordinates(
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

Eigen::VectorXi poly_fem::FEBasis2d::tri_edge_local_nodes(const int p, const Mesh2D &mesh, Navigation::Index index)
{
	int f = index.face;
	assert(mesh.is_simplex(f));

	// Local to global mapping of node indices
	auto l2g = linear_tri_local_to_global(mesh, f);

	// Extract requested interface
	Eigen::VectorXi result(2 + (p - 1));
	result[0] = find_index(l2g.begin(), l2g.end(), index.vertex);
	result[result.size()-1] = find_index(l2g.begin(), l2g.end(), mesh.switch_vertex(index).vertex);

	if((result[0] == 0 && result[result.size()-1] == 1) || (result[0] == 1 && result[result.size()-1] == 2) || (result[0] == 2 && result[result.size()-1] == 0))
	{
		for(int i = 0; i < p - 1; ++i)
		{
			result[i+1] = 3 + result[0]*(p-1) + i;
		}
	}
	else
	{
		for(int i = 0; i < p - 1; ++i)
		{
			result[i+1] = 3 + (result[0] + (result[0] == 0 ? 3 : 0))*(p-1) - i - 1;
		}
	}

	return result;
}

std::array<int, 3> poly_fem::FEBasis2d::quadr_tri_edge_local_nodes(
	const Mesh2D &mesh, Navigation::Index index)
{
	int f = index.face;
	assert(mesh.is_simplex(f));
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

Eigen::MatrixXd poly_fem::FEBasis2d::quadr_tri_edge_local_nodes_coordinates(
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
// -----------------------------------------------------------------------------

Eigen::RowVector2d poly_fem::FEBasis2d::linear_quad_local_node_coordinates(int local_index) {
	assert(local_index >= 0 && local_index < 4);
	auto p = linear_quad_local_node[local_index];
	return Eigen::RowVector2d(p[0], p[1]);
}

Eigen::RowVector2d poly_fem::FEBasis2d::quadr_quad_local_node_coordinates(int local_index) {
	assert(local_index >= 0 && local_index < 9);
	auto p = quadr_quad_local_node[local_index];
	return Eigen::RowVector2d(p[0], p[1]) / 2.0;
}

std::array<int, 4> poly_fem::FEBasis2d::linear_quad_local_to_global(const Mesh2D &mesh, int f) {
	assert(mesh.is_cube(f));

	// Vertex nodes
	std::array<int, 4> l2g;
	for (int lv = 0; lv < 4; ++lv) {
		l2g[lv] = mesh.face_vertex(f, lv);
	}

	return l2g;
}

// -----------------------------------------------------------------------------

std::array<int, 9> poly_fem::FEBasis2d::quadr_quad_local_to_global(const Mesh2D &mesh, int f) {
	assert(mesh.is_cube(f));

	int edge_offset = mesh.n_vertices();
	int face_offset = edge_offset + mesh.n_edges();

	// Vertex nodes
	auto v = linear_quad_local_to_global(mesh, f);

	// Edge nodes
	Eigen::Matrix<int, 4, 1> e;
	Eigen::Matrix<int, 4, 2> ev;
	ev.row(0) << v[0], v[1];
	ev.row(1) << v[1], v[2];
	ev.row(2) << v[2], v[3];
	ev.row(3) << v[3], v[0];
	for (int le = 0; le < e.rows(); ++le) {
		e[le] = find_edge(mesh, f, ev(le, 0), ev(le, 1)).edge;
	}

	// Local to global mapping of node indices
	std::array<int, 9> l2g;

	// Assign global ids to local nodes
	{
		int i = 0;
		for (size_t lv = 0; lv < v.size(); ++lv) {
			l2g[i++] = v[lv];
		}
		for (int le = 0; le < e.rows(); ++le) {
			l2g[i++] = edge_offset + e[le];
		}
		l2g[i++] = face_offset + f;
	}

	return l2g;
}

Eigen::MatrixXd poly_fem::FEBasis2d::quad_local_node_coordinates_from_edge(int le)
{
	Eigen::MatrixXd res(2,2);
	res.row(0) = linear_quad_local_node_coordinates(le);
	res.row(1) = linear_quad_local_node_coordinates((le+1)%4);

	return res;
}

////////////////////////////////////////////////////////////////////////////////

std::array<int, 2> poly_fem::FEBasis2d::linear_quad_edge_local_nodes(
	const Mesh2D &mesh, Navigation::Index index)
{
	int f = index.face;
	assert(mesh.is_cube(f));

	// Local to global mapping of node indices
	auto l2g = linear_quad_local_to_global(mesh, f);

	// Extract requested interface
	std::array<int, 2> result;
	result[0] = find_index(l2g.begin(), l2g.end(), index.vertex);
	result[1] = find_index(l2g.begin(), l2g.end(), mesh.switch_vertex(index).vertex);
	return result;
}

Eigen::MatrixXd poly_fem::FEBasis2d::linear_quad_edge_local_nodes_coordinates(
	const Mesh2D &mesh, Navigation::Index index)
{
	auto idx = linear_quad_edge_local_nodes(mesh, index);
	Eigen::MatrixXd res(idx.size(), 2);
	int cnt = 0;
	for (int i : idx) {
		res.row(cnt++) = linear_quad_local_node_coordinates(i);
	}
	return res;
}

// -----------------------------------------------------------------------------

std::array<int, 3> poly_fem::FEBasis2d::quadr_quad_edge_local_nodes(
	const Mesh2D &mesh, Navigation::Index index)
{
	int f = index.face;
	assert(mesh.is_cube(f));
	int e_offset = mesh.n_vertices();

	// Local to global mapping of node indices
	auto l2g = quadr_quad_local_to_global(mesh, f);

	// Extract requested interface
	std::array<int, 3> result;
	result[0] = find_index(l2g.begin(), l2g.end(), index.vertex);
	result[1] = find_index(l2g.begin(), l2g.end(), e_offset + index.edge);
	result[2] = find_index(l2g.begin(), l2g.end(), mesh.switch_vertex(index).vertex);
	return result;
}

Eigen::MatrixXd poly_fem::FEBasis2d::quadr_quad_edge_local_nodes_coordinates(
	const Mesh2D &mesh, Navigation::Index index)
{
	auto idx = quadr_quad_edge_local_nodes(mesh, index);
	Eigen::MatrixXd res(idx.size(), 2);
	int cnt = 0;
	for (int i : idx) {
		res.row(cnt++) = quadr_quad_local_node_coordinates(i);
	}
	return res;
}

////////////////////////////////////////////////////////////////////////////////

void poly_fem::FEBasis2d::quadr_quad_basis_value(
	const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
{
	auto u=uv.col(0).array();
	auto v=uv.col(1).array();

	std::array<int, 2> idx = quadr_quad_local_node[local_index];
	val = theta(idx[0], u).array() * theta(idx[1], v).array();
}

void poly_fem::FEBasis2d::quadr_quad_basis_grad(
	const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
{
	auto u=uv.col(0).array();
	auto v=uv.col(1).array();

	std::array<int, 2> idx = quadr_quad_local_node[local_index];

	val.resize(uv.rows(), 2);
	val.col(0) = dtheta(idx[0], u).array() * theta(idx[1], v).array();
	val.col(1) = theta(idx[0], u).array() * dtheta(idx[1], v).array();
}

// -----------------------------------------------------------------------------

int poly_fem::FEBasis2d::build_bases(
	const Mesh2D &mesh,
	const int quadrature_order,
	const int discr_order,
	std::vector<ElementBases> &bases,
	std::vector<LocalBoundary> &local_boundary,
	std::map<int, InterfaceData> &poly_edge_to_data)
{

	Eigen::VectorXi discr_orders(mesh.n_faces());
	discr_orders.setConstant(discr_order);

	return build_bases(mesh, quadrature_order, discr_orders, bases, local_boundary, poly_edge_to_data);
}

int poly_fem::FEBasis2d::build_bases(
	const Mesh2D &mesh,
	const int quadrature_order,
	const Eigen::VectorXi &discr_orders,
	std::vector<ElementBases> &bases,
	std::vector<LocalBoundary> &local_boundary,
	std::map<int, InterfaceData> &poly_edge_to_data)
{
	assert(!mesh.is_volume());
	assert(discr_orders.size() == mesh.n_faces());

	const int max_p = discr_orders.maxCoeff();
	const int nn = max_p > 2 ? (max_p - 2) : 0;
	const int n_face_nodes = std::max(nn * (nn + 1) / 2, (max_p-1)*(max_p-1));

	//TODO works only for P
	MeshNodes nodes(mesh, max_p - 1, n_face_nodes);
	std::vector<std::vector<int>> element_nodes_id;
	compute_nodes(mesh, discr_orders, nodes, element_nodes_id, local_boundary, poly_edge_to_data);
	// boundary_nodes = nodes.boundary_nodes();


	bases.resize(mesh.n_faces());
	std::vector<int> interface_elements; interface_elements.reserve(mesh.n_faces());

	for (int e = 0; e < mesh.n_faces(); ++e) {
		ElementBases &b = bases[e];
		const int discr_order = discr_orders(e);
		const int n_el_bases = element_nodes_id[e].size();
		b.bases.resize(n_el_bases);

		bool skip_interface_element = false;

		for (int j = 0; j < n_el_bases; ++j){
			const int global_index = element_nodes_id[e][j];
			if(global_index < 0)
			{
				skip_interface_element = true;
				break;
			}
		}

		if(skip_interface_element){
			interface_elements.push_back(e);
		}

		if (mesh.is_cube(e)) {
			b.set_quadrature([quadrature_order](Quadrature &quad){
				QuadQuadrature quad_quadrature;
				quad_quadrature.get_quadrature(quadrature_order, quad);
			});
			// quad_quadrature.get_quadrature(quadrature_order, b.quadrature);

			b.set_local_node_from_primitive_func([discr_order, e](const int primitive_id, const Mesh &mesh)
			{
				const auto &mesh2d = dynamic_cast<const Mesh2D &>(mesh);
				auto index = mesh2d.get_index_from_face(e);

				for(int le = 0; le < mesh2d.n_face_vertices(e); ++le)
				{
					if(index.edge == primitive_id)
						break;
					index = mesh2d.next_around_face(index);
				}
				assert(index.edge == primitive_id);

				Eigen::VectorXi res;
				if(discr_order == 1)
				{
					const auto indices = linear_quad_edge_local_nodes(mesh2d, index);
					res.resize(indices.size());

					for(size_t i = 0; i< indices.size(); ++i)
						res(i)=indices[i];
				}
				else if(discr_order == 2)
				{
					const auto indices = quadr_quad_edge_local_nodes(mesh2d, index);
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

				// if(!skip_interface_element)
				b.bases[j].init(global_index, j, nodes.node_position(global_index));


				if (discr_order == 1) {
					b.bases[j].set_basis([j](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
						{ linear_quad_basis_value(j, uv, val); });
					b.bases[j].set_grad([j](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
						{ linear_quad_basis_grad(j, uv, val); });
				} else if (discr_order == 2) {
					b.bases[j].set_basis([j](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
						{ quadr_quad_basis_value(j, uv, val); });
					b.bases[j].set_grad([j](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
						{ quadr_quad_basis_grad(j, uv, val); });
				} else {
					assert(false);
				}
			}
		} else if(mesh.is_simplex(e))
		{
			const int real_order = std::max(quadrature_order, (discr_order - 1) * (discr_order - 1));
			b.set_quadrature([real_order](Quadrature &quad){
				TriQuadrature tri_quadrature;
				tri_quadrature.get_quadrature(real_order, quad);
			});

			b.set_local_node_from_primitive_func([discr_order, nodes, e](const int primitive_id, const Mesh &mesh)
			{
				const auto &mesh2d = dynamic_cast<const Mesh2D &>(mesh);
				auto index = mesh2d.get_index_from_face(e);

				for(int le = 0; le < mesh2d.n_face_vertices(e); ++le)
				{
					if(index.edge == primitive_id)
						break;
					index = mesh2d.next_around_face(index);
				}
				assert(index.edge == primitive_id);
				return tri_edge_local_nodes(discr_order, mesh2d, index);

				// Eigen::VectorXi res;
				// if(discr_order == 1)
				// {
				// 	const auto indices = linear_tri_edge_local_nodes(mesh2d, index);
				// 	res.resize(indices.size());

				// 	for(size_t i = 0; i< indices.size(); ++i)
				// 		res(i)=indices[i];
				// }
				// else if(discr_order == 2)
				// {
				// 	const auto indices = quadr_tri_edge_local_nodes(mesh2d, index);
				// 	res.resize(indices.size());

				// 	for(size_t i = 0; i< indices.size(); ++i)
				// 		res(i)=indices[i];
				// }
				// else
				// 	assert(false);

				// return res;
			});



			for (int j = 0; j < n_el_bases; ++j) {
				const int global_index = element_nodes_id[e][j];

				if(!skip_interface_element){
					b.bases[j].init(global_index, j, nodes.node_position(global_index));
				}

				b.bases[j].set_basis([discr_order, j](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { poly_fem::autogen::p_basis_value     (discr_order, j, uv, val); });
				b.bases[j].set_grad ([discr_order, j](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { poly_fem::autogen::p_grad_basis_value(discr_order, j, uv, val); });
			}
		}
		else {
			// Polygon bases are built later on
		}
	}


	for (int e : interface_elements) {
		ElementBases &b = bases[e];
		const int discr_order = discr_orders(e);
		const int n_el_bases = element_nodes_id[e].size();
		assert(discr_order > 1);

		if (mesh.is_cube(e)) {
			assert(false); //TODO
			// for (int j = 0; j < n_el_bases; ++j) {
			// 	const int global_index = element_nodes_id[e][j];

			// 	b.bases[j].init(global_index, j, nodes.node_position(global_index));

			// 	if (discr_order == 2) {
			// 		b.bases[j].set_basis([j](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
			// 			{ quadr_quad_basis_value(j, uv, val); });
			// 		b.bases[j].set_grad([j](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
			// 			{ quadr_quad_basis_grad(j, uv, val); });
			// 	} else {
			// 		assert(false);
			// 	}
			// }
		} else if(mesh.is_simplex(e))
		{
			for (int j = 0; j < n_el_bases; ++j) {
				const int global_index = element_nodes_id[e][j];

				if(global_index >= 0)
					b.bases[j].init(global_index, j, nodes.node_position(global_index));
				else
				{
					const auto le = -(global_index+1);

					auto v = linear_tri_local_to_global(mesh, e);
					Eigen::Matrix<int, 3, 2> ev;
					ev.row(0) << v[0], v[1];
					ev.row(1) << v[1], v[2];
					ev.row(2) << v[2], v[0];

					const auto index = mesh.switch_face(find_edge(mesh, e, ev(le, 0), ev(le, 1)));
					const auto other_face = index.face;


					Eigen::RowVector2d node_position;
					assert(discr_order > 1);

					auto indices = tri_edge_local_nodes(discr_order, mesh, index);
					Eigen::MatrixXd lnodes; autogen::p_nodes(discr_order, lnodes);

					if( j < 3)
						node_position = lnodes.row(indices(0));
					else if( j < 3 + 3*(discr_order-1))
						node_position = lnodes.row(indices(((j-3) % (discr_order-1)) + 1));
					else
						assert(false);



					// std::cout<<indices.transpose()<<std::endl;
					// auto asd = quadr_tri_edge_local_nodes(mesh, index);
					// std::cout<<asd[0]<<" "<<asd[1]<<" "<<asd[2]<<std::endl;


					// std::cout<<"\n"<<lnodes<<"\nnewp\n"<<node_position<<"\n"<<std::endl;
					// const auto param_p = quadr_tri_edge_local_nodes_coordinates(mesh, index);

					// if( j < 3)
					// 	node_position = param_p.row(0);
					// else if( j < 3 + 3*(discr_order-1)){
					// 	node_position = param_p.row( (j-3) % (discr_order-1) + 1);
					// }
					// else
					// 	assert(false);
					// std::cout<<node_position<<"\n\n----\n"<<std::endl;
					

					const auto &other_bases = bases[other_face];
					Eigen::MatrixXd w;
					other_bases.evaluate_bases(node_position, w);

					for(long i = 0; i < w.size(); ++i)
					{
						if(std::abs(w(i))<1e-8)
							continue;

						assert(other_bases.bases[i].global().size() == 1);
						const auto &other_global = other_bases.bases[i].global().front();
						// std::cout<<"e "<<e<<" " <<j << " gid "<<other_global.index<<std::endl;
						b.bases[j].global().emplace_back(other_global.index, other_global.node, w(i));
					}
				}
			}
		}
		else {
			// Polygon bases are built later on
		}
	}

	return nodes.n_nodes();
}
