
////////////////////////////////////////////////////////////////////////////////
#include <polyfem/FEBasis3d.hpp>
#include <polyfem/MeshNodes.hpp>
#include <polyfem/TetQuadrature.hpp>
#include <polyfem/HexQuadrature.hpp>

#include <polyfem/auto_p_bases.hpp>
#include <polyfem/auto_q_bases.hpp>

#include <cassert>
#include <array>
////////////////////////////////////////////////////////////////////////////////

using namespace polyfem;

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

v3──x──v2
 │⋱   ╱  ╲
 x  ⋱╱    ╲
 │  x  x   x
 │ ╱     ⋱  ╲
 │╱         ⋱╲
v0─────x──────v1

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

template<class InputIterator, class T>
int find_index(InputIterator first, InputIterator last, const T& val)
{
	return std::distance(first, std::find(first, last, val));
}

Navigation3D::Index find_quad_face(const polyfem::Mesh3D &mesh, int c, int v1, int v2, int v3, int v4) {
	std::array<int, 4> v = {{v1, v2, v3, v4}};
	std::sort(v.begin(), v.end());
	for (int lf = 0; lf < mesh.n_cell_faces(c); ++lf) {
		auto idx = mesh.get_index_from_element(c, lf, 0);
		assert(mesh.n_face_vertices(idx.face) == 4);
		std::array<int, 4> u;
		for (int lv = 0; lv < mesh.n_face_vertices(idx.face); ++lv) {
			u[lv] = idx.vertex;
			idx = mesh.next_around_face(idx);
		}
		std::sort(u.begin(), u.end());
		if (u == v) {
			return idx;
		}
	}
	assert(false);
	return Navigation3D::Index();
}

std::array<int, 4> tet_vertices_local_to_global(const Mesh3D &mesh, int c) {
	// Vertex nodes
	assert(mesh.is_simplex(c));
	std::array<int, 4> l2g;
	int lv = 0;
	for (int vi : mesh.get_ordered_vertices_from_tet(c)) {
		l2g[lv++] = vi;
	}

	return l2g;
}

std::array<int, 8> hex_vertices_local_to_global(const Mesh3D &mesh, int c) {
	assert(mesh.is_cube(c));

	// Vertex nodes
	std::array<int, 8> l2g;
	int lv = 0;
	for (int vi : mesh.get_ordered_vertices_from_hex(c)) {
		l2g[lv++] = vi;
	}

	return l2g;
}


void tet_local_to_global(const int p, const Mesh3D &mesh, int c, const Eigen::VectorXi &discr_order, std::vector<int> &res, polyfem::MeshNodes &nodes)
{
	const int n_edge_nodes = p > 1 ? ((p-1)*6) : 0;
	const int nn = p > 2 ? (p - 2) : 0;
	const int n_loc_f = (nn * (nn + 1) / 2);
	const int n_face_nodes = n_loc_f * 4;
	const int n_cell_nodes = p == 4 ? 1 : 0; //P5 not supported


	if(p == 0)
	{
		res.push_back(nodes.node_id_from_cell(c));
		return;
	}


	// std::vector<int> res;
	res.reserve(4 + n_edge_nodes + n_face_nodes + n_cell_nodes);

	// Edge nodes
	Eigen::Matrix<Navigation3D::Index, 4, 1> f;

	auto v = tet_vertices_local_to_global(mesh, c);
	Eigen::Matrix<int, 4, 3> fv;
	fv.row(0) << v[0], v[1], v[2];
	fv.row(1) << v[0], v[1], v[3];
	fv.row(2) << v[1], v[2], v[3];
	fv.row(3) << v[2], v[0], v[3];

	for(long lf = 0; lf < fv.rows(); ++lf)
	{
		const auto index =  mesh.get_index_from_element_face(c, fv(lf,0), fv(lf,1), fv(lf,2));
		f[lf] = index;
	}

	Eigen::Matrix<Navigation3D::Index, 6, 1> e;
	Eigen::Matrix<int, 6, 2> ev;
	ev.row(0)  << v[0], v[1];
	ev.row(1)  << v[1], v[2];
	ev.row(2)  << v[2], v[0];

	ev.row(3)  << v[0], v[3];
	ev.row(4)  << v[1], v[3];
	ev.row(5)  << v[2], v[3];

	for (int le = 0; le < e.rows(); ++le) {
		// const auto index =  find_edge(mesh, c, ev(le, 0), ev(le, 1));
		const auto index = mesh.get_index_from_element_edge(c, ev(le, 0), ev(le, 1));
		e[le] = index;
	}


	//vertices
	for (size_t lv = 0; lv < v.size(); ++lv) {
		res.push_back(nodes.node_id_from_primitive(v[lv]));
	}

	//Edges
	for (int le = 0; le < e.rows(); ++le) {
		const auto index = e[le];
		auto neighs = mesh.edge_neighs(index.edge);
		int min_p = discr_order.size() > 0 ? discr_order(c) : 0;

		for(auto cid : neighs)
		{
			min_p = std::min(min_p, discr_order.size() > 0 ? discr_order(cid) : 0);
		}

		if(discr_order.size() > 0 && discr_order(c) >  min_p)
		{
			for(int tmp = 0; tmp < p - 1; ++tmp)
				res.push_back(-le - 10);
		}
		else
		{
			auto node_ids = nodes.node_ids_from_edge(index, p - 1);
			res.insert(res.end(), node_ids.begin(), node_ids.end());
		}
	}


	//faces
	for (int lf = 0; lf < f.rows(); ++lf) {
		const auto index = f[lf];
		const auto other_cell = mesh.switch_element(index).element;

		const bool skip_other = discr_order.size() > 0 && other_cell >= 0 && discr_order(c) >  discr_order(other_cell);

		if(skip_other)
		{
			for(int tmp = 0; tmp < n_loc_f; ++tmp)
				res.push_back(-lf - 1);
		}
		else
		{
			auto node_ids = nodes.node_ids_from_face(index, p - 2);
			res.insert(res.end(), node_ids.begin(), node_ids.end());
		}
	}

	//cells
	if (n_cell_nodes > 0) {
		const auto index = f[0];

		auto node_ids = nodes.node_ids_from_cell(index, p - 3);
		res.insert(res.end(), node_ids.begin(), node_ids.end());
	}

	assert(res.size() == size_t(4 + n_edge_nodes + n_face_nodes + n_cell_nodes));
}

void hex_local_to_global(const int q, const Mesh3D &mesh, int c, const Eigen::VectorXi &discr_order, std::vector<int> &res, polyfem::MeshNodes &nodes)
{
	assert(mesh.is_cube(c));

	const int n_edge_nodes = ((q-1)*12);
	const int nn = (q - 1);
	const int n_loc_f = nn*nn;
	const int n_face_nodes = n_loc_f * 6;
	const int n_cell_nodes = nn * nn * nn;


	if(q == 0)
	{
		res.push_back(nodes.node_id_from_cell(c));
		return;
	}


	// std::vector<int> res;
	res.reserve(8 + n_edge_nodes + n_face_nodes + n_cell_nodes);

	// Vertex nodes
	auto v = hex_vertices_local_to_global(mesh, c);

	// Edge nodes
	Eigen::Matrix<Navigation3D::Index, 12, 1> e;
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
		// e[le] = find_edge(mesh, c, ev(le, 0), ev(le, 1)).edge;
		e[le] = mesh.get_index_from_element_edge(c, ev(le, 0), ev(le, 1));
	}

	// Face nodes
	Eigen::Matrix<Navigation3D::Index, 6, 1> f;
	Eigen::Matrix<int, 6, 4> fv;
	fv.row(0) << v[0], v[3], v[4], v[7];
	fv.row(1) << v[1], v[2], v[5], v[6];
	fv.row(2) << v[0], v[1], v[5], v[4];
	fv.row(3) << v[3], v[2], v[6], v[7];
	fv.row(4) << v[0], v[1], v[2], v[3];
	fv.row(5) << v[4], v[5], v[6], v[7];
	for (int lf = 0; lf < f.rows(); ++lf) {
		const auto index = find_quad_face(mesh, c, fv(lf, 0), fv(lf, 1), fv(lf, 2), fv(lf, 3));
		f[lf] = index;
	}


	//vertices
	for (size_t lv = 0; lv < v.size(); ++lv) {
		res.push_back(nodes.node_id_from_primitive(v[lv]));
	}
	assert(res.size() == size_t(8));

	//Edges
	for (int le = 0; le < e.rows(); ++le) {
		const auto index = e[le];
		auto neighs = mesh.edge_neighs(index.edge);
		int min_q = discr_order.size() > 0 ? discr_order(c) : 0;

		for(auto cid : neighs)
		{
			min_q = std::min(min_q, discr_order.size() > 0 ? discr_order(cid) : 0);
		}

		if(discr_order.size() > 0 && discr_order(c) >  min_q)
		{
			for(int tmp = 0; tmp < q - 1; ++tmp)
				res.push_back(-le - 10);
		}
		else
		{
			auto node_ids = nodes.node_ids_from_edge(index, q - 1);
			res.insert(res.end(), node_ids.begin(), node_ids.end());
		}
	}
	assert(res.size() == size_t(8 + n_edge_nodes));


	//faces
	for (int lf = 0; lf < f.rows(); ++lf) {
		const auto index = f[lf];
		const auto other_cell = mesh.switch_element(index).element;

		const bool skip_other = discr_order.size() > 0 && other_cell >= 0 && discr_order(c) >  discr_order(other_cell);

		if(skip_other)
		{
			for(int tmp = 0; tmp < n_loc_f; ++tmp)
				res.push_back(-lf - 1);
		}
		else
		{
			auto node_ids = nodes.node_ids_from_face(index, q - 1);
			assert(node_ids.size() == n_loc_f);
			res.insert(res.end(), node_ids.begin(), node_ids.end());
		}
	}
	assert(res.size() == size_t(8 + n_edge_nodes + n_face_nodes));

	//cells
	if (n_cell_nodes > 0) {
		const auto index = f[0];

		auto node_ids = nodes.node_ids_from_cell(index, q - 1);
		res.insert(res.end(), node_ids.begin(), node_ids.end());
	}

	assert(res.size() == size_t(8 + n_edge_nodes + n_face_nodes + n_cell_nodes));
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
	const polyfem::Mesh3D &mesh,
	const Eigen::VectorXi &discr_orders,
	const bool has_polys,
	MeshNodes &nodes,
	std::vector<std::vector<int> > &element_nodes_id,
	std::vector<polyfem::LocalBoundary> &local_boundary,
	std::map<int, polyfem::InterfaceData> &poly_face_to_data)
{
	// Step 1: Assign global node ids for each quads
	local_boundary.clear();
	// local_boundary.resize(mesh.n_faces());
	element_nodes_id.resize(mesh.n_faces());

	for (int c = 0; c < mesh.n_cells(); ++c) {
		const int discr_order = discr_orders(c);

		if (mesh.is_cube(c)) {
			hex_local_to_global(discr_order, mesh, c, discr_orders, element_nodes_id[c], nodes);


			auto v = hex_vertices_local_to_global(mesh, c);
			Eigen::Matrix<int, 6, 4> fv;
			fv.row(0) << v[0], v[3], v[4], v[7];
			fv.row(1) << v[1], v[2], v[5], v[6];
			fv.row(2) << v[0], v[1], v[5], v[4];
			fv.row(3) << v[3], v[2], v[6], v[7];
			fv.row(4) << v[0], v[1], v[2], v[3];
			fv.row(5) << v[4], v[5], v[6], v[7];

			LocalBoundary lb(c, BoundaryType::Quad);
			for(int i = 0; i < fv.rows(); ++i)
			{
				const int f = find_quad_face(mesh, c, fv(i, 0), fv(i, 1), fv(i, 2), fv(i, 3)).face;

				if (mesh.is_boundary_face(f) || mesh.get_boundary_id(f) != 0){
					lb.add_boundary_primitive(f, i);
				}
			}

			if(!lb.empty())
				local_boundary.emplace_back(lb);
		} else if(mesh.is_simplex(c)) {
			// element_nodes_id[c] = polyfem::FEBasis3d::tet_local_to_global(discr_order, mesh, c, discr_orders, nodes);
			tet_local_to_global(discr_order, mesh, c, discr_orders, element_nodes_id[c], nodes);


			auto v = tet_vertices_local_to_global(mesh, c);
			Eigen::Matrix<int, 4, 3> fv;
			fv.row(0) << v[0], v[1], v[2];
			fv.row(1) << v[0], v[1], v[3];
			fv.row(2) << v[1], v[2], v[3];
			fv.row(3) << v[2], v[0], v[3];

			LocalBoundary lb(c, BoundaryType::Tri);
			for(long i = 0; i < fv.rows(); ++i)
			{
				const int f = mesh.get_index_from_element_face(c, fv(i,0), fv(i,1), fv(i,2)).face;

				if(mesh.is_boundary_face(f) || mesh.get_boundary_id(f) > 0){
					lb.add_boundary_primitive(f, i);
				}
			}

			if(!lb.empty())
				local_boundary.emplace_back(lb);
		}
	}

	if(!has_polys)
		return;

	// Step 2: Iterate over edges of polygons and compute interface weights
	Eigen::VectorXi indices;
	for (int c = 0; c < mesh.n_cells(); ++c) {
		// Skip non-polytopes
		if (!mesh.is_polytope(c)) { continue; }

		for (int lf = 0; lf < mesh.n_cell_faces(c); ++lf) {
			auto index = mesh.get_index_from_element(c, lf, 0);
			auto index2 = mesh.switch_element(index);
			int c2 = index2.element;
			assert(c2 >= 0);

			const int discr_order = discr_orders(c2);
			if(mesh.is_cube(c2))
			{
				indices = polyfem::FEBasis3d::hex_face_local_nodes(discr_order, mesh, index2);
			} else if(mesh.is_simplex(c2)) {
				indices = polyfem::FEBasis3d::tet_face_local_nodes(discr_order, mesh, index2);
			}

			polyfem::InterfaceData data;
			data.local_indices.insert(data.local_indices.begin(), indices.data(), indices.data() + indices.size());
			assert(indices.size() == data.local_indices.size());
			poly_face_to_data[index2.face] = data;
		}
	}
}

} //anonymous namespace




Eigen::VectorXi polyfem::FEBasis3d::tet_face_local_nodes(const int p, const Mesh3D &mesh, Navigation3D::Index index)
{
	const int nn = p > 2 ? (p - 2) : 0;
	const int n_edge_nodes = (p-1)*6;
	const int n_face_nodes = nn * (nn + 1) / 2;

	const int c = index.element;
	assert(mesh.is_simplex(c));

	// Local to global mapping of node indices
	const auto l2g = tet_vertices_local_to_global(mesh, c);


	// Extract requested interface
	Eigen::VectorXi result(3 + (p - 1) * 3 + n_face_nodes);
	result[0] = find_index(l2g.begin(), l2g.end(), index.vertex);
	result[1] = find_index(l2g.begin(), l2g.end(), mesh.next_around_face(index).vertex);
	result[2] = find_index(l2g.begin(), l2g.end(), mesh.next_around_face(mesh.next_around_face(index)).vertex);

	Eigen::Matrix<Navigation3D::Index, 6, 1> e;
	Eigen::Matrix<int, 6, 2> ev;
	ev.row(0)  << l2g[0], l2g[1];
	ev.row(1)  << l2g[1], l2g[2];
	ev.row(2)  << l2g[2], l2g[0];

	ev.row(3)  << l2g[0], l2g[3];
	ev.row(4)  << l2g[1], l2g[3];
	ev.row(5)  << l2g[2], l2g[3];

	Navigation3D::Index tmp = index;

	for (int le = 0; le < e.rows(); ++le) {
		// const auto index =  find_edge(mesh, c, ev(le, 0), ev(le, 1));
		const auto l_index = mesh.get_index_from_element_edge(c, ev(le, 0), ev(le, 1));
		e[le] = l_index;
	}

	int ii = 3;
	for(int k = 0; k < 3; ++k)
	{
		bool reverse = false;
		int le = 0;
		for (; le < ev.rows(); ++le)
		{
			// const auto l_index =  find_edge(mesh, c, ev(le, 0), ev(le, 1));
			// const auto l_index = mesh.get_index_from_element_edge(c, ev(le, 0), ev(le, 1));
			const auto l_index = e[le];
			if(l_index.edge == tmp.edge)
			{
				if(l_index.vertex == tmp.vertex)
					reverse = false;
				else
				{
					reverse = true;
					assert(mesh.switch_vertex(tmp).vertex == l_index.vertex);
				}

				break;
			}
		}
		assert(le < 6);


		if(!reverse)
		{

			for(int i = 0; i < p - 1; ++i)
			{
				result[ii++] = 4 + le*(p-1) + i;
			}
		}
		else
		{
			for(int i = 0; i < p - 1; ++i)
			{
				result[ii++] = 4 + (le+1)*(p-1) - i - 1;
			}
		}

		tmp = mesh.next_around_face(tmp);
	}

	//faces

	Eigen::Matrix<int, 4, 3> fv;
	fv.row(0) << l2g[0], l2g[1], l2g[2];
	fv.row(1) << l2g[0], l2g[1], l2g[3];
	fv.row(2) << l2g[1], l2g[2], l2g[3];
	fv.row(3) << l2g[2], l2g[0], l2g[3];

	long lf = 0;
	for(; lf < fv.rows(); ++lf)
	{
		const auto l_index =  mesh.get_index_from_element_face(c, fv(lf,0), fv(lf,1), fv(lf,2));
		if(l_index.face == index.face)
			break;
	}

	assert(lf < fv.rows());

	if(n_face_nodes == 1)
		result[ii++] = 4 + n_edge_nodes + lf;
	else if(n_face_nodes == 3)
	{
		Eigen::MatrixXd nodes; autogen::p_nodes_3d(4, nodes);
		// auto pos = polyfem::FEBasis3d::linear_tet_face_local_nodes_coordinates(mesh, index);
		// Local to global mapping of node indices

		// Extract requested interface
		std::array<int, 3> idx;
		for (int lv = 0; lv < 3; ++lv) {
			idx[lv] = find_index(l2g.begin(), l2g.end(), index.vertex);
			index = mesh.next_around_face(index);
		}
		Eigen::Matrix3d pos(3, 3);
		int cnt = 0;
		for (int i : idx) {
			pos.row(cnt++) = nodes.row(i);
		}

		const Eigen::RowVector3d bary = pos.colwise().mean();

		const int offset = 4 + n_edge_nodes;
		bool found= false;
		for(int lff = 0; lff < 4; ++lff)
		{
			Eigen::Matrix3d loc_nodes = nodes.block<3,3>(offset+lff*n_face_nodes, 0);
			Eigen::RowVector3d node_bary = loc_nodes.colwise().mean();

			if((node_bary-bary).norm() < 1e-10)
			{
				int sum = 0;
				for(int m = 0; m < 3; ++m)
				{
					auto t = pos.row(m);
					int min_n = -1; double min_dis = 10000;

					for(int n = 0; n < 3; ++n)
					{
						double dis = (loc_nodes.row(n)-t).squaredNorm();
						if(dis<min_dis)
						{
							min_dis = dis;
							min_n = n;
						}
					}

					assert(min_n >= 0);
					assert(min_n < 3);

					sum += min_n;

					result[ii++] = 4 + n_edge_nodes + min_n + lf*n_face_nodes;
				}

				assert(sum == 3);


				found = true;
				assert(lff==lf);
			}

			if(found)
				break;
		}

		assert(found);
	}
	else
	{
		assert(n_face_nodes == 0);
	}

	assert(ii == result.size());
	return result;
}


Eigen::VectorXi polyfem::FEBasis3d::hex_face_local_nodes(const int q, const Mesh3D &mesh, Navigation3D::Index index)
{
	const int nn = q - 1;
	const int n_edge_nodes = nn * 12;
	const int n_face_nodes = nn * nn;

	const int c = index.element;
	assert(mesh.is_cube(c));

	// Local to global mapping of node indices
	const auto l2g = hex_vertices_local_to_global(mesh, c);


	// Extract requested interface
	Eigen::VectorXi result(4 + nn * 4 + n_face_nodes);
	result[0] = find_index(l2g.begin(), l2g.end(), index.vertex);
	result[1] = find_index(l2g.begin(), l2g.end(), mesh.next_around_face(index).vertex);
	result[2] = find_index(l2g.begin(), l2g.end(), mesh.next_around_face(mesh.next_around_face(index)).vertex);
	result[3] = find_index(l2g.begin(), l2g.end(), mesh.next_around_face(mesh.next_around_face(mesh.next_around_face(index))).vertex);

	Eigen::Matrix<Navigation3D::Index, 12, 1> e;
	Eigen::Matrix<int, 12, 2> ev;
	ev.row(0)  << l2g[0], l2g[1];
	ev.row(1)  << l2g[1], l2g[2];
	ev.row(2)  << l2g[2], l2g[3];
	ev.row(3)  << l2g[3], l2g[0];
	ev.row(4)  << l2g[0], l2g[4];
	ev.row(5)  << l2g[1], l2g[5];
	ev.row(6)  << l2g[2], l2g[6];
	ev.row(7)  << l2g[3], l2g[7];
	ev.row(8)  << l2g[4], l2g[5];
	ev.row(9)  << l2g[5], l2g[6];
	ev.row(10) << l2g[6], l2g[7];
	ev.row(11) << l2g[7], l2g[4];

	Navigation3D::Index tmp = index;

	for (int le = 0; le < e.rows(); ++le) {
		const auto l_index = mesh.get_index_from_element_edge(c, ev(le, 0), ev(le, 1));
		e[le] = l_index;
	}

	int ii = 4;
	for(int k = 0; k < 4; ++k)
	{
		bool reverse = false;
		int le = 0;
		for (; le < ev.rows(); ++le)
		{
			// const auto l_index =  find_edge(mesh, c, ev(le, 0), ev(le, 1));
			// const auto l_index = mesh.get_index_from_element_edge(c, ev(le, 0), ev(le, 1));
			const auto l_index = e[le];
			if(l_index.edge == tmp.edge)
			{
				if(l_index.vertex == tmp.vertex)
					reverse = false;
				else
				{
					reverse = true;
					assert(mesh.switch_vertex(tmp).vertex == l_index.vertex);
				}

				break;
			}
		}
		assert(le < 12);


		if(!reverse)
		{

			for(int i = 0; i < q - 1; ++i)
			{
				result[ii++] = 8 + le*(q-1) + i;
			}
		}
		else
		{
			for(int i = 0; i < q - 1; ++i)
			{
				result[ii++] = 8 + (le+1)*(q-1) - i - 1;
			}
		}

		tmp = mesh.next_around_face(tmp);
	}

	//faces

	Eigen::Matrix<int, 6, 4> fv;
	fv.row(0) << l2g[0], l2g[3], l2g[4], l2g[7];
	fv.row(1) << l2g[1], l2g[2], l2g[5], l2g[6];
	fv.row(2) << l2g[0], l2g[1], l2g[5], l2g[4];
	fv.row(3) << l2g[3], l2g[2], l2g[6], l2g[7];
	fv.row(4) << l2g[0], l2g[1], l2g[2], l2g[3];
	fv.row(5) << l2g[4], l2g[5], l2g[6], l2g[7];

	long lf = 0;
	for (; lf < fv.rows(); ++lf) {
		const auto l_index = find_quad_face(mesh, c, fv(lf, 0), fv(lf, 1), fv(lf, 2), fv(lf, 3));
		if(l_index.face == index.face)
			break;
	}

	assert(lf < fv.rows());

	if(n_face_nodes == 1)
		result[ii++] = 8 + n_edge_nodes + lf;
	else if(n_face_nodes != 0)
	{
		Eigen::MatrixXd nodes; autogen::q_nodes_3d(q, nodes);
		// auto pos = polyfem::FEBasis3d::linear_tet_face_local_nodes_coordinates(mesh, index);
		// Local to global mapping of node indices

		// Extract requested interface
		std::array<int, 4> idx;
		for (int lv = 0; lv < 4; ++lv) {
			idx[lv] = find_index(l2g.begin(), l2g.end(), index.vertex);
			index = mesh.next_around_face(index);
		}
		Eigen::Matrix<double, 4, 3> pos(4, 3);
		int cnt = 0;
		for (int i : idx) {
			pos.row(cnt++) = nodes.row(i);
		}

		const Eigen::RowVector3d bary = pos.colwise().mean();

		const int offset = 8 + n_edge_nodes;
		bool found= false;
		for(int lff = 0; lff < 6; ++lff)
		{
			Eigen::Matrix<double, 4, 3> loc_nodes = nodes.block<4,3>(offset+lff*n_face_nodes, 0);
			Eigen::RowVector3d node_bary = loc_nodes.colwise().mean();

			if((node_bary-bary).norm() < 1e-10)
			{
				int sum = 0;
				for(int m = 0; m < 4; ++m)
				{
					auto t = pos.row(m);
					int min_n = -1; double min_dis = 10000;

					for(int n = 0; n < 4; ++n)
					{
						double dis = (loc_nodes.row(n)-t).squaredNorm();
						if(dis<min_dis)
						{
							min_dis = dis;
							min_n = n;
						}
					}

					assert(min_n >= 0);
					assert(min_n < 4);

					sum += min_n;

					result[ii++] = 8 + n_edge_nodes + min_n + lf*n_face_nodes;
				}

				assert(sum == 6); // 0 + 1 + 2 + 3


				found = true;
				assert(lff==lf);
			}

			if(found)
				break;
		}

		assert(found);
	}

	assert(ii == result.size());
	return result;
}



int polyfem::FEBasis3d::build_bases(
	const Mesh3D &mesh,
	const int quadrature_order,
	const int discr_order,
	const bool has_polys,
	std::vector< ElementBases > &bases,
	std::vector< LocalBoundary > &local_boundary,
	std::map<int, InterfaceData> &poly_face_to_data)
{
	Eigen::VectorXi discr_orders(mesh.n_cells());
	discr_orders.setConstant(discr_order);

	return build_bases(mesh, quadrature_order, discr_orders, has_polys, bases, local_boundary, poly_face_to_data);
}

int polyfem::FEBasis3d::build_bases(
	const Mesh3D &mesh,
	const int quadrature_order,
	const Eigen::VectorXi &discr_orders,
	const bool has_polys,
	std::vector< ElementBases > &bases,
	std::vector< LocalBoundary > &local_boundary,
	std::map<int, InterfaceData> &poly_face_to_data)
{
	assert(mesh.is_volume());
	assert(discr_orders.size() == mesh.n_cells());

	// Navigation3D::get_index_from_element_face_time = 0;
	// Navigation3D::switch_vertex_time = 0;
	// Navigation3D::switch_edge_time = 0;
	// Navigation3D::switch_face_time = 0;
	// Navigation3D::switch_element_time = 0;

	const int max_p = discr_orders.maxCoeff();
	assert(max_p < 5); //P5 not supported

	const int nn = max_p > 1 ? (max_p - 1) : 0;
	const int n_face_nodes = nn*nn;
	const int n_cells_nodes = nn*nn*nn;

	MeshNodes nodes(mesh, has_polys, nn, n_face_nodes, max_p == 0 ? 1 : n_cells_nodes);
	std::vector<std::vector<int>> element_nodes_id;
	compute_nodes(mesh, discr_orders, has_polys, nodes, element_nodes_id, local_boundary, poly_face_to_data);
	// boundary_nodes = nodes.boundary_nodes();


	// std::cout<<"get_index_from_element_face_time " << Navigation3D::get_index_from_element_face_time <<std::endl;
	// std::cout<<"switch_vertex_time " << Navigation3D::switch_vertex_time <<std::endl;
	// std::cout<<"switch_edge_time " << Navigation3D::switch_edge_time <<std::endl;
	// std::cout<<"switch_face_time " << Navigation3D::switch_face_time <<std::endl;
	// std::cout<<"switch_element_time " << Navigation3D::switch_element_time <<std::endl;

	bases.resize(mesh.n_cells());
	std::vector<int> interface_elements; interface_elements.reserve(mesh.n_faces());

	for (int e = 0; e < mesh.n_cells(); ++e) {
		ElementBases &b = bases[e];
		const int discr_order = discr_orders(e);
		const int n_el_bases = (int) element_nodes_id[e].size();
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
			// hex_quadrature.get_quadrature(quadrature_order, b.quadrature);
			b.set_quadrature([quadrature_order](Quadrature &quad){
				HexQuadrature hex_quadrature;
				hex_quadrature.get_quadrature(quadrature_order, quad);
			});


			b.set_local_node_from_primitive_func([discr_order, e](const int primitive_id, const Mesh &mesh)
			{
				const auto &mesh3d = dynamic_cast<const Mesh3D &>(mesh);
				Navigation3D::Index index;

				for(int lf = 0; lf < 6; ++lf)
				{
					index = mesh3d.get_index_from_element(e, lf, 0);
					if(index.face == primitive_id)
						break;
				}
				assert(index.face == primitive_id);
				return hex_face_local_nodes(discr_order, mesh3d, index);
			});

			for (int j = 0; j < n_el_bases; ++j) {
				const int global_index = element_nodes_id[e][j];

				b.bases[j].init(discr_order, global_index, j, nodes.node_position(global_index));

				b.bases[j].set_basis([discr_order, j](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { polyfem::autogen::q_basis_value_3d     (discr_order, j, uv, val); });
				b.bases[j].set_grad ([discr_order, j](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { polyfem::autogen::q_grad_basis_value_3d(discr_order, j, uv, val); });
			}
		}
		else if(mesh.is_simplex(e)) {
			const int real_order = std::max(quadrature_order, (discr_order - 1) * (discr_order - 1));

			b.set_quadrature([real_order](Quadrature &quad){
				TetQuadrature tet_quadrature;
				tet_quadrature.get_quadrature(real_order, quad);
			});


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
				return tet_face_local_nodes(discr_order, mesh3d, index);
			});


			for (int j = 0; j < n_el_bases; ++j) {
				const int global_index = element_nodes_id[e][j];
				if(!skip_interface_element){
					b.bases[j].init(discr_order, global_index, j, nodes.node_position(global_index));
				}

				b.bases[j].set_basis([discr_order, j](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { polyfem::autogen::p_basis_value_3d     (discr_order, j, uv, val); });
				b.bases[j].set_grad ([discr_order, j](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { polyfem::autogen::p_grad_basis_value_3d(discr_order, j, uv, val); });
			}

		}
		else {
			// Polyhedra bases are built later on
			// assert(false);
		}
	}


	for(int pp = 2; pp <= autogen::MAX_P_BASES; ++pp)
	{
		for (int e : interface_elements) {
			ElementBases &b = bases[e];
			const int discr_order = discr_orders(e);
			const int n_el_bases = element_nodes_id[e].size();
			assert(discr_order > 1);
			if(discr_order != pp)
				continue;


			if (mesh.is_cube(e)) {
				//TODO
				assert(false);
			}
			else if(mesh.is_simplex(e))
			{
				for (int j = 0; j < n_el_bases; ++j) {
					const int global_index = element_nodes_id[e][j];

					if(global_index >= 0)
						b.bases[j].init(discr_order, global_index, j, nodes.node_position(global_index));
					else
					{
						const int lnn = max_p > 2 ? (discr_order - 2) : 0;
						const int ln_edge_nodes = discr_order - 1;
						const int ln_face_nodes = lnn * (lnn + 1) / 2;

						const auto v = tet_vertices_local_to_global(mesh, e);
						Navigation3D::Index index;
						if(global_index <= -30)
						{
							assert(false);
							// const auto lv = -(global_index + 30);
							// assert(lv>=0 && lv < 4);
							// assert(j < 4);

							// if(lv == 3)
							// {
							// 	index = mesh.switch_element(find_edge(mesh, e, v[lv], v[0]));
							// 	if(index.element < 0)
							// 		index = mesh.switch_element(find_edge(mesh, e, v[lv], v[1]));
							// 	if(index.element < 0)
							// 		index = mesh.switch_element(find_edge(mesh, e, v[lv], v[2]));
							// }
							// else
							// {
							// 	index = mesh.switch_element(find_edge(mesh, e, v[lv], v[(lv+1)%3]));
							// 	if(index.element < 0)
							// 		index = mesh.switch_element(find_edge(mesh, e, v[lv], v[(lv+2)%3]));
							// 	if(index.element < 0)
							// 		index = mesh.switch_element(find_edge(mesh, e, v[lv], v[3]));
							// }
						}
						else if (global_index <= -10)
						{
							const auto le = -(global_index+10);
							assert(le>=0 && le < 6);
							assert(j >= 4 && j < 4 + 6*ln_edge_nodes);

							Eigen::Matrix<int, 6, 2> ev;
							ev.row(0)  << v[0], v[1];
							ev.row(1)  << v[1], v[2];
							ev.row(2)  << v[2], v[0];

							ev.row(3)  << v[0], v[3];
							ev.row(4)  << v[1], v[3];
							ev.row(5)  << v[2], v[3];

							// const auto edge_index = find_edge(mesh, e, ev(le, 0), ev(le, 1));
							const auto edge_index = mesh.get_index_from_element_edge(e, ev(le, 0), ev(le, 1));
							auto neighs = mesh.edge_neighs(edge_index.edge);
							int min_p = discr_order;
							int min_cell = index.element;

							for(auto cid : neighs)
							{
								if(discr_orders[cid] < min_p)
								{
									min_p = discr_orders[cid];
									min_cell = cid;
								}
							}

							bool found = false;
							for(int lf = 0; lf < 4; ++lf)
							{
								for(int lv = 0; lv < 4; ++lv)
								{
									index = mesh.get_index_from_element(min_cell, lf, lv);

									if(index.vertex == edge_index.vertex)
									{
										if(index.edge != edge_index.edge)
										{
											auto tmp = index;
											index = mesh.switch_edge(tmp);

											if(index.edge != edge_index.edge)
											{
												index = mesh.switch_edge(mesh.switch_face(tmp));
											}
										}
										found = true;
										break;
									}
								}

								if(found)
									break;
							}

							assert(found);
							assert(index.vertex == edge_index.vertex && index.edge == edge_index.edge);
							assert(index.element != edge_index.element);
						}
						else
						{
							const auto lf = -(global_index+1);
							assert(lf>=0 && lf < 4);
							assert(j >= 4 + 6*ln_edge_nodes && j <  4 + 6*ln_edge_nodes + 4*ln_face_nodes);

							Eigen::Matrix<int, 4, 3> fv;
							fv.row(0) << v[0], v[1], v[2];
							fv.row(1) << v[0], v[1], v[3];
							fv.row(2) << v[1], v[2], v[3];
							fv.row(3) << v[2], v[0], v[3];

							index = mesh.switch_element(mesh.get_index_from_element_face(e, fv(lf, 0), fv(lf, 1), fv(lf, 2)));
						}

						const auto other_cell = index.element;
						assert(other_cell >= 0);
						assert(discr_order > discr_orders(other_cell));


						auto indices = tet_face_local_nodes(discr_order, mesh, index);
						Eigen::MatrixXd lnodes; autogen::p_nodes_3d(discr_order, lnodes);
						Eigen::RowVector3d node_position; // = lnodes.row(indices(ii));


						if( j < 4)
							node_position = lnodes.row(indices(0));
						else if( j < 4 + 6*ln_edge_nodes)
							node_position = lnodes.row(indices(((j - 4) % ln_edge_nodes) + 3));
						else if(j <  4 + 6*ln_edge_nodes + 4*ln_face_nodes)
						{
							// node_position = lnodes.row(indices(((j - 4 - 6*ln_edge_nodes) % ln_face_nodes) + 3 + 3*ln_edge_nodes));
							auto me_indices = tet_face_local_nodes(discr_order, mesh, mesh.switch_element(index));
							int ii;
							for(ii=0; ii< me_indices.size(); ++ii)
							{
								if(me_indices(ii) == j)
									break;
							}

							assert(ii >= 3 + 3*ln_edge_nodes);
							assert(ii < me_indices.size());

							node_position = lnodes.row(indices(ii));
						}
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


						const auto &other_bases = bases[other_cell];
						// Eigen::MatrixXd w;
						std::vector<AssemblyValues> w;
						other_bases.evaluate_bases(node_position, w);

						assert(b.bases[j].global().size() == 0);

						for(long i = 0; i < w.size(); ++i)
						{
							assert(w[i].val.size() == 1);
							if(std::abs(w[i].val(0))<1e-8)
								continue;

							// assert(other_bases.bases[i].global().size() == 1);
							for(size_t ii = 0; ii < other_bases.bases[i].global().size(); ++ii)
							{
								const auto &other_global = other_bases.bases[i].global()[ii];
								// std::cout<<"e "<<e<<" " <<j << " gid "<<other_global.index<<std::endl;
								b.bases[j].global().emplace_back(other_global.index, other_global.node, w[i].val(0)*other_global.val);
							}
						}
					}
				}
			}
			else {
			// Polygon bases are built later on
			}
		}
	}




	return nodes.n_nodes();
}
