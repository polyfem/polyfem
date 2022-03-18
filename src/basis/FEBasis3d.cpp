
////////////////////////////////////////////////////////////////////////////////
#include <polyfem/FEBasis3d.hpp>
#include <polyfem/MeshNodes.hpp>
#include <polyfem/TetQuadrature.hpp>
#include <polyfem/HexQuadrature.hpp>

#include <polyfem/auto_p_bases.hpp>
#include <polyfem/auto_q_bases.hpp>

#include <polyfem/MaybeParallelFor.hpp>
#include <polyfem/Logger.hpp>

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

namespace
{

	template <class InputIterator, class T>
	int find_index(InputIterator first, InputIterator last, const T &val)
	{
		return std::distance(first, std::find(first, last, val));
	}

	Navigation3D::Index find_quad_face(const polyfem::Mesh3D &mesh, int c, int v1, int v2, int v3, int v4)
	{
		std::array<int, 4> v = {{v1, v2, v3, v4}};
		std::sort(v.begin(), v.end());
		for (int lf = 0; lf < mesh.n_cell_faces(c); ++lf)
		{
			auto idx = mesh.get_index_from_element(c, lf, 0);
			assert(mesh.n_face_vertices(idx.face) == 4);
			std::array<int, 4> u;
			for (int lv = 0; lv < mesh.n_face_vertices(idx.face); ++lv)
			{
				u[lv] = idx.vertex;
				idx = mesh.next_around_face(idx);
			}
			std::sort(u.begin(), u.end());
			if (u == v)
			{
				return idx;
			}
		}
		assert(false);
		return Navigation3D::Index();
	}

	std::array<int, 4> tet_vertices_local_to_global(const Mesh3D &mesh, int c)
	{
		// Vertex nodes
		assert(mesh.is_simplex(c));
		std::array<int, 4> l2g;
		int lv = 0;
		for (int vi : mesh.get_ordered_vertices_from_tet(c))
		{
			l2g[lv++] = vi;
		}

		return l2g;
	}

	std::array<int, 8> hex_vertices_local_to_global(const Mesh3D &mesh, int c)
	{
		assert(mesh.is_cube(c));

		// Vertex nodes
		std::array<int, 8> l2g;
		int lv = 0;
		for (int vi : mesh.get_ordered_vertices_from_hex(c))
		{
			l2g[lv++] = vi;
		}

		return l2g;
	}

	void tet_local_to_global(const bool is_geom_bases, const int p, const Mesh3D &mesh, int c, const Eigen::VectorXi &discr_order, std::vector<int> &res, polyfem::MeshNodes &nodes)
	{
		const int n_edge_nodes = p > 1 ? ((p - 1) * 6) : 0;
		const int nn = p > 2 ? (p - 2) : 0;
		const int n_loc_f = (nn * (nn + 1) / 2);
		const int n_face_nodes = n_loc_f * 4;
		int n_cell_nodes = 0;
		for (int pp = 4; pp <= p; ++pp)
			n_cell_nodes += ((pp - 3) * ((pp - 3) + 1) / 2);

		if (p == 0)
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

		for (long lf = 0; lf < fv.rows(); ++lf)
		{
			const auto index = mesh.get_index_from_element_face(c, fv(lf, 0), fv(lf, 1), fv(lf, 2));
			f[lf] = index;
		}

		Eigen::Matrix<Navigation3D::Index, 6, 1> e;
		Eigen::Matrix<int, 6, 2> ev;
		ev.row(0) << v[0], v[1];
		ev.row(1) << v[1], v[2];
		ev.row(2) << v[2], v[0];

		ev.row(3) << v[0], v[3];
		ev.row(4) << v[1], v[3];
		ev.row(5) << v[2], v[3];

		for (int le = 0; le < e.rows(); ++le)
		{
			// const auto index =  find_edge(mesh, c, ev(le, 0), ev(le, 1));
			const auto index = mesh.get_index_from_element_edge(c, ev(le, 0), ev(le, 1));
			e[le] = index;
		}

		// vertices
		for (size_t lv = 0; lv < v.size(); ++lv)
		{
			if (mesh.ncmesh && !is_geom_bases) {
				auto& ncmesh = mesh.ncmesh;
				// hanging vertex
				if (ncmesh->vertices[ncmesh->valid2AllVertex(v[lv])].edge >= 0 || ncmesh->vertices[ncmesh->valid2AllVertex(v[lv])].face >= 0)
					res.push_back(-lv - 1);
				else
					res.push_back(nodes.node_id_from_primitive(v[lv]));
			}
			else
				res.push_back(nodes.node_id_from_primitive(v[lv]));
		}

		// Edges
		for (int le = 0; le < e.rows(); ++le)
		{
			const auto index = e[le];
			auto neighs = mesh.edge_neighs(index.edge);
			int min_p = discr_order.size() > 0 ? discr_order(c) : 0;

			if (is_geom_bases)
			{
				auto node_ids = nodes.node_ids_from_edge(index, p - 1);
				res.insert(res.end(), node_ids.begin(), node_ids.end());
			}
			else
			{
				if (mesh.ncmesh && !is_geom_bases) {
					auto &ncmesh = *dynamic_cast<ncMesh3D *>(mesh.ncmesh.get());
					auto& ncelem = ncmesh.elements[ncmesh.valid2All(c)];
					const int edge_id = ncmesh.findEdge(ncmesh.valid2AllVertex(ev(le, 0)), ncmesh.valid2AllVertex(ev(le, 1)));
					auto& ncedge = ncmesh.edges[edge_id];
					// slave edge
					if (ncedge.master >= 0 || ncedge.master_face >= 0) {
						for (int tmp = 0; tmp < p - 1; ++tmp)
							res.push_back(-le - 1);
					}
					// master or conforming edge with constrained order
					else if (ncedge.order < ncelem.order) {
						for (int tmp = 0; tmp < p - 1; ++tmp)
							res.push_back(-le - 1);

						int min_order_elem = ncmesh.getLowestOrderElementOnEdge(edge_id);
						// master edge, add extra nodes
						if (min_order_elem == ncmesh.valid2All(c))
							ncedge.global_ids = nodes.node_ids_from_edge(index, ncedge.order - 1);
					}
					else {
						auto node_ids = nodes.node_ids_from_edge(index, p - 1);
						res.insert(res.end(), node_ids.begin(), node_ids.end());
					}
				}
				else {
					for (auto cid : neighs)
					{
						min_p = std::min(min_p, discr_order.size() > 0 ? discr_order(cid) : 0);
					}

					if (discr_order.size() > 0 && discr_order(c) > min_p)
					{
						for (int tmp = 0; tmp < p - 1; ++tmp)
							res.push_back(-le - 10);
					}
					else
					{
						auto node_ids = nodes.node_ids_from_edge(index, p - 1);
						res.insert(res.end(), node_ids.begin(), node_ids.end());
					}
				}
			}
		}

		// faces
		for (int lf = 0; lf < f.rows(); ++lf)
		{
			const auto index = f[lf];
			const auto other_cell = mesh.switch_element(index).element;

			const bool skip_other = discr_order.size() > 0 && other_cell >= 0 && discr_order(c) > discr_order(other_cell);

			if (is_geom_bases)
			{
				auto node_ids = nodes.node_ids_from_face(index, p - 2);
				res.insert(res.end(), node_ids.begin(), node_ids.end());
			}
			else
			{
				if (mesh.ncmesh && !is_geom_bases) {
					auto &ncmesh = *dynamic_cast<ncMesh3D *>(mesh.ncmesh.get());
					auto &ncelem = ncmesh.elements[ncmesh.valid2All(c)];
					auto &ncface = ncmesh.faces[ncmesh.findFace(ncmesh.valid2AllVertex(fv(lf, 0)), ncmesh.valid2AllVertex(fv(lf, 1)), ncmesh.valid2AllVertex(fv(lf, 2)))];
					// slave face
					if (ncface.master >= 0) {
						for (int tmp = 0; tmp < n_loc_f; ++tmp)
							res.push_back(-lf - 1);
					}
					// master face or conforming face with constrained order
					else if (ncface.order < ncelem.order) {
						for (int tmp = 0; tmp < n_loc_f; ++tmp)
							res.push_back(-lf - 1);
						// master face
						if (ncface.slaves.size() > 0 && ncface.order > 2)
							ncface.global_ids = nodes.node_ids_from_face(index, ncface.order - 2);
					}
					else {
						auto node_ids = nodes.node_ids_from_face(index, p - 2);
						res.insert(res.end(), node_ids.begin(), node_ids.end());
					}
				}
				else {
					if (skip_other)
					{
						for (int tmp = 0; tmp < n_loc_f; ++tmp)
							res.push_back(-lf - 1);
					}
					else
					{
						auto node_ids = nodes.node_ids_from_face(index, p - 2);
						res.insert(res.end(), node_ids.begin(), node_ids.end());
					}
				}
			}
		}

		// cells
		if (n_cell_nodes > 0)
		{
			const auto index = f[0];

			auto node_ids = nodes.node_ids_from_cell(index, p - 3);
			res.insert(res.end(), node_ids.begin(), node_ids.end());
		}

		assert(res.size() == size_t(4 + n_edge_nodes + n_face_nodes + n_cell_nodes));
	}

	void hex_local_to_global(const bool serendipity, const int q, const Mesh3D &mesh, int c, const Eigen::VectorXi &discr_order, std::vector<int> &res, polyfem::MeshNodes &nodes)
	{
		assert(mesh.is_cube(c));

		const int n_edge_nodes = ((q - 1) * 12);
		const int nn = (q - 1);
		const int n_loc_f = serendipity ? 0 : (nn * nn);
		const int n_face_nodes = serendipity ? 0 : (n_loc_f * 6);
		const int n_cell_nodes = serendipity ? 0 : (nn * nn * nn);

		if (q == 0)
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
		ev.row(0) << v[0], v[1];
		ev.row(1) << v[1], v[2];
		ev.row(2) << v[2], v[3];
		ev.row(3) << v[3], v[0];
		ev.row(4) << v[0], v[4];
		ev.row(5) << v[1], v[5];
		ev.row(6) << v[2], v[6];
		ev.row(7) << v[3], v[7];
		ev.row(8) << v[4], v[5];
		ev.row(9) << v[5], v[6];
		ev.row(10) << v[6], v[7];
		ev.row(11) << v[7], v[4];
		for (int le = 0; le < e.rows(); ++le)
		{
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
		for (int lf = 0; lf < f.rows(); ++lf)
		{
			const auto index = find_quad_face(mesh, c, fv(lf, 0), fv(lf, 1), fv(lf, 2), fv(lf, 3));
			f[lf] = index;
		}

		// vertices
		for (size_t lv = 0; lv < v.size(); ++lv)
		{
			res.push_back(nodes.node_id_from_primitive(v[lv]));
		}
		assert(res.size() == size_t(8));

		// Edges
		for (int le = 0; le < e.rows(); ++le)
		{
			const auto index = e[le];
			auto neighs = mesh.edge_neighs(index.edge);
			int min_q = discr_order.size() > 0 ? discr_order(c) : 0;

			for (auto cid : neighs)
			{
				min_q = std::min(min_q, discr_order.size() > 0 ? discr_order(cid) : 0);
			}

			if (discr_order.size() > 0 && discr_order(c) > min_q)
			{
				for (int tmp = 0; tmp < q - 1; ++tmp)
					res.push_back(-le - 10);
			}
			else
			{
				auto node_ids = nodes.node_ids_from_edge(index, q - 1);
				res.insert(res.end(), node_ids.begin(), node_ids.end());
			}
		}
		assert(res.size() == size_t(8 + n_edge_nodes));

		// faces
		for (int lf = 0; lf < f.rows(); ++lf)
		{
			const auto index = f[lf];
			const auto other_cell = mesh.switch_element(index).element;

			const bool skip_other = discr_order.size() > 0 && other_cell >= 0 && discr_order(c) > discr_order(other_cell);

			if (skip_other)
			{
				for (int tmp = 0; tmp < n_loc_f; ++tmp)
					res.push_back(-lf - 1);
			}
			else
			{
				auto node_ids = nodes.node_ids_from_face(index, serendipity ? 0 : (q - 1));
				assert(node_ids.size() == n_loc_f);
				res.insert(res.end(), node_ids.begin(), node_ids.end());
			}
		}
		assert(res.size() == size_t(8 + n_edge_nodes + n_face_nodes));

		// cells
		if (n_cell_nodes > 0)
		{
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
		const bool serendipity,
		const bool has_polys,
		const bool is_geom_bases,
		MeshNodes &nodes,
		std::vector<std::vector<int>> &element_nodes_id,
		std::vector<polyfem::LocalBoundary> &local_boundary,
		std::map<int, polyfem::InterfaceData> &poly_face_to_data)
	{
		// Step 1: Assign global node ids for each quads
		local_boundary.clear();
		// local_boundary.resize(mesh.n_faces());
		element_nodes_id.resize(mesh.n_faces());

		for (int c = 0; c < mesh.n_cells(); ++c)
		{
			const int discr_order = discr_orders(c);

			if (mesh.is_cube(c))
			{
				hex_local_to_global(serendipity, discr_order, mesh, c, discr_orders, element_nodes_id[c], nodes);

				auto v = hex_vertices_local_to_global(mesh, c);
				Eigen::Matrix<int, 6, 4> fv;
				fv.row(0) << v[0], v[3], v[4], v[7];
				fv.row(1) << v[1], v[2], v[5], v[6];
				fv.row(2) << v[0], v[1], v[5], v[4];
				fv.row(3) << v[3], v[2], v[6], v[7];
				fv.row(4) << v[0], v[1], v[2], v[3];
				fv.row(5) << v[4], v[5], v[6], v[7];

				LocalBoundary lb(c, BoundaryType::Quad);
				for (int i = 0; i < fv.rows(); ++i)
				{
					const int f = find_quad_face(mesh, c, fv(i, 0), fv(i, 1), fv(i, 2), fv(i, 3)).face;

					if (mesh.is_boundary_face(f) || mesh.get_boundary_id(f) > 0)
					{
						lb.add_boundary_primitive(f, i);
					}
				}

				if (!lb.empty())
					local_boundary.emplace_back(lb);
			}
			else if (mesh.is_simplex(c))
			{
				// element_nodes_id[c] = polyfem::FEBasis3d::tet_local_to_global(discr_order, mesh, c, discr_orders, nodes);
				tet_local_to_global(is_geom_bases, discr_order, mesh, c, discr_orders, element_nodes_id[c], nodes);

				auto v = tet_vertices_local_to_global(mesh, c);
				Eigen::Matrix<int, 4, 3> fv;
				fv.row(0) << v[0], v[1], v[2];
				fv.row(1) << v[0], v[1], v[3];
				fv.row(2) << v[1], v[2], v[3];
				fv.row(3) << v[2], v[0], v[3];

				LocalBoundary lb(c, BoundaryType::Tri);
				for (long i = 0; i < fv.rows(); ++i)
				{
					const int f = mesh.get_index_from_element_face(c, fv(i, 0), fv(i, 1), fv(i, 2)).face;

					if (mesh.is_boundary_face(f) || mesh.get_boundary_id(f) > 0)
					{
						lb.add_boundary_primitive(f, i);
					}
				}

				if (!lb.empty())
					local_boundary.emplace_back(lb);
			}
		}

		if (!has_polys)
			return;

		// Step 2: Iterate over edges of polygons and compute interface weights
		Eigen::VectorXi indices;
		for (int c = 0; c < mesh.n_cells(); ++c)
		{
			// Skip non-polytopes
			if (!mesh.is_polytope(c))
			{
				continue;
			}

			for (int lf = 0; lf < mesh.n_cell_faces(c); ++lf)
			{
				auto index = mesh.get_index_from_element(c, lf, 0);
				auto index2 = mesh.switch_element(index);
				int c2 = index2.element;
				assert(c2 >= 0);

				const int discr_order = discr_orders(c2);
				if (mesh.is_cube(c2))
				{
					indices = polyfem::FEBasis3d::hex_face_local_nodes(serendipity, discr_order, mesh, index2);
				}
				else if (mesh.is_simplex(c2))
				{
					indices = polyfem::FEBasis3d::tet_face_local_nodes(discr_order, mesh, index2);
				}
				else
					continue;

				polyfem::InterfaceData data;
				data.local_indices.insert(data.local_indices.begin(), indices.data(), indices.data() + indices.size());
				assert(indices.size() == data.local_indices.size());
				poly_face_to_data[index2.face] = data;
			}
		}
	}

} // anonymous namespace

Eigen::VectorXi polyfem::FEBasis3d::tet_face_local_nodes(const int p, const Mesh3D &mesh, Navigation3D::Index index)
{
	const int nn = p > 2 ? (p - 2) : 0;
	const int n_edge_nodes = (p - 1) * 6;
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
	ev.row(0) << l2g[0], l2g[1];
	ev.row(1) << l2g[1], l2g[2];
	ev.row(2) << l2g[2], l2g[0];

	ev.row(3) << l2g[0], l2g[3];
	ev.row(4) << l2g[1], l2g[3];
	ev.row(5) << l2g[2], l2g[3];

	Navigation3D::Index tmp = index;

	for (int le = 0; le < e.rows(); ++le)
	{
		// const auto index =  find_edge(mesh, c, ev(le, 0), ev(le, 1));
		const auto l_index = mesh.get_index_from_element_edge(c, ev(le, 0), ev(le, 1));
		e[le] = l_index;
	}

	int ii = 3;
	for (int k = 0; k < 3; ++k)
	{
		bool reverse = false;
		int le = 0;
		for (; le < ev.rows(); ++le)
		{
			// const auto l_index =  find_edge(mesh, c, ev(le, 0), ev(le, 1));
			// const auto l_index = mesh.get_index_from_element_edge(c, ev(le, 0), ev(le, 1));
			const auto l_index = e[le];
			if (l_index.edge == tmp.edge)
			{
				if (l_index.vertex == tmp.vertex)
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

		if (!reverse)
		{

			for (int i = 0; i < p - 1; ++i)
			{
				result[ii++] = 4 + le * (p - 1) + i;
			}
		}
		else
		{
			for (int i = 0; i < p - 1; ++i)
			{
				result[ii++] = 4 + (le + 1) * (p - 1) - i - 1;
			}
		}

		tmp = mesh.next_around_face(tmp);
	}

	// faces

	Eigen::Matrix<int, 4, 3> fv;
	fv.row(0) << l2g[0], l2g[1], l2g[2];
	fv.row(1) << l2g[0], l2g[1], l2g[3];
	fv.row(2) << l2g[1], l2g[2], l2g[3];
	fv.row(3) << l2g[2], l2g[0], l2g[3];

	long lf = 0;
	for (; lf < fv.rows(); ++lf)
	{
		const auto l_index = mesh.get_index_from_element_face(c, fv(lf, 0), fv(lf, 1), fv(lf, 2));
		if (l_index.face == index.face)
			break;
	}

	assert(lf < fv.rows());

	if (n_face_nodes == 0)
	{
	}
	else if (n_face_nodes == 1)
		result[ii++] = 4 + n_edge_nodes + lf;
	else // if (n_face_nodes == 3)
	{

		const auto get_order = [&p, &nn, &n_face_nodes](const std::array<int, 3> &corners) {
			int index;
			int start;
			int offset;

			std::vector<int> order1(n_face_nodes); // A-> B
			for (int k = 0; k < n_face_nodes; ++k)
				order1[k] = k;

			std::vector<int> order2(n_face_nodes); // B-> A
			index = 0;
			start = nn - 1;
			for (int k = 0; k < nn; ++k)
			{
				for (int l = 0; l < nn - k; ++l)
				{
					order2[index] = start - l;
					index++;
				}
				start += (nn - 1) - k;
			}

			std::vector<int> order3(n_face_nodes); // A->C
			index = 0;
			for (int k = 0; k < nn; ++k)
			{
				offset = k;
				for (int l = 0; l < nn - k; ++l)
				{
					order3[index] = offset;
					offset += nn - l;

					index++;
				}
			}

			std::vector<int> order4(n_face_nodes); // C-> A
			index = 0;
			start = n_face_nodes - 1;
			for (int k = 0; k < nn; ++k)
			{
				offset = 0;
				for (int l = 0; l < nn - k; ++l)
				{
					order4[index] = start - offset;
					offset += k + 2 + l;
					index++;
				}

				start += -k - 1;
			}

			std::vector<int> order5(n_face_nodes); // B-> C
			index = 0;
			start = nn - 1;
			for (int k = 0; k < nn; ++k)
			{
				offset = 0;
				for (int l = 0; l < nn - k; ++l)
				{
					order5[index] = start + offset;
					offset += nn - 1 - l;
					index++;
				}

				start--;
			}

			std::vector<int> order6(n_face_nodes); // C-> B
			index = 0;
			start = n_face_nodes;
			for (int k = 0; k < nn; ++k)
			{
				offset = 0;
				start = start - k - 1;
				for (int l = 0; l < nn - k; ++l)
				{
					order6[index] = start - offset;
					offset += l + 1 + k;
					index++;
				}
			}

			if (corners[0] == order1[0] && corners[1] == order1[nn - 1])
			{
				assert(corners[2] == order1[n_face_nodes - 1]);
				return order1;
			}

			if (corners[0] == order2[0] && corners[1] == order2[nn - 1])
			{
				assert(corners[2] == order2[n_face_nodes - 1]);
				return order2;
			}

			if (corners[0] == order3[0] && corners[1] == order3[nn - 1])
			{
				assert(corners[2] == order3[n_face_nodes - 1]);
				return order3;
			}

			if (corners[0] == order4[0] && corners[1] == order4[nn - 1])
			{
				assert(corners[2] == order4[n_face_nodes - 1]);
				return order4;
			}

			if (corners[0] == order5[0] && corners[1] == order5[nn - 1])
			{
				assert(corners[2] == order5[n_face_nodes - 1]);
				return order5;
			}

			if (corners[0] == order6[0] && corners[1] == order6[nn - 1])
			{
				assert(corners[2] == order6[n_face_nodes - 1]);
				return order6;
			}

			assert(false);
			return order1;
		};

		Eigen::MatrixXd nodes;
		autogen::p_nodes_3d(p, nodes);
		// auto pos = polyfem::FEBasis3d::linear_tet_face_local_nodes_coordinates(mesh, index);
		// Local to global mapping of node indices

		// Extract requested interface
		std::array<int, 3> idx;
		for (int lv = 0; lv < 3; ++lv)
		{
			idx[lv] = find_index(l2g.begin(), l2g.end(), index.vertex);
			index = mesh.next_around_face(index);
		}
		Eigen::Matrix3d pos(3, 3);
		int cnt = 0;
		for (int i : idx)
		{
			pos.row(cnt++) = nodes.row(i);
		}

		const Eigen::RowVector3d bary = pos.colwise().mean();

		const int offset = 4 + n_edge_nodes;
		bool found = false;
		for (int lff = 0; lff < 4; ++lff)
		{
			Eigen::MatrixXd loc_nodes = nodes.block(offset + lff * n_face_nodes, 0, n_face_nodes, 3);
			Eigen::RowVector3d node_bary = loc_nodes.colwise().mean();

			if ((node_bary - bary).norm() < 1e-10)
			{
				std::array<int, 3> corners;
				int sum = 0;
				for (int m = 0; m < 3; ++m)
				{
					auto t = pos.row(m);
					int min_n = -1;
					double min_dis = 10000;

					for (int n = 0; n < n_face_nodes; ++n)
					{
						double dis = (loc_nodes.row(n) - t).squaredNorm();
						if (dis < min_dis)
						{
							min_dis = dis;
							min_n = n;
						}
					}

					assert(min_n >= 0);
					assert(min_n < n_face_nodes);
					corners[m] = min_n;
				}

				const auto indices = get_order(corners);
				for (int min_n : indices)
				{
					sum += min_n;
					result[ii++] = 4 + n_edge_nodes + min_n + lf * n_face_nodes;
				}

				assert(sum == (n_face_nodes - 1) * n_face_nodes / 2);

				found = true;
				assert(lff == lf);

				break;
			}
		}

		assert(found);
	}
	// else
	// {
	// 	assert(n_face_nodes == 0);
	// }

	assert(ii == result.size());
	return result;
}

Eigen::VectorXi polyfem::FEBasis3d::hex_face_local_nodes(const bool serendipity, const int q, const Mesh3D &mesh, Navigation3D::Index index)
{
	const int nn = q - 1;
	const int n_edge_nodes = nn * 12;
	const int n_face_nodes = serendipity ? 0 : nn * nn;

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
	ev.row(0) << l2g[0], l2g[1];
	ev.row(1) << l2g[1], l2g[2];
	ev.row(2) << l2g[2], l2g[3];
	ev.row(3) << l2g[3], l2g[0];
	ev.row(4) << l2g[0], l2g[4];
	ev.row(5) << l2g[1], l2g[5];
	ev.row(6) << l2g[2], l2g[6];
	ev.row(7) << l2g[3], l2g[7];
	ev.row(8) << l2g[4], l2g[5];
	ev.row(9) << l2g[5], l2g[6];
	ev.row(10) << l2g[6], l2g[7];
	ev.row(11) << l2g[7], l2g[4];

	Navigation3D::Index tmp = index;

	for (int le = 0; le < e.rows(); ++le)
	{
		const auto l_index = mesh.get_index_from_element_edge(c, ev(le, 0), ev(le, 1));
		e[le] = l_index;
	}

	int ii = 4;
	for (int k = 0; k < 4; ++k)
	{
		bool reverse = false;
		int le = 0;
		for (; le < ev.rows(); ++le)
		{
			// const auto l_index =  find_edge(mesh, c, ev(le, 0), ev(le, 1));
			// const auto l_index = mesh.get_index_from_element_edge(c, ev(le, 0), ev(le, 1));
			const auto l_index = e[le];
			if (l_index.edge == tmp.edge)
			{
				if (l_index.vertex == tmp.vertex)
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

		if (!reverse)
		{

			for (int i = 0; i < q - 1; ++i)
			{
				result[ii++] = 8 + le * (q - 1) + i;
			}
		}
		else
		{
			for (int i = 0; i < q - 1; ++i)
			{
				result[ii++] = 8 + (le + 1) * (q - 1) - i - 1;
			}
		}

		tmp = mesh.next_around_face(tmp);
	}

	// faces

	Eigen::Matrix<int, 6, 4> fv;
	fv.row(0) << l2g[0], l2g[3], l2g[4], l2g[7];
	fv.row(1) << l2g[1], l2g[2], l2g[5], l2g[6];
	fv.row(2) << l2g[0], l2g[1], l2g[5], l2g[4];
	fv.row(3) << l2g[3], l2g[2], l2g[6], l2g[7];
	fv.row(4) << l2g[0], l2g[1], l2g[2], l2g[3];
	fv.row(5) << l2g[4], l2g[5], l2g[6], l2g[7];

	long lf = 0;
	for (; lf < fv.rows(); ++lf)
	{
		const auto l_index = find_quad_face(mesh, c, fv(lf, 0), fv(lf, 1), fv(lf, 2), fv(lf, 3));
		if (l_index.face == index.face)
			break;
	}

	assert(lf < fv.rows());

	if (n_face_nodes == 1)
		result[ii++] = 8 + n_edge_nodes + lf;
	else if (n_face_nodes != 0)
	{
		Eigen::MatrixXd nodes;
		autogen::q_nodes_3d(q, nodes);
		// auto pos = polyfem::FEBasis3d::linear_tet_face_local_nodes_coordinates(mesh, index);
		// Local to global mapping of node indices

		// Extract requested interface
		std::array<int, 4> idx;
		for (int lv = 0; lv < 4; ++lv)
		{
			idx[lv] = find_index(l2g.begin(), l2g.end(), index.vertex);
			index = mesh.next_around_face(index);
		}
		Eigen::Matrix<double, 4, 3> pos(4, 3);
		int cnt = 0;
		for (int i : idx)
		{
			pos.row(cnt++) = nodes.row(i);
		}

		const Eigen::RowVector3d bary = pos.colwise().mean();

		const int offset = 8 + n_edge_nodes;
		bool found = false;
		for (int lff = 0; lff < 6; ++lff)
		{
			Eigen::Matrix<double, 4, 3> loc_nodes = nodes.block<4, 3>(offset + lff * n_face_nodes, 0);
			Eigen::RowVector3d node_bary = loc_nodes.colwise().mean();

			if ((node_bary - bary).norm() < 1e-10)
			{
				int sum = 0;
				for (int m = 0; m < 4; ++m)
				{
					auto t = pos.row(m);
					int min_n = -1;
					double min_dis = 10000;

					for (int n = 0; n < 4; ++n)
					{
						double dis = (loc_nodes.row(n) - t).squaredNorm();
						if (dis < min_dis)
						{
							min_dis = dis;
							min_n = n;
						}
					}

					assert(min_n >= 0);
					assert(min_n < 4);

					sum += min_n;

					result[ii++] = 8 + n_edge_nodes + min_n + lf * n_face_nodes;
				}

				assert(sum == 6); // 0 + 1 + 2 + 3

				found = true;
				assert(lff == lf);
			}

			if (found)
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
	const bool serendipity,
	const bool has_polys,
	const bool is_geom_bases,
	std::vector<ElementBases> &bases,
	std::vector<LocalBoundary> &local_boundary,
	std::map<int, InterfaceData> &poly_face_to_data)
{
	Eigen::VectorXi discr_orders(mesh.n_cells());
	discr_orders.setConstant(discr_order);

	return build_bases(mesh, quadrature_order, discr_orders, serendipity, has_polys, is_geom_bases, bases, local_boundary, poly_face_to_data);
}

int polyfem::FEBasis3d::build_bases(
	const Mesh3D &mesh,
	const int quadrature_order,
	const Eigen::VectorXi &discr_orders,
	const bool serendipity,
	const bool has_polys,
	const bool is_geom_bases,
	std::vector<ElementBases> &bases,
	std::vector<LocalBoundary> &local_boundary,
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
	// assert(max_p < 5); //P5 not supported

	const int nn = max_p > 1 ? (max_p - 1) : 0;
	const int n_face_nodes = nn * nn;
	const int n_cells_nodes = nn * nn * nn;

	if (mesh.ncmesh) {
		for (int c = 0; c < mesh.n_cells(); c++) {
			const auto& verts = mesh.ncmesh->elements[mesh.ncmesh->valid2All(c)].vertices;
			auto v = tet_vertices_local_to_global(mesh, c);
			for (int i = 0; i < v.size(); i++)
				v[i] = mesh.ncmesh->valid2AllVertex(v[i]);
			for (int i = 0; i < v.size(); i++) {
				if (v[i] != verts[i]) {
					logger().trace("Different local vertex indexing of element {}", mesh.ncmesh->valid2All(c));
					break;
				}
			}
			for (int i = 0; i < v.size(); i++) {
				bool flag = false;
				for (int j = 0; j < verts.size(); j++) {
					if (verts[j] == v[i])
						flag = true;
				}
				assert(flag);
			}
			auto& verts_ = mesh.ncmesh->elements[mesh.ncmesh->valid2All(c)].vertices;
			for (int i = 0; i < v.size(); i++)
				verts_[i] = v[i];
		}
	}

	MeshNodes nodes(mesh, has_polys, !is_geom_bases, nn, n_face_nodes * (is_geom_bases ? 2 : 1), max_p == 0 ? 1 : n_cells_nodes);
	std::vector<std::vector<int>> element_nodes_id;
	compute_nodes(mesh, discr_orders, serendipity, has_polys, is_geom_bases, nodes, element_nodes_id, local_boundary, poly_face_to_data);
	// boundary_nodes = nodes.boundary_nodes();

	// std::cout<<"get_index_from_element_face_time " << Navigation3D::get_index_from_element_face_time <<std::endl;
	// std::cout<<"switch_vertex_time " << Navigation3D::switch_vertex_time <<std::endl;
	// std::cout<<"switch_edge_time " << Navigation3D::switch_edge_time <<std::endl;
	// std::cout<<"switch_face_time " << Navigation3D::switch_face_time <<std::endl;
	// std::cout<<"switch_element_time " << Navigation3D::switch_element_time <<std::endl;

	bases.resize(mesh.n_cells());
	std::vector<int> interface_elements;
	interface_elements.reserve(mesh.n_faces());

	for (int e = 0; e < mesh.n_cells(); ++e)
	{
		ElementBases &b = bases[e];
		const int discr_order = discr_orders(e);
		const int n_el_bases = (int)element_nodes_id[e].size();
		b.bases.resize(n_el_bases);

		bool skip_interface_element = false;

		for (int j = 0; j < n_el_bases; ++j)
		{
			const int global_index = element_nodes_id[e][j];
			if (global_index < 0)
			{
				skip_interface_element = true;
				break;
			}
		}

		if (skip_interface_element)
		{
			interface_elements.push_back(e);
		}

		if (mesh.is_cube(e))
		{
			const int real_order = std::max(quadrature_order, (discr_order - 1) * 2 + 1);
			// hex_quadrature.get_quadrature(real_order, b.quadrature);
			b.set_quadrature([real_order](Quadrature &quad) {
				HexQuadrature hex_quadrature;
				hex_quadrature.get_quadrature(real_order, quad);
			});

			b.set_local_node_from_primitive_func([serendipity, discr_order, e](const int primitive_id, const Mesh &mesh) {
				const auto &mesh3d = dynamic_cast<const Mesh3D &>(mesh);
				Navigation3D::Index index;

				for (int lf = 0; lf < 6; ++lf)
				{
					index = mesh3d.get_index_from_element(e, lf, 0);
					if (index.face == primitive_id)
						break;
				}
				assert(index.face == primitive_id);
				return hex_face_local_nodes(serendipity, discr_order, mesh3d, index);
			});

			for (int j = 0; j < n_el_bases; ++j)
			{
				const int global_index = element_nodes_id[e][j];

				b.bases[j].init(discr_order, global_index, j, nodes.node_position(global_index));

				const int dtmp = serendipity ? -2 : discr_order;

				b.bases[j].set_basis([dtmp, j](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { polyfem::autogen::q_basis_value_3d(dtmp, j, uv, val); });
				b.bases[j].set_grad([dtmp, j](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { polyfem::autogen::q_grad_basis_value_3d(dtmp, j, uv, val); });
			}
		}
		else if (mesh.is_simplex(e))
		{
			const int real_order = std::max(quadrature_order, (discr_order - 1) * 2 + 1);

			b.set_quadrature([real_order](Quadrature &quad) {
				TetQuadrature tet_quadrature;
				tet_quadrature.get_quadrature(real_order, quad);
			});

			b.set_local_node_from_primitive_func([discr_order, e](const int primitive_id, const Mesh &mesh) {
				const auto &mesh3d = dynamic_cast<const Mesh3D &>(mesh);
				Navigation3D::Index index;

				for (int lf = 0; lf < mesh3d.n_cell_faces(e); ++lf)
				{
					index = mesh3d.get_index_from_element(e, lf, 0);
					if (index.face == primitive_id)
						break;
				}
				assert(index.face == primitive_id);
				return tet_face_local_nodes(discr_order, mesh3d, index);
			});

			const bool rational = is_geom_bases && mesh.is_rational() && !mesh.cell_weights(e).empty();
			assert(!rational);

			for (int j = 0; j < n_el_bases; ++j)
			{
				const int global_index = element_nodes_id[e][j];
				if (!skip_interface_element)
				{
					b.bases[j].init(discr_order, global_index, j, nodes.node_position(global_index));
				}

				b.bases[j].set_basis([discr_order, j](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { polyfem::autogen::p_basis_value_3d(discr_order, j, uv, val); });
				b.bases[j].set_grad([discr_order, j](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { polyfem::autogen::p_grad_basis_value_3d(discr_order, j, uv, val); });
			}
		}
		else
		{
			// Polyhedra bases are built later on
			// assert(false);
		}
	}

	if (!is_geom_bases)
	{
		if (mesh.ncmesh) {
			
			// static int write_num = 0;
			// if (mesh.ncmesh->n_elements > 50) {
			// 	mesh.ncmesh->writeRefineHistory("history"+std::to_string(write_num)+".log");
			// 	mesh.ncmesh->writeOrders("order"+std::to_string(write_num)+".log");
			// 	write_num++;
			// }

			auto evaluate_boundary_bases = [](const int p, const ElementBases &base, const Eigen::MatrixXd &uv, std::vector<AssemblyValues> &w, int type = 2)
			{
				int i = 0;
				assert(p > 0);
				int max_num;
				if (type == 0)
					max_num = 4;
				else if (type == 1)
					max_num = 4 + 6 * (p - 1);
				else if (type >= 2)
					max_num = 4 + 6 * (p - 1) + 2 * (p - 1) * (p - 2);
				w.resize(max_num);
				maybe_parallel_for(max_num, [&](int start, int end, int thread_id) {
					for (int i = start; i < end; i++) {
						base.bases[i].eval_basis(uv, w[i].val);
						assert(i < max_num || abs(w[i].val(0)) < 1e-12);
						assert(w[i].val.size() == uv.rows());
					}
				});
			};

			auto &ncmesh = *dynamic_cast<ncMesh3D *>(mesh.ncmesh.get());
			// std::vector<int> elementOrder;
			// ncmesh.reorderElements(elementOrder);
			std::vector<std::vector<int> > elementOrder;
			ncmesh.reorderElements(elementOrder);

			for (const auto& bucket : elementOrder) {
				if (bucket.size() == 0)
					continue;
				maybe_parallel_for(bucket.size(), [&](int start, int end, int thread_id) {
				for (int e_aux = start; e_aux < end; e_aux++) {
					const int e = bucket[e_aux];
					const int e_ = ncmesh.all2Valid(e);
					ElementBases& b = bases[e_];
					const int discr_order = discr_orders(e_);
					const int n_edge_nodes = discr_order - 1;
					const int n_face_nodes = (discr_order - 1) * (discr_order - 2) / 2;
					const int n_el_bases = element_nodes_id[e_].size();

					auto v = tet_vertices_local_to_global(mesh, e_);
					for (int i = 0; i < v.size(); i++)
						v[i] = ncmesh.valid2AllVertex(v[i]);

					Eigen::Matrix<int, 6, 2> ev;
					ev.row(0) << v[0], v[1];
					ev.row(1) << v[1], v[2];
					ev.row(2) << v[2], v[0];
					ev.row(3) << v[0], v[3];
					ev.row(4) << v[1], v[3];
					ev.row(5) << v[2], v[3];

					Eigen::Matrix<int, 6, 2> ev_;
					ev_.row(0) << 0, 1;
					ev_.row(1) << 1, 2;
					ev_.row(2) << 2, 0;
					ev_.row(3) << 0, 3;
					ev_.row(4) << 1, 3;
					ev_.row(5) << 2, 3;

					Eigen::Matrix<int, 4, 3> fv;
					fv.row(0) << v[0], v[1], v[2];
					fv.row(1) << v[0], v[1], v[3];
					fv.row(2) << v[1], v[2], v[3];
					fv.row(3) << v[2], v[0], v[3];

					Eigen::Matrix<int, 4, 3> fv_;
					fv_.row(0) << 0, 1, 2;
					fv_.row(1) << 0, 1, 3;
					fv_.row(2) << 1, 2, 3;
					fv_.row(3) << 2, 0, 3;

					for (int i = 0; i < ncmesh.elements[e].vertices.size(); i++)
						assert(v[i] == ncmesh.elements[e].vertices(i));

					for (int j = 0; j < n_el_bases; j++)
					{
						const int global_index = element_nodes_id[e_][j];

						if (global_index >= 0)
							b.bases[j].init(discr_order, global_index, j, nodes.node_position(global_index));
						else
						{
							const auto& ncelem = ncmesh.elements[e];
							
							// vertex node - hanging vertex
							if (j < 4) {
								const auto& vert = ncmesh.vertices[v[j]];
								int large_elem = -1;
								if (vert.edge >= 0) {
									large_elem = ncmesh.getLowestOrderElementOnEdge(vert.edge);
									// large_elem = ncmesh.edges[vert.edge].get_element();
								}
								else if (vert.face >= 0) {
									large_elem = ncmesh.faces[vert.face].get_element();
								}
								else
									assert(false);

								Eigen::MatrixXd node_position = vert.pos.transpose();
								ncmesh.global2Local(large_elem, node_position);

								// evaluate the basis of the large element at this node
								const auto& other_bases = bases[ncmesh.all2Valid(large_elem)];
								std::vector<AssemblyValues> w;
								// other_bases.evaluate_bases(node_position, w);
								evaluate_boundary_bases(ncmesh.elements[large_elem].order, other_bases, node_position, w);

								// apply basis projection
								for (long i = 0; i < w.size(); ++i)
								{
									assert(w[i].val.size() == 1);
									if (std::abs(w[i].val(0)) < 1e-12)
										continue;
									// else
									// 	std::cout << "elem "<<e<<" node "<<j<<" -> elem "<<large_elem<<" node "<<i<<", w = " << w[i].val(0) << std::endl;

									assert(other_bases.bases[i].global().size() > 0);
									for (size_t ii = 0; ii < other_bases.bases[i].global().size(); ++ii)
									{
										const auto& other_global = other_bases.bases[i].global()[ii];
										assert(other_global.index >= 0);
										b.bases[j].global().emplace_back(other_global.index, other_global.node, w[i].val(0) * other_global.val);
									}
								}
							}
							// edge node - slave edge / edge on face / constrained order
							else if (j < 4 + 6 * n_edge_nodes) {
								const int local_edge_id = (j - 4) / n_edge_nodes;
								const int edge_id = ncmesh.findEdge(ev.row(local_edge_id));
								const auto& edge = ncmesh.edges[edge_id];
								bool need_extra_fake_nodes = false;
								int large_elem = -1;

								// slave edge
								if (edge.master >= 0) {
									// large_elem = ncmesh.getLowestOrderElementOnEdge(edge.master);
									large_elem = ncmesh.edges[edge.master].get_element();
								}
								// edge on face
								else if (edge.master_face >= 0) {
									large_elem = ncmesh.faces[edge.master_face].get_element();
								}
								// constrained order
								else if (edge.order < ncelem.order) {
									int min_order_elem = ncmesh.getLowestOrderElementOnEdge(edge_id);
									// if haven't built min_order_elem? directly contribute to extra nodes
									if (ncmesh.elements[min_order_elem].order < ncelem.order)
										large_elem = min_order_elem;

									// constrained order, master edge -- need extra fake nodes
									if (large_elem < 0) {
										// assert((edge.order < 2 || edge.global_ids.size() > 0) && edge.slaves.size() > 0);
										need_extra_fake_nodes = true;
									}
								}
								else
									assert(false);

								assert(large_elem >= 0 || need_extra_fake_nodes);
								Eigen::MatrixXd lnodes;
								autogen::p_nodes_3d(discr_order, lnodes);
								Eigen::MatrixXd node_position = lnodes.row(j);
								if (need_extra_fake_nodes) {
									logger().trace("Need extra fake nodes for ncelem {} ncedge {}", e, edge_id);
									// use a global way
									// double edge_weight = ncMesh3D::elemWeight2edgeWeight(local_edge_id, node_position.transpose());

									Eigen::MatrixXd point_weight;
									ncmesh.local2Global(e, node_position);
									ncmesh.global2LocalEdge(edge_id, node_position, point_weight);

									std::function<double(const int, const int, const double)> basis_1d = [](const int order, const int id, const double x) -> double {
										assert(id <= order && id >= 0);
										double y = 1;
										for (int o = 0; o <= order; o++) {
											if (o != id)
												y *= (x * order - o) / (id - o);
										}
										return y;
									};
									
									// contribution to edge nodes
									for (int i = 0; i < edge.global_ids.size(); i++) {
										const int global_index = edge.global_ids[i];
										// const double weight = basis_1d(edge.order, i+1, edge_weight);
										Eigen::MatrixXd node_weight;
										ncmesh.global2LocalEdge(edge_id, nodes.node_position(global_index), node_weight);
										const int basis_id = std::lround(node_weight(0) * edge.order);
										const double weight = basis_1d(edge.order, basis_id, point_weight(0));
										if (std::abs(weight) < 1e-12)
											continue;
										b.bases[j].global().emplace_back(global_index, nodes.node_position(global_index), weight);
									}

									// contribution to vertex nodes
									for (int i = 0; i < 2; i++) {
										const int lv = ev_(local_edge_id, i);
										const auto& global_ = b.bases[lv].global();
										// double weight = basis_1d(edge.order, (i == 1) ? edge.order : 0, edge_weight);
										Eigen::MatrixXd node_weight;
										ncmesh.global2LocalEdge(edge_id, ncmesh.vertices[ncelem.vertices(lv)].pos.transpose(), node_weight);
										const int basis_id = std::lround(node_weight(0) * edge.order);
										const double weight = basis_1d(edge.order, basis_id, point_weight(0));
										if (std::abs(weight) > 1e-12) {
											assert(global_.size() > 0);
											for (size_t ii = 0; ii < global_.size(); ++ii)
												b.bases[j].global().emplace_back(global_[ii].index, global_[ii].node, weight * global_[ii].val);
										}
									}
								}
								else {
									ncmesh.local2Global(e, node_position);
									ncmesh.global2Local(large_elem, node_position);

									// evaluate the basis of the large element at this node
									const auto& other_bases = bases[ncmesh.all2Valid(large_elem)];
									std::vector<AssemblyValues> w;
									// other_bases.evaluate_bases(node_position, w);
									evaluate_boundary_bases(ncmesh.elements[large_elem].order, other_bases, node_position, w);

									// apply basis projection
									for (long i = 0; i < w.size(); ++i)
									{
										assert(w[i].val.size() == 1);
										if (std::abs(w[i].val(0)) < 1e-12)
											continue;
										// else
										// 	std::cout << "elem "<<e<<" node "<<j<<" -> elem "<<large_elem<<" node "<<i<<", w = " << w[i].val(0) << std::endl;

										assert(other_bases.bases[i].global().size() > 0);
										for (size_t ii = 0; ii < other_bases.bases[i].global().size(); ++ii)
										{
											const auto& other_global = other_bases.bases[i].global()[ii];
											assert(other_global.index >= 0);
											b.bases[j].global().emplace_back(other_global.index, other_global.node, w[i].val(0) * other_global.val);
										}
									}
								}
							}
							// face node - slave face / constrained order
							else if (j < 4 + 6 * n_edge_nodes + 4 * n_face_nodes) {
								const int local_face_id = (j - (4 + 6 * n_edge_nodes)) / n_face_nodes;
								const int face_id = ncmesh.findFace(fv.row(local_face_id));
								const auto& ncface = ncmesh.faces[face_id];
								int large_elem = -1;
								bool need_extra_fake_nodes = false;

								// slave face
								if (ncface.master >= 0) {
									large_elem = ncmesh.faces[ncface.master].get_element();
								}
								// constrained order, conforming face
								else if (ncface.order < ncelem.order && ncface.n_elem() == 2) {
									large_elem = ncface.find_opposite_element(e);
								}
								// constrained order, master face -- need extra fake nodes
								else if (ncface.order < ncelem.order && ncface.slaves.size() > 0) {
									// assert(ncface.global_ids.size() > 0 || ncface.order < 3);
									need_extra_fake_nodes = true;
								}
								else
									assert(false);

								assert(large_elem >= 0 || need_extra_fake_nodes);
								Eigen::MatrixXd lnodes;
								autogen::p_nodes_3d(discr_order, lnodes);
								Eigen::MatrixXd node_position = lnodes.row(j);
								if (need_extra_fake_nodes) {
									logger().trace("Need extra fake nodes for ncelem {} ncface {}", e, face_id);
									// auto face_weight = ncMesh3D::elemWeight2faceWeight(local_face_id, node_position.transpose());
									ncmesh.local2Global(e, node_position);
									Eigen::MatrixXd tmp;
									ncmesh.global2LocalFace(face_id, node_position, tmp);
									Eigen::VectorXd face_weight = tmp.transpose();

									std::function<double(const int, const int, const double)> basis_aux = [](const int order, const int id, const double x) -> double {
										assert(id <= order && id >= 0);
										double y = 1;
										for (int o = 0; o < id; o++)
											y *= (x * order - o) / (id - o);
										return y;
									};

									std::function<double(const int, const int, const int, const Eigen::Vector2d)> basis_2d = [&basis_aux](const int order, const int i, const int j, const Eigen::Vector2d uv) -> double {
										assert(i + j <= order && i >= 0 && j >= 0);
										double u = uv(0), v = uv(1);
										return basis_aux(order, i, u) * basis_aux(order, j, v) * basis_aux(order, order - i - j, 1 - u - v);
									};

									// contribution to face nodes
									for (int global_ : ncface.global_ids) {
										auto low_order_node = nodes.node_position(global_);
										Eigen::MatrixXd low_order_node_face_weight;
										ncmesh.global2LocalFace(face_id, low_order_node, low_order_node_face_weight);
										int x = round(low_order_node_face_weight(0) * ncface.order), y = round(low_order_node_face_weight(1) * ncface.order);
										const double weight = basis_2d(ncface.order, x, y, face_weight);
										if (std::abs(weight) < 1e-12)
											continue;
										b.bases[j].global().emplace_back(global_, nodes.node_position(global_), weight);
									}

									// contribution to vertex nodes
									for (int i = 0; i < 3; i++) {
										const auto& global_ = b.bases[fv_(local_face_id, i)].global();
										auto low_order_node = ncmesh.vertices[fv(local_face_id, i)].pos;
										Eigen::MatrixXd low_order_node_face_weight;
										ncmesh.global2LocalFace(face_id, low_order_node.transpose(), low_order_node_face_weight);
										int x = round(low_order_node_face_weight(0) * ncface.order), y = round(low_order_node_face_weight(1) * ncface.order);
										double weight = basis_2d(ncface.order, x, y, face_weight);
										if (std::abs(weight) > 1e-12) {
											assert(global_.size() > 0);
											for (size_t ii = 0; ii < global_.size(); ++ii)
												b.bases[j].global().emplace_back(global_[ii].index, global_[ii].node, weight * global_[ii].val);
										}
									}

									// contribution to edge nodes, two steps
									for (int x = 0, idx = 0; x <= ncface.order; x++) {
										for (int y = 0; x + y <= ncface.order; y++) {
											const int z = ncface.order - x - y;
											int flag = (int)(x == 0) + (int)(y == 0) + (int)(z == 0);
											if (flag != 1)
												continue;

											// first step
											const double weight = basis_2d(ncface.order, x, y, face_weight);
											if (std::abs(weight) < 1e-12)
												continue;
											Eigen::MatrixXd face_weight(1, 2);
											face_weight << (double) x / ncface.order, (double) y / ncface.order;
											Eigen::MatrixXd pos;
											ncmesh.local2GlobalFace(face_id, face_weight, pos);
											ncmesh.global2Local(e, pos);
											// RowVectorNd pos = (ncMesh3D::faceWeight2ElemWeight(local_face_id, Eigen::Vector2d((double)x / ncface.order, (double)y / ncface.order))).transpose();
											Local2Global step1(idx, pos, weight);
											idx++;

											// second step
											// int le = -1;
											// if (x == 0) le = 2;
											// else if (z == 0) le = 1;
											// else if (y == 0) le = 0;
											// else assert(false);

											// int ge = ncmesh.findEdge(fv(local_face_id, le), fv(local_face_id, (le+1)%3));
											// const auto& edge = ncmesh.edges[ge];
											// if (edge.master >= 0) {
											// 	large_elem = ncmesh.getLowestOrderElementOnEdge(edge.master);
											// }
											// else if (edge.master_face >= 0) {
											// 	large_elem = ncmesh.faces[edge.master_face].get_element();
											// }
											// else {
											// 	// TODO: if haven't built min_order_elem?
											// 	int min_order_elem = ncmesh.getLowestOrderElementOnEdge(ge);
											// 	if (ncmesh.elements[min_order_elem].order < ncelem.order)
											// 		large_elem = min_order_elem;
											// 	else
											// 		large_elem = e;
											// }

											Eigen::MatrixXd pos_ = pos;
											// ncmesh.local2Global(e, pos_);
											// ncmesh.global2Local(large_elem, pos_);
											{
												// evaluate the basis of the large element at this node
												const auto& other_bases = bases[ncmesh.all2Valid(e)];
												std::vector<AssemblyValues> w;
												// other_bases.evaluate_bases(pos_, w);
												evaluate_boundary_bases(ncmesh.elements[e].order, other_bases, pos_, w, 1);

												// apply basis projection
												for (long i = 0; i < w.size(); ++i)
												{
													assert(w[i].val.size() == 1);
													if (std::abs(w[i].val(0)) < 1e-12)
														continue;

													assert(other_bases.bases[i].global().size() > 0);
													for (size_t ii = 0; ii < other_bases.bases[i].global().size(); ++ii)
													{
														const auto& other_global = other_bases.bases[i].global()[ii];
														assert(other_global.index >= 0);
														b.bases[j].global().emplace_back(other_global.index, other_global.node, step1.val * w[i].val(0) * other_global.val);
													}
												}
											}
										}
									}
								}
								else {
									ncmesh.local2Global(e, node_position);
									ncmesh.global2Local(large_elem, node_position);

									// evaluate the basis of the large element at this node
									const auto& other_bases = bases[ncmesh.all2Valid(large_elem)];
									std::vector<AssemblyValues> w;
									// other_bases.evaluate_bases(node_position, w);
									evaluate_boundary_bases(ncmesh.elements[large_elem].order, other_bases, node_position, w);

									// apply basis projection
									for (long i = 0; i < w.size(); ++i)
									{
										assert(w[i].val.size() == 1);
										if (std::abs(w[i].val(0)) < 1e-12)
											continue;

										assert(other_bases.bases[i].global().size() > 0);
										for (size_t ii = 0; ii < other_bases.bases[i].global().size(); ++ii)
										{
											const auto& other_global = other_bases.bases[i].global()[ii];
											assert(other_global.index >= 0);
											b.bases[j].global().emplace_back(other_global.index, other_global.node, w[i].val(0) * other_global.val);
										}
									}
								}						
							}
							else
								assert(false);
						
							auto& global_ = b.bases[j].global();
							if (global_.size() <= 1)
								continue;

							std::map<int, Local2Global> list;
							for (size_t ii = 0; ii < global_.size(); ii++) {
								auto pair = list.insert({global_[ii].index, global_[ii]});
								if (!pair.second) {
									assert((pair.first->second.node - global_[ii].node).norm() < 1e-12);
									pair.first->second.val += global_[ii].val;
								}
							}

							global_.clear();
							for (auto it = list.begin(); it != list.end(); ++it) {
								if (std::abs(it->second.val) > 1e-12)
									global_.push_back(it->second);
							}
						}
					}
				}
				});
			}
		}
		else {
			for (int pp = 2; pp <= autogen::MAX_P_BASES; ++pp)
			{
				for (int e : interface_elements)
				{
					ElementBases &b = bases[e];
					const int discr_order = discr_orders(e);
					const int n_el_bases = element_nodes_id[e].size();
					assert(discr_order > 1);
					if (discr_order != pp)
						continue;

					if (mesh.is_cube(e))
					{
						//TODO
						assert(false);
					}
					else if (mesh.is_simplex(e))
					{
						for (int j = 0; j < n_el_bases; ++j)
						{
							const int global_index = element_nodes_id[e][j];

							if (global_index >= 0)
								b.bases[j].init(discr_order, global_index, j, nodes.node_position(global_index));
							else
							{
								const int lnn = max_p > 2 ? (discr_order - 2) : 0;
								const int ln_edge_nodes = discr_order - 1;
								const int ln_face_nodes = lnn * (lnn + 1) / 2;

								const auto v = tet_vertices_local_to_global(mesh, e);
								Navigation3D::Index index;
								if (global_index <= -30)
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
									const auto le = -(global_index + 10);
									assert(le >= 0 && le < 6);
									assert(j >= 4 && j < 4 + 6 * ln_edge_nodes);

									Eigen::Matrix<int, 6, 2> ev;
									ev.row(0) << v[0], v[1];
									ev.row(1) << v[1], v[2];
									ev.row(2) << v[2], v[0];

									ev.row(3) << v[0], v[3];
									ev.row(4) << v[1], v[3];
									ev.row(5) << v[2], v[3];

									// const auto edge_index = find_edge(mesh, e, ev(le, 0), ev(le, 1));
									const auto edge_index = mesh.get_index_from_element_edge(e, ev(le, 0), ev(le, 1));
									auto neighs = mesh.edge_neighs(edge_index.edge);
									int min_p = discr_order;
									int min_cell = index.element;

									for (auto cid : neighs)
									{
										if (discr_orders[cid] < min_p)
										{
											min_p = discr_orders[cid];
											min_cell = cid;
										}
									}

									bool found = false;
									for (int lf = 0; lf < 4; ++lf)
									{
										for (int lv = 0; lv < 4; ++lv)
										{
											index = mesh.get_index_from_element(min_cell, lf, lv);

											if (index.vertex == edge_index.vertex)
											{
												if (index.edge != edge_index.edge)
												{
													auto tmp = index;
													index = mesh.switch_edge(tmp);

													if (index.edge != edge_index.edge)
													{
														index = mesh.switch_edge(mesh.switch_face(tmp));
													}
												}
												found = true;
												break;
											}
										}

										if (found)
											break;
									}

									assert(found);
									assert(index.vertex == edge_index.vertex && index.edge == edge_index.edge);
									assert(index.element != edge_index.element);
								}
								else
								{
									const auto lf = -(global_index + 1);
									assert(lf >= 0 && lf < 4);
									assert(j >= 4 + 6 * ln_edge_nodes && j < 4 + 6 * ln_edge_nodes + 4 * ln_face_nodes);

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
								Eigen::MatrixXd lnodes;
								autogen::p_nodes_3d(discr_order, lnodes);
								Eigen::RowVector3d node_position; // = lnodes.row(indices(ii));

								if (j < 4)
									node_position = lnodes.row(indices(0));
								else if (j < 4 + 6 * ln_edge_nodes)
									node_position = lnodes.row(indices(((j - 4) % ln_edge_nodes) + 3));
								else if (j < 4 + 6 * ln_edge_nodes + 4 * ln_face_nodes)
								{
									// node_position = lnodes.row(indices(((j - 4 - 6*ln_edge_nodes) % ln_face_nodes) + 3 + 3*ln_edge_nodes));
									auto me_indices = tet_face_local_nodes(discr_order, mesh, mesh.switch_element(index));
									int ii;
									for (ii = 0; ii < me_indices.size(); ++ii)
									{
										if (me_indices(ii) == j)
											break;
									}

									assert(ii >= 3 + 3 * ln_edge_nodes);
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

								for (long i = 0; i < w.size(); ++i)
								{
									assert(w[i].val.size() == 1);
									if (std::abs(w[i].val(0)) < 1e-12)
										continue;

									// assert(other_bases.bases[i].global().size() == 1);
									for (size_t ii = 0; ii < other_bases.bases[i].global().size(); ++ii)
									{
										const auto &other_global = other_bases.bases[i].global()[ii];
										// std::cout<<"e "<<e<<" " <<j << " gid "<<other_global.index<<std::endl;
										b.bases[j].global().emplace_back(other_global.index, other_global.node, w[i].val(0) * other_global.val);
									}
								}
							}
						}
					}
					else
					{
						// Polygon bases are built later on
					}
				}
			}
		}
	}

	return nodes.n_nodes();
}
