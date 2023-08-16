////////////////////////////////////////////////////////////////////////////////
#include "LagrangeBasis2d.hpp"

#include <polyfem/quadrature/TriQuadrature.hpp>
#include <polyfem/quadrature/QuadQuadrature.hpp>
#include <polyfem/autogen/auto_p_bases.hpp>
#include <polyfem/autogen/auto_q_bases.hpp>

#include <polyfem/assembler/AssemblerUtils.hpp>

#include <polyfem/utils/MaybeParallelFor.hpp>

#include <cassert>
#include <array>
////////////////////////////////////////////////////////////////////////////////

using namespace polyfem;
using namespace polyfem::assembler;
using namespace polyfem::basis;
using namespace polyfem::mesh;
using namespace polyfem::quadrature;
using namespace polyfem::utils;

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

namespace
{

	void sort(std::array<int, 2> &array)
	{
		if (array[0] > array[1])
		{
			std::swap(array[0], array[1]);
		}

		assert(array[0] <= array[1]);
	}

	template <class InputIterator, class T>
	int find_index(InputIterator first, InputIterator last, const T &val)
	{
		return std::distance(first, std::find(first, last, val));
	}

	Navigation::Index find_edge(const Mesh2D &mesh, int f, int v1, int v2)
	{
		std::array<int, 2> v = {{v1, v2}};
		std::array<int, 2> u;
		// std::sort(v.begin(), v.end());
		sort(v);
		auto idx = mesh.get_index_from_face(f, 0);
		for (int lv = 0; lv < mesh.n_face_vertices(idx.face); ++lv)
		{
			u[0] = idx.vertex;
			u[1] = mesh.switch_vertex(idx).vertex;
			// std::sort(u.begin(), u.end());
			sort(u);
			if (u == v)
			{
				if (idx.vertex != v1)
					idx = mesh.switch_vertex(idx);
				assert(idx.vertex == v1);
				return idx;
			}
			idx = mesh.next_around_face(idx);
		}
		throw std::runtime_error("Edge not found");
	}

	std::array<int, 3> tri_vertices_local_to_global(const Mesh2D &mesh, int f)
	{
		assert(mesh.is_simplex(f));

		// Vertex nodes
		std::array<int, 3> l2g;
		for (int lv = 0; lv < 3; ++lv)
		{
			l2g[lv] = mesh.face_vertex(f, lv);
		}

		return l2g;
	}

	std::array<int, 4> quad_vertices_local_to_global(const Mesh2D &mesh, int f)
	{
		assert(mesh.is_cube(f));

		// Vertex nodes
		std::array<int, 4> l2g;
		for (int lv = 0; lv < 4; ++lv)
		{
			l2g[lv] = mesh.face_vertex(f, lv);
		}

		return l2g;
	}

	void tri_local_to_global(const bool is_geom_bases, const int p, const Mesh2D &mesh, int f, const Eigen::VectorXi &discr_order, const Eigen::VectorXi &edge_orders, std::vector<int> &res, MeshNodes &nodes, std::vector<std::vector<int>> &edge_virtual_nodes)
	{
		int edge_offset = mesh.n_vertices();
		int face_offset = edge_offset + mesh.n_edges();

		const int n_edge_nodes = p > 1 ? ((p - 1) * 3) : 0;
		const int nn = p > 2 ? (p - 2) : 0;
		const int n_face_nodes = nn * (nn + 1) / 2;

		if (p == 0)
		{
			res.push_back(nodes.node_id_from_face(f));
			return;
		}

		// std::vector<int> res;
		res.reserve(3 + n_edge_nodes + n_face_nodes);

		// Vertex nodes
		auto v = tri_vertices_local_to_global(mesh, f);

		// Edge nodes
		Eigen::Matrix<Navigation::Index, 3, 1> e;
		Eigen::Matrix<int, 3, 2> ev;
		ev.row(0) << v[0], v[1];
		ev.row(1) << v[1], v[2];
		ev.row(2) << v[2], v[0];
		for (int le = 0; le < e.rows(); ++le)
		{
			const auto index = find_edge(mesh, f, ev(le, 0), ev(le, 1));
			e[le] = index;
		}

		for (size_t lv = 0; lv < v.size(); ++lv)
		{
			const auto index = e[lv];
			const auto other_face = mesh.switch_face(index).face;

			if (is_geom_bases)
				res.push_back(nodes.node_id_from_primitive(v[lv]));
			else
			{
				if (!mesh.is_conforming())
				{
					const auto &ncmesh = dynamic_cast<const NCMesh2D &>(mesh);
					// hanging vertex
					if (ncmesh.leader_edge_of_vertex(ncmesh.face_vertex(f, lv)) >= 0)
						res.push_back(-lv - 1);
					else
						res.push_back(nodes.node_id_from_primitive(v[lv]));
				}
				else
				{
					if (discr_order.size() > 0 && other_face >= 0 && discr_order(f) > discr_order(other_face))
						res.push_back(-lv - 1);
					else
						res.push_back(nodes.node_id_from_primitive(v[lv]));
				}
			}
		}

		for (int le = 0; le < e.rows(); ++le)
		{
			const auto index = e[le];
			const auto other_face = mesh.switch_face(index).face;

			if (is_geom_bases)
			{
				auto node_ids = nodes.node_ids_from_edge(index, p - 1);
				res.insert(res.end(), node_ids.begin(), node_ids.end());
			}
			else
			{
				if (!mesh.is_conforming())
				{
					const auto &ncmesh = dynamic_cast<const NCMesh2D &>(mesh);
					const int edge_id = ncmesh.face_edge(f, le);
					// follower edge
					if (ncmesh.leader_edge_of_edge(edge_id) >= 0)
					{
						for (int tmp = 0; tmp < p - 1; ++tmp)
							res.push_back(-le - 1);
					}
					// leader or conforming edge with constrained order
					else if (edge_orders[edge_id] < discr_order[f])
					{
						for (int tmp = 0; tmp < p - 1; ++tmp)
							res.push_back(-le - 1);
						// leader edge
						if (ncmesh.n_follower_edges(edge_id) > 0)
							edge_virtual_nodes[edge_id] = nodes.node_ids_from_edge(index, edge_orders[edge_id] - 1);
					}
					// leader or conforming edge with unconstrained order
					else
					{
						auto node_ids = nodes.node_ids_from_edge(index, p - 1);
						res.insert(res.end(), node_ids.begin(), node_ids.end());
					}
				}
				else
				{
					bool skip_other = discr_order.size() > 0 && other_face >= 0 && discr_order(f) > discr_order(other_face);

					if (discr_order.size() > 0 && other_face >= 0 && discr_order(f) > discr_order(other_face))
					{
						for (int tmp = 0; tmp < p - 1; ++tmp)
							res.push_back(-le - 1);
					}
					else
					{
						auto node_ids = nodes.node_ids_from_edge(index, p - 1);
						res.insert(res.end(), node_ids.begin(), node_ids.end());
					}
				}
			}
		}

		if (n_face_nodes > 0)
		{
			const auto index = e[0];

			auto node_ids = nodes.node_ids_from_face(index, p - 2);
			res.insert(res.end(), node_ids.begin(), node_ids.end());
		}

		assert(res.size() == size_t(3 + n_edge_nodes + n_face_nodes));
		// return res;
	}

	void quad_local_to_global(const bool serendipity, const int q, const Mesh2D &mesh, int f, const Eigen::VectorXi &discr_order, std::vector<int> &res, MeshNodes &nodes)
	{
		int edge_offset = mesh.n_vertices();
		int face_offset = edge_offset + mesh.n_edges();

		const int nn = q > 1 ? (q - 1) : 0;
		const int n_edge_nodes = nn * 4;
		const int n_face_nodes = serendipity ? 0 : nn * nn;

		if (q == 0)
		{
			res.push_back(nodes.node_id_from_face(f));
			return;
		}

		// std::vector<int> res;
		res.reserve(4 + n_edge_nodes + n_face_nodes);

		// Vertex nodes
		auto v = quad_vertices_local_to_global(mesh, f);

		// Edge nodes
		Eigen::Matrix<Navigation::Index, 4, 1> e;
		Eigen::Matrix<int, 4, 2> ev;
		ev.row(0) << v[0], v[1];
		ev.row(1) << v[1], v[2];
		ev.row(2) << v[2], v[3];
		ev.row(3) << v[3], v[0];
		for (int le = 0; le < e.rows(); ++le)
		{
			const auto index = find_edge(mesh, f, ev(le, 0), ev(le, 1));
			e[le] = index;
		}

		for (size_t lv = 0; lv < v.size(); ++lv)
		{
			const auto index = e[lv];
			const auto other_face = mesh.switch_face(index).face;

			if (discr_order.size() > 0 && other_face >= 0 && discr_order(f) > discr_order(other_face))
				res.push_back(-lv - 1);
			else
				res.push_back(nodes.node_id_from_primitive(v[lv]));
		}

		for (int le = 0; le < e.rows(); ++le)
		{
			const auto index = e[le];
			const auto other_face = mesh.switch_face(index).face;

			bool skip_other = discr_order.size() > 0 && other_face >= 0 && discr_order(f) > discr_order(other_face);

			if (discr_order.size() > 0 && other_face >= 0 && discr_order(f) > discr_order(other_face))
			{
				for (int tmp = 0; tmp < q - 1; ++tmp)
					res.push_back(-le - 1);
			}
			else
			{
				auto node_ids = nodes.node_ids_from_edge(index, q - 1);
				res.insert(res.end(), node_ids.begin(), node_ids.end());
			}
		}

		if (n_face_nodes > 0)
		{
			const auto index = e[0];

			auto node_ids = nodes.node_ids_from_face(index, q - 1);
			res.insert(res.end(), node_ids.begin(), node_ids.end());
		}

		assert(res.size() == size_t(4 + n_edge_nodes + n_face_nodes));
		// return res;
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
		const Mesh2D &mesh,
		const Eigen::VectorXi &discr_orders,
		const Eigen::VectorXi &edge_orders,
		const bool serendipity,
		const bool has_polys,
		const bool is_geom_bases,
		MeshNodes &nodes,
		std::vector<std::vector<int>> &edge_virtual_nodes,
		std::vector<std::vector<int>> &element_nodes_id,
		std::vector<LocalBoundary> &local_boundary,
		std::map<int, InterfaceData> &poly_edge_to_data)
	{
		// Step 1: Assign global node ids for each quads
		local_boundary.clear();

		element_nodes_id.resize(mesh.n_faces());

		if (!mesh.is_conforming())
		{
			const auto &ncmesh = dynamic_cast<const NCMesh2D &>(mesh);
			edge_virtual_nodes.resize(ncmesh.n_edges());
		}

		for (int f = 0; f < mesh.n_faces(); ++f)
		{
			const int discr_order = discr_orders(f);
			if (mesh.is_cube(f))
			{
				quad_local_to_global(serendipity, discr_order, mesh, f, discr_orders, element_nodes_id[f], nodes);

				LocalBoundary lb(f, BoundaryType::QUAD_LINE);

				auto v = quad_vertices_local_to_global(mesh, f);
				Eigen::Matrix<int, 4, 2> ev;
				ev.row(0) << v[0], v[1];
				ev.row(1) << v[1], v[2];
				ev.row(2) << v[2], v[3];
				ev.row(3) << v[3], v[0];

				for (int i = 0; i < int(ev.rows()); ++i)
				{
					const auto index = find_edge(mesh, f, ev(i, 0), ev(i, 1));
					const int edge = index.edge;

					if (mesh.is_boundary_edge(edge) || mesh.get_boundary_id(edge) > 0)
					{
						lb.add_boundary_primitive(edge, i);
					}
				}

				if (!lb.empty())
					local_boundary.emplace_back(lb);
			}
			else if (mesh.is_simplex(f))
			{
				tri_local_to_global(is_geom_bases, discr_order, mesh, f, discr_orders, edge_orders, element_nodes_id[f], nodes, edge_virtual_nodes);

				auto v = tri_vertices_local_to_global(mesh, f);

				Eigen::Matrix<int, 3, 2> ev;
				ev.row(0) << v[0], v[1];
				ev.row(1) << v[1], v[2];
				ev.row(2) << v[2], v[0];

				LocalBoundary lb(f, BoundaryType::TRI_LINE);

				for (int i = 0; i < int(ev.rows()); ++i)
				{
					const auto index = find_edge(mesh, f, ev(i, 0), ev(i, 1));
					const int edge = index.edge;

					if (mesh.is_boundary_edge(edge) || mesh.get_boundary_id(edge) > 0)
					{
						lb.add_boundary_primitive(edge, i);
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
		for (int f = 0; f < mesh.n_faces(); ++f)
		{
			// Skip non-polytopes
			if (!mesh.is_polytope(f))
			{
				continue;
			}

			auto index = mesh.get_index_from_face(f, 0);
			for (int lv = 0; lv < mesh.n_face_vertices(f); ++lv)
			{
				auto index2 = mesh.switch_face(index);
				if (index2.face >= 0)
				{
					// Opposite face is a quad, we need to set interface data
					int f2 = index2.face;
					const int discr_order = discr_orders(f2);

					if (mesh.is_cube(f2))
					{
						indices = LagrangeBasis2d::quad_edge_local_nodes(discr_order, mesh, index2);
					}
					else if (mesh.is_simplex(f2))
					{
						indices = LagrangeBasis2d::tri_edge_local_nodes(discr_order, mesh, index2);
					}
					else
					{
						// assert(false);
						continue;
					}
					InterfaceData data;
					assert(indices.size() == 2 + discr_order - 1);
					data.local_indices.insert(data.local_indices.begin(), indices.data(), indices.data() + indices.size());
					assert(indices.size() == data.local_indices.size());
					poly_edge_to_data[index2.edge] = data;
				}
				index = mesh.next_around_face(index);
			}
		}
	}

	/// @brief      map barycentric coordinates of a triangle to global coordinates
	///
	/// @param[in]  verts           The vertices of a triangle, 3 x 2
	/// @param[in]  uv        		The barycentric coordinates, N x 2
	/// @param[out] pts             Output global coordinates, N x 2
	///
	void local_to_global(const Eigen::MatrixXd &verts, const Eigen::MatrixXd &uv, Eigen::MatrixXd &pts)
	{
		const int dim = verts.cols();
		const int N = uv.rows();
		assert(dim == 2);
		assert(uv.cols() == dim);
		assert(verts.rows() == dim + 1);

		pts.setZero(N, dim);
		for (int i = 0; i < N; i++)
			pts.row(i) = uv(i, 0) * verts.row(1) + uv(i, 1) * verts.row(2) + (1.0 - uv(i, 0) - uv(i, 1)) * verts.row(0);
	}

	/// @brief      map global coordinates to barycentric coordinates of a triangle
	///
	/// @param[in]  verts           The vertices of a triangle, 3 x 2
	/// @param[out] uv        		The barycentric coordinates, N x 2
	/// @param[in]  pts             Output global coordinates, N x 2
	///
	void global_to_local(const Eigen::MatrixXd &verts, const Eigen::MatrixXd &pts, Eigen::MatrixXd &uv)
	{
		const int dim = verts.cols();
		const int N = pts.rows();
		assert(dim == 2);
		assert(verts.rows() == dim + 1);
		assert(pts.cols() == dim);

		Eigen::Matrix2d J;
		for (int i = 0; i < dim; i++)
			J.col(i) = verts.row(i + 1) - verts.row(0);

		double detJ = J(0, 0) * J(1, 1) - J(0, 1) * J(1, 0);
		J /= detJ;

		uv.setZero(N, dim);
		maybe_parallel_for(N, [&](int start, int end, int thread_id) {
			for (int i = start; i < end; i++)
			{
				auto point = pts.row(i) - verts.row(0);
				uv(i, 0) = J(1, 1) * point(0) - J(0, 1) * point(1);
				uv(i, 1) = J(0, 0) * point(1) - J(1, 0) * point(0);
			}
		});
	}

	/// @brief      compute edge orders given element orders, assure basis continuity
	///
	/// @param[in]  mesh            Input ncmesh
	/// @param[in]  elem_orders		Element orders
	/// @param[out] edge_orders     Edge orders
	///
	void compute_edge_orders(const NCMesh2D &mesh, const Eigen::VectorXi &elem_orders, Eigen::VectorXi &edge_orders)
	{
		const int max_order = elem_orders.maxCoeff();
		edge_orders.setConstant(mesh.n_edges(), max_order);

		for (int i = 0; i < mesh.n_faces(); i++)
			for (int j = 0; j < mesh.n_face_vertices(i); j++)
				edge_orders[mesh.face_edge(i, j)] = std::min(edge_orders[mesh.face_edge(i, j)], elem_orders[i]);

		for (int i = 0; i < mesh.n_edges(); i++)
			if (mesh.leader_edge_of_edge(i) >= 0)
				edge_orders[mesh.leader_edge_of_edge(i)] = std::min(edge_orders[mesh.leader_edge_of_edge(i)], edge_orders[i]);

		for (int i = 0; i < mesh.n_edges(); i++)
			if (mesh.leader_edge_of_edge(i) >= 0)
				edge_orders[i] = std::min(edge_orders[mesh.leader_edge_of_edge(i)], edge_orders[i]);
	}
} // anonymous namespace

Eigen::VectorXi LagrangeBasis2d::tri_edge_local_nodes(const int p, const Mesh2D &mesh, Navigation::Index index)
{
	int f = index.face;
	assert(mesh.is_simplex(f));

	// Local to global mapping of node indices
	auto l2g = tri_vertices_local_to_global(mesh, f);

	// Extract requested interface
	Eigen::VectorXi result(2 + (p - 1));
	result[0] = find_index(l2g.begin(), l2g.end(), index.vertex);
	result[result.size() - 1] = find_index(l2g.begin(), l2g.end(), mesh.switch_vertex(index).vertex);

	if ((result[0] == 0 && result[result.size() - 1] == 1) || (result[0] == 1 && result[result.size() - 1] == 2) || (result[0] == 2 && result[result.size() - 1] == 0))
	{
		for (int i = 0; i < p - 1; ++i)
		{
			result[i + 1] = 3 + result[0] * (p - 1) + i;
		}
	}
	else
	{
		for (int i = 0; i < p - 1; ++i)
		{
			result[i + 1] = 3 + (result[0] + (result[0] == 0 ? 3 : 0)) * (p - 1) - i - 1;
		}
	}

	return result;
}

Eigen::VectorXi LagrangeBasis2d::quad_edge_local_nodes(const int q, const Mesh2D &mesh, Navigation::Index index)
{
	int f = index.face;
	assert(mesh.is_cube(f));

	// Local to global mapping of node indices
	auto l2g = quad_vertices_local_to_global(mesh, f);

	// Extract requested interface
	Eigen::VectorXi result(2 + (q - 1));
	result[0] = find_index(l2g.begin(), l2g.end(), index.vertex);
	result[result.size() - 1] = find_index(l2g.begin(), l2g.end(), mesh.switch_vertex(index).vertex);

	if ((result[0] == 0 && result[result.size() - 1] == 1) || (result[0] == 1 && result[result.size() - 1] == 2) || (result[0] == 2 && result[result.size() - 1] == 3) || (result[0] == 3 && result[result.size() - 1] == 0))
	{
		for (int i = 0; i < q - 1; ++i)
		{
			result[i + 1] = 4 + result[0] * (q - 1) + i;
		}
	}
	else
	{
		for (int i = 0; i < q - 1; ++i)
		{
			result[i + 1] = 4 + (result[0] + (result[0] == 0 ? 4 : 0)) * (q - 1) - i - 1;
		}
	}

	return result;
}

// -----------------------------------------------------------------------------

int LagrangeBasis2d::build_bases(
	const Mesh2D &mesh,
	const std::string &assembler,
	const int quadrature_order,
	const int mass_quadrature_order,
	const int discr_order,
	const bool serendipity,
	const bool has_polys,
	const bool is_geom_bases,
	std::vector<ElementBases> &bases,
	std::vector<LocalBoundary> &local_boundary,
	std::map<int, InterfaceData> &poly_edge_to_data,
	std::shared_ptr<MeshNodes> &mesh_nodes)
{

	Eigen::VectorXi discr_orders(mesh.n_faces());
	discr_orders.setConstant(discr_order);

	return build_bases(mesh, assembler, quadrature_order, mass_quadrature_order, discr_orders, serendipity, has_polys, is_geom_bases, bases, local_boundary, poly_edge_to_data, mesh_nodes);
}

int LagrangeBasis2d::build_bases(
	const Mesh2D &mesh,
	const std::string &assembler,
	const int quadrature_order,
	const int mass_quadrature_order,
	const Eigen::VectorXi &discr_orders,
	const bool serendipity,
	const bool has_polys,
	const bool is_geom_bases,
	std::vector<ElementBases> &bases,
	std::vector<LocalBoundary> &local_boundary,
	std::map<int, InterfaceData> &poly_edge_to_data,
	std::shared_ptr<MeshNodes> &mesh_nodes)
{
	assert(!mesh.is_volume());
	assert(discr_orders.size() == mesh.n_faces());

	const int max_p = discr_orders.maxCoeff();
	const int nn = max_p > 1 ? (max_p - 1) : 0;
	const int n_face_nodes = std::max(nn * nn, max_p == 1 ? 1 : 0);

	Eigen::VectorXi edge_orders;
	if (!mesh.is_conforming())
	{
		const auto &ncmesh = dynamic_cast<const NCMesh2D &>(mesh);
		compute_edge_orders(ncmesh, discr_orders, edge_orders);
	}

	mesh_nodes = std::make_shared<MeshNodes>(mesh, has_polys, !is_geom_bases, (max_p > 1 ? (max_p - 1) : 0) * (is_geom_bases ? 2 : 1), max_p == 0 ? 1 : n_face_nodes);
	MeshNodes &nodes = *mesh_nodes;
	std::vector<std::vector<int>> element_nodes_id, edge_virtual_nodes;
	compute_nodes(mesh, discr_orders, edge_orders, serendipity, has_polys, is_geom_bases, nodes, edge_virtual_nodes, element_nodes_id, local_boundary, poly_edge_to_data);
	// boundary_nodes = nodes.boundary_nodes();

	bases.resize(mesh.n_faces());
	std::vector<int> interface_elements;
	interface_elements.reserve(mesh.n_faces());

	for (int e = 0; e < mesh.n_faces(); ++e)
	{
		ElementBases &b = bases[e];
		const int discr_order = discr_orders(e);
		const int n_el_bases = element_nodes_id[e].size();
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
			const int real_order = quadrature_order > 0 ? quadrature_order : AssemblerUtils::quadrature_order(assembler, discr_order, AssemblerUtils::BasisType::CUBE_LAGRANGE, 2);
			const int real_mass_order = mass_quadrature_order > 0 ? mass_quadrature_order : AssemblerUtils::quadrature_order("Mass", discr_order, AssemblerUtils::BasisType::CUBE_LAGRANGE, 2);
			b.set_quadrature([real_order](Quadrature &quad) {
				QuadQuadrature quad_quadrature;
				quad_quadrature.get_quadrature(real_order, quad);
			});
			b.set_mass_quadrature([real_mass_order](Quadrature &quad) {
				QuadQuadrature quad_quadrature;
				quad_quadrature.get_quadrature(real_mass_order, quad);
			});
			// quad_quadrature.get_quadrature(real_order, b.quadrature);

			b.set_local_node_from_primitive_func([discr_order, e](const int primitive_id, const Mesh &mesh) {
				const auto &mesh2d = dynamic_cast<const Mesh2D &>(mesh);
				auto index = mesh2d.get_index_from_face(e);

				for (int le = 0; le < mesh2d.n_face_vertices(e); ++le)
				{
					if (index.edge == primitive_id)
						break;
					index = mesh2d.next_around_face(index);
				}
				assert(index.edge == primitive_id);
				return quad_edge_local_nodes(discr_order, mesh2d, index);
			});

			for (int j = 0; j < n_el_bases; ++j)
			{
				const int global_index = element_nodes_id[e][j];

				// if(!skip_interface_element)
				b.bases[j].init(discr_order, global_index, j, nodes.node_position(global_index));

				const int dtmp = serendipity ? -2 : discr_order;

				b.bases[j].set_basis([dtmp, j](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { autogen::q_basis_value_2d(dtmp, j, uv, val); });
				b.bases[j].set_grad([dtmp, j](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { autogen::q_grad_basis_value_2d(dtmp, j, uv, val); });
			}
		}
		else if (mesh.is_simplex(e))
		{
			const int real_order = quadrature_order > 0 ? quadrature_order : AssemblerUtils::quadrature_order(assembler, discr_order, AssemblerUtils::BasisType::SIMPLEX_LAGRANGE, 2);
			const int real_mass_order = mass_quadrature_order > 0 ? mass_quadrature_order : AssemblerUtils::quadrature_order("Mass", discr_order, AssemblerUtils::BasisType::SIMPLEX_LAGRANGE, 2);
			b.set_quadrature([real_order](Quadrature &quad) {
				TriQuadrature tri_quadrature;
				tri_quadrature.get_quadrature(real_order, quad);
			});
			b.set_mass_quadrature([real_mass_order](Quadrature &quad) {
				TriQuadrature tri_quadrature;
				tri_quadrature.get_quadrature(real_mass_order, quad);
			});

			b.set_local_node_from_primitive_func([discr_order, e](const int primitive_id, const Mesh &mesh) {
				const auto &mesh2d = dynamic_cast<const Mesh2D &>(mesh);
				auto index = mesh2d.get_index_from_face(e);

				for (int le = 0; le < mesh2d.n_face_vertices(e); ++le)
				{
					if (index.edge == primitive_id)
						break;
					index = mesh2d.next_around_face(index);
				}
				assert(index.edge == primitive_id);
				return tri_edge_local_nodes(discr_order, mesh2d, index);
			});

			const bool rational = is_geom_bases && mesh.is_rational() && !mesh.cell_weights(e).empty();

			for (int j = 0; j < n_el_bases; ++j)
			{
				const int global_index = element_nodes_id[e][j];

				if (!skip_interface_element)
				{
					b.bases[j].init(discr_order, global_index, j, nodes.node_position(global_index));
				}

				if (rational)
				{
					const auto &w = mesh.cell_weights(e);
					assert(discr_order == 2);
					assert(w.size() == 6);

					b.bases[j].set_basis([discr_order, j, w](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) {
						autogen::p_basis_value_2d(discr_order, j, uv, val);
						Eigen::MatrixXd denom = val;
						denom.setZero();
						Eigen::MatrixXd tmp;

						for (int k = 0; k < 6; ++k)
						{
							autogen::p_basis_value_2d(discr_order, k, uv, tmp);
							denom += w[k] * tmp;
						}

						val = (w[j] * val.array() / denom.array()).eval();
					});

					b.bases[j].set_grad([discr_order, j, w](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) {
						Eigen::MatrixXd b;
						autogen::p_basis_value_2d(discr_order, j, uv, b);
						autogen::p_grad_basis_value_2d(discr_order, j, uv, val);
						Eigen::MatrixXd denom = b;
						denom.setZero();
						Eigen::MatrixXd denom_prime = val;
						denom_prime.setZero();
						Eigen::MatrixXd tmp;

						for (int k = 0; k < 6; ++k)
						{
							autogen::p_basis_value_2d(discr_order, k, uv, tmp);
							denom += w[k] * tmp;

							autogen::p_grad_basis_value_2d(discr_order, k, uv, tmp);
							denom_prime += w[k] * tmp;
						}

						val.col(0) = ((w[j] * val.col(0).array() * denom.array() - w[j] * b.array() * denom_prime.col(0).array()) / (denom.array() * denom.array())).eval();
						val.col(1) = ((w[j] * val.col(1).array() * denom.array() - w[j] * b.array() * denom_prime.col(1).array()) / (denom.array() * denom.array())).eval();
					});
				}
				else
				{
					b.bases[j].set_basis([discr_order, j](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { autogen::p_basis_value_2d(discr_order, j, uv, val); });
					b.bases[j].set_grad([discr_order, j](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { autogen::p_grad_basis_value_2d(discr_order, j, uv, val); });
				}
			}
		}
		else
		{
			// Polygon bases are built later on
		}

#ifndef NDEBUG
		if (mesh.is_conforming())
		{
			Eigen::MatrixXd uv(4, 2);
			uv << 0.1, 0.1, 0.3, 0.3, 0.9, 0.01, 0.01, 0.9;
			Eigen::MatrixXd dx(4, 1);
			dx.setConstant(1e-6);
			Eigen::MatrixXd uvdx = uv;
			uvdx.col(0) += dx;
			Eigen::MatrixXd uvdy = uv;
			uvdy.col(1) += dx;
			Eigen::MatrixXd grad, val, vdx, vdy;

			for (int j = 0; j < n_el_bases; ++j)
			{
				b.bases[j].eval_grad(uv, grad);

				b.bases[j].eval_basis(uv, val);
				b.bases[j].eval_basis(uvdx, vdx);
				b.bases[j].eval_basis(uvdy, vdy);

				assert((grad.col(0) - (vdx - val) / 1e-6).norm() < 1e-4);
				assert((grad.col(1) - (vdy - val) / 1e-6).norm() < 1e-4);
			}
		}
#endif
	}

	if (!is_geom_bases)
	{
		if (!mesh.is_conforming())
		{
			const auto &ncmesh = dynamic_cast<const NCMesh2D &>(mesh);

			std::vector<std::vector<int>> elementOrder;
			{
				const int max_order = discr_orders.maxCoeff(), min_order = discr_orders.minCoeff();
				int max_level = 0;
				for (int e = 0; e < ncmesh.n_faces(); e++)
					if (max_level < ncmesh.face_ref_level(e))
						max_level = ncmesh.face_ref_level(e);

				elementOrder.resize((max_level + 1) * (max_order - min_order + 1));
				int N = 0;
				int cur_level = 0;
				while (cur_level <= max_level)
				{
					int order = min_order;
					while (order <= max_order)
					{
						int cur_bucket = (max_order - min_order + 1) * cur_level + (order - min_order);
						for (int i = 0; i < ncmesh.n_faces(); i++)
						{
							if (ncmesh.face_ref_level(i) != cur_level || discr_orders[i] != order)
								continue;

							N++;
							elementOrder[cur_bucket].push_back(i);
						}
						order++;
					}
					cur_level++;
				}
			}

			for (const auto &bucket : elementOrder)
			{
				if (bucket.size() == 0)
					continue;
				for (const int e : bucket)
				{
					ElementBases &b = bases[e];
					const int discr_order = discr_orders(e);
					const int n_el_bases = element_nodes_id[e].size();

					for (int j = 0; j < n_el_bases; ++j)
					{
						const int global_index = element_nodes_id[e][j];

						if (global_index >= 0)
							b.bases[j].init(discr_order, global_index, j, nodes.node_position(global_index));
						else
						{
							int large_edge = -1, local_edge = -1;
							int opposite_element = -1;
							bool is_on_leader_edge = false;

							if (j < 3 + 3 * (discr_order - 1) && j >= 3)
							{
								local_edge = (j - 3) / (discr_order - 1);
								large_edge = ncmesh.face_edge(e, local_edge);
								if (ncmesh.n_follower_edges(large_edge) > 0)
									is_on_leader_edge = true;
							}

							// this node is on a leader edge, but its order is constrained
							if (is_on_leader_edge)
							{
								const int edge_order = edge_orders[large_edge];

								// indices of fake nodes on this edge in the opposite element
								Eigen::VectorXi indices;
								indices.resize(discr_order + 1);
								for (int i = 0; i < discr_order - 1; i++)
								{
									indices[i + 1] = 3 + local_edge * (discr_order - 1) + i;
								}
								int index = indices[(j - 3) % (discr_order - 1) + 1];

								// the position of node j in the opposite element
								Eigen::MatrixXd lnodes;
								autogen::p_nodes_2d(discr_order, lnodes);
								Eigen::Vector2d node_position = lnodes.row(index);

								std::function<double(const int, const int, const double)> basis_1d = [](const int order, const int id, const double x) -> double {
									assert(id <= order && id >= 0);
									double y = 1;
									for (int o = 0; o <= order; o++)
									{
										if (o != id)
											y *= (x * order - o) / (id - o);
									}
									return y;
								};

								// apply basis projection
								double x = NCMesh2D::element_weight_to_edge_weight(local_edge, node_position);
								for (int i = 0; i < edge_virtual_nodes[large_edge].size(); i++)
								{
									const int global_index = edge_virtual_nodes[large_edge][i];
									const double weight = basis_1d(edge_order, i + 1, x);
									if (std::abs(weight) < 1e-12)
										continue;
									b.bases[j].global().emplace_back(global_index, nodes.node_position(global_index), weight);
								}

								const auto &global_1 = b.bases[local_edge].global();
								const auto &global_2 = b.bases[(local_edge + 1) % 3].global();
								double weight = basis_1d(edge_order, 0, x);
								if (std::abs(weight) > 1e-12)
									for (size_t ii = 0; ii < global_1.size(); ++ii)
										b.bases[j].global().emplace_back(global_1[ii].index, global_1[ii].node, weight * global_1[ii].val);
								weight = basis_1d(edge_order, edge_order, x);
								if (std::abs(weight) > 1e-12)
									for (size_t ii = 0; ii < global_2.size(); ++ii)
										b.bases[j].global().emplace_back(global_2[ii].index, global_2[ii].node, weight * global_2[ii].val);
							}
							else
							{
								Eigen::MatrixXd global_position;
								// this node is a hanging vertex, and it's not on a follower edge
								if (j < 3)
								{
									const int v_id = ncmesh.face_vertex(e, j);
									large_edge = ncmesh.leader_edge_of_vertex(v_id);
									opposite_element = ncmesh.face_neighbor(large_edge);
									global_position = ncmesh.point(v_id);
								}
								// this node is on a follower edge
								else
								{
									// find the local id of small edge
									int small_edge = ncmesh.face_edge(e, local_edge);
									large_edge = ncmesh.leader_edge_of_edge(small_edge);

									// this edge is non-conforming
									if (ncmesh.n_face_neighbors(small_edge) == 1)
										opposite_element = ncmesh.face_neighbor(large_edge);
									// this edge is conforming, but the order on two sides is different
									else
									{
										Navigation::Index idx;
										idx.edge = small_edge;
										idx.vertex = ncmesh.edge_vertex(small_edge, 0);
										idx.face_corner = -1;
										idx.face = e;
										opposite_element = ncmesh.switch_face(idx).face;
									}

									// the position of node j in the opposite element
									Eigen::MatrixXd lnodes;
									autogen::p_nodes_2d(discr_order, lnodes);
									global_position = lnodes.row(j);

									Eigen::MatrixXd verts(ncmesh.n_face_vertices(e), 2);
									for (int i = 0; i < verts.rows(); i++)
										verts.row(i) = ncmesh.point(ncmesh.face_vertex(e, i));

									Eigen::MatrixXd local_position = lnodes.row(j);
									local_to_global(verts, local_position, global_position);
								}

								Eigen::MatrixXd verts(ncmesh.n_face_vertices(e), 2);
								for (int i = 0; i < verts.rows(); i++)
									verts.row(i) = ncmesh.point(ncmesh.face_vertex(opposite_element, i));

								Eigen::MatrixXd node_position;
								global_to_local(verts, global_position, node_position);

								// evaluate the basis of the opposite element at this node
								const auto &other_bases = bases[opposite_element];
								std::vector<AssemblyValues> w;
								other_bases.evaluate_bases(node_position, w);

								// apply basis projection
								for (long i = 0; i < w.size(); ++i)
								{
									assert(w[i].val.size() == 1);
									if (std::abs(w[i].val(0)) < 1e-12)
										continue;

									for (size_t ii = 0; ii < other_bases.bases[i].global().size(); ++ii)
									{
										const auto &other_global = other_bases.bases[i].global()[ii];
										b.bases[j].global().emplace_back(other_global.index, other_global.node, w[i].val(0) * other_global.val);
									}
								}
							}

							auto &global_ = b.bases[j].global();
							if (global_.size() <= 1)
								continue;

							std::map<int, Local2Global> list;
							for (size_t ii = 0; ii < global_.size(); ii++)
							{
								auto pair = list.insert({global_[ii].index, global_[ii]});
								if (!pair.second && pair.first != list.end())
								{
									assert((pair.first->second.node - global_[ii].node).norm() < 1e-12);
									pair.first->second.val += global_[ii].val;
								}
							}

							global_.clear();
							for (auto it = list.begin(); it != list.end(); ++it)
							{
								if (std::abs(it->second.val) > 1e-12)
									global_.push_back(it->second);
							}
						}
					}
				}
			}
		}
		else
		{
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
						// TODO
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
								const auto le = -(global_index + 1);

								auto v = tri_vertices_local_to_global(mesh, e);
								Eigen::Matrix<int, 3, 2> ev;
								ev.row(0) << v[0], v[1];
								ev.row(1) << v[1], v[2];
								ev.row(2) << v[2], v[0];

								const auto index = mesh.switch_face(find_edge(mesh, e, ev(le, 0), ev(le, 1)));
								const auto other_face = index.face;

								Eigen::RowVector2d node_position;
								assert(discr_order > 1);

								auto indices = tri_edge_local_nodes(discr_order, mesh, index);
								Eigen::MatrixXd lnodes;
								autogen::p_nodes_2d(discr_order, lnodes);

								if (j < 3)
									node_position = lnodes.row(indices(0));
								else if (j < 3 + 3 * (discr_order - 1))
									node_position = lnodes.row(indices(((j - 3) % (discr_order - 1)) + 1));
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
								// Eigen::MatrixXd w;
								std::vector<AssemblyValues> w;
								other_bases.evaluate_bases(node_position, w);

								for (long i = 0; i < w.size(); ++i)
								{
									assert(w[i].val.size() == 1);
									if (std::abs(w[i].val(0)) < 1e-8)
										continue;

									// assert(other_bases.bases[i].global().size() == 1);
									// const auto &other_global = other_bases.bases[i].global().front();
									// std::cout<<"e "<<e<<" " <<j << " gid "<<other_global.index<<std::endl;
									// b.bases[j].global().emplace_back(other_global.index, other_global.node, w(i));
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
