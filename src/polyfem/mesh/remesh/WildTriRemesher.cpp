#include "WildRemesher.hpp"

#include <polyfem/mesh/remesh/wild_remesh/OperationCache.hpp>
#include <polyfem/utils/GeometryUtils.hpp>

#include <wmtk/utils/TriQualityUtils.hpp>

#include <igl/predicates/predicates.h>

namespace polyfem::mesh
{
	using Tuple = WildTriRemesher::Tuple;

	template <>
	void WildTriRemesher::init_attributes_and_connectivity(
		const size_t num_vertices, const Eigen::MatrixXi &triangles)
	{
		// Register attributes
		p_vertex_attrs = &vertex_attrs;
		p_edge_attrs = &boundary_attrs;
		p_face_attrs = &element_attrs;

		// Convert from eigen to internal representation
		std::vector<std::array<size_t, 3>> tri(triangles.rows());
		for (int i = 0; i < triangles.rows(); i++)
			for (int j = 0; j < triangles.cols(); j++)
				tri[i][j] = (size_t)triangles(i, j);

		// Initialize the trimesh class which handles connectivity
		wmtk::TriMesh::create_mesh(num_vertices, tri);
	}

	template <>
	WildTriRemesher::BoundaryMap<int> WildTriRemesher::boundary_ids() const
	{
		const std::vector<Tuple> edges = get_edges();
		EdgeMap<int> boundary_ids;
		for (const Tuple &edge : edges)
		{
			const size_t e0 = edge.vid(*this);
			const size_t e1 = edge.switch_vertex(*this).vid(*this);
			boundary_ids[{{e0, e1}}] = boundary_attrs[edge.eid(*this)].boundary_id;
		}
		return boundary_ids;
	}

	template <>
	bool WildTriRemesher::is_rest_inverted(const Tuple &loc) const
	{
		// Get the vertices ids
		const std::array<size_t, 3> vids = oriented_tri_vids(loc);

		igl::predicates::exactinit();

		// Use igl for checking orientation
		const igl::predicates::Orientation orientation =
			igl::predicates::orient2d(
				vertex_attrs[vids[0]].rest_position,
				vertex_attrs[vids[1]].rest_position,
				vertex_attrs[vids[2]].rest_position);

		// The element is inverted if it not positive (i.e. it is negative or it is degenerate)
		return orientation != igl::predicates::Orientation::POSITIVE;
	}

	template <>
	bool WildTriRemesher::is_inverted(const Tuple &loc) const
	{
		// Get the vertices ids
		const std::array<size_t, 3> vids = oriented_tri_vids(loc);

		igl::predicates::exactinit();

		for (int i = 0; i < n_quantities() / 3 + 2; ++i)
		{
			// Use igl for checking orientation
			const igl::predicates::Orientation orientation =
				igl::predicates::orient2d(
					vertex_attrs[vids[0]].position_i(i),
					vertex_attrs[vids[1]].position_i(i),
					vertex_attrs[vids[2]].position_i(i));

			// The element is inverted if it not positive (i.e. it is negative or it is degenerate)
			if (orientation != igl::predicates::Orientation::POSITIVE)
				return true;
		}

		return false;
	}

	template <>
	double WildTriRemesher::element_volume(const Tuple &e) const
	{
		const std::array<size_t, 3> vids = oriented_tri_vids(e);
		return utils::triangle_area_2D(
			vertex_attrs[vids[0]].rest_position,
			vertex_attrs[vids[1]].rest_position,
			vertex_attrs[vids[2]].rest_position);
	}

	template <>
	size_t WildTriRemesher::facet_id(const Tuple &t) const
	{
		return t.eid(*this);
	}

	template <>
	size_t WildTriRemesher::element_id(const Tuple &t) const
	{
		return t.fid(*this);
	}

	template <>
	Tuple WildTriRemesher::tuple_from_element(size_t elem_id) const
	{
		return tuple_from_tri(elem_id);
	}

	template <>
	Tuple WildTriRemesher::tuple_from_facet(size_t elem_id, int local_facet_id) const
	{
		return tuple_from_edge(elem_id, local_facet_id);
	}

	template <>
	bool WildTriRemesher::is_boundary_edge(const Tuple &e) const
	{
		return TriMesh::is_boundary_edge(e);
	}

	template <>
	bool WildTriRemesher::is_body_boundary_edge(const Tuple &e) const
	{
		const auto adj_face = e.switch_face(*this);
		return !adj_face.has_value()
			   || element_attrs[element_id(e)].body_id
					  != element_attrs[element_id(*adj_face)].body_id;
	}

	template <>
	bool WildTriRemesher::is_boundary_vertex(const Tuple &v) const
	{
		return TriMesh::is_boundary_vertex(v);
	}

	template <>
	bool WildTriRemesher::is_body_boundary_vertex(const Tuple &v) const
	{
		for (const auto &e : get_one_ring_edges_for_vertex(v))
			if (is_body_boundary_edge(e))
				return true;
		return false;
	}

	template <>
	bool WildTriRemesher::is_boundary_facet(const Tuple &t) const
	{
		return is_boundary_edge(t);
	}

	template <>
	bool WildTriRemesher::is_boundary_op() const
	{
		return op_cache->is_boundary_op();
	}

	template <>
	Eigen::MatrixXi WildTriRemesher::boundary_edges() const
	{
		const std::vector<Tuple> boundary_edge_tuples = boundary_facets();
		Eigen::MatrixXi BE(boundary_edge_tuples.size(), 2);
		for (int i = 0; i < BE.rows(); ++i)
		{
			BE(i, 0) = boundary_edge_tuples[i].vid(*this);
			BE(i, 1) = boundary_edge_tuples[i].switch_vertex(*this).vid(*this);
		}
		if (obstacle().n_edges() > 0)
			utils::append_rows(BE, obstacle().e().array() + vert_capacity());
		return BE;
	}

	template <>
	Eigen::MatrixXi WildTriRemesher::boundary_faces() const
	{
		return Eigen::MatrixXi();
	}

	template <>
	std::vector<Tuple> WildTriRemesher::get_facets() const
	{
		return get_edges();
	}

	template <>
	std::vector<Tuple> WildTriRemesher::get_elements() const
	{
		return get_faces();
	}

	template <>
	std::array<Tuple, 2> WildTriRemesher::facet_vertices(const Tuple &t) const
	{
		return {{t, t.switch_vertex(*this)}};
	}

	template <>
	std::array<size_t, 2> WildTriRemesher::facet_vids(const Tuple &t) const
	{
		return {{t.vid(*this), t.switch_vertex(*this).vid(*this)}};
	}

	template <>
	std::array<Tuple, 3> WildTriRemesher::element_vertices(const Tuple &t) const
	{
		return oriented_tri_vertices(t);
	}

	template <>
	std::array<size_t, 3> WildTriRemesher::element_vids(const Tuple &t) const
	{
		return oriented_tri_vids(t);
	}

	template <>
	std::array<size_t, 3> WildTriRemesher::orient_preserve_element_reorder(
		const std::array<size_t, 3> &conn, const size_t v0) const
	{
		return wmtk::orient_preserve_tri_reorder(conn, v0);
	}

	template <>
	std::vector<Tuple> WildTriRemesher::get_one_ring_elements_for_vertex(const Tuple &t) const
	{
		return get_one_ring_tris_for_vertex(t);
	}

	template <>
	std::vector<Tuple> WildTriRemesher::get_one_ring_boundary_edges_for_vertex(const Tuple &v) const
	{
		std::vector<Tuple> edges;
		for (const auto &e : get_one_ring_edges_for_vertex(v))
			if (is_boundary_edge(e))
				edges.push_back(e);
		return edges;
	}

	template <>
	std::vector<Tuple> WildTriRemesher::get_incident_elements_for_edge(const Tuple &t) const
	{
		std::vector<Tuple> tris{{t}};
		if (t.switch_face(*this))
			tris.push_back(t.switch_face(*this).value());
		return tris;
	}

	template <>
	CollapseEdgeTo WildTriRemesher::collapse_boundary_edge_to(const Tuple &e) const
	{
		const int eid = e.eid(*this);
		const int v0i = e.vid(*this);
		const int v1i = e.switch_vertex(*this).vid(*this);

		const Eigen::Vector2d &v0 = vertex_attrs[v0i].rest_position;
		const Eigen::Vector2d &v1 = vertex_attrs[v1i].rest_position;

		const int boundary_id = boundary_attrs[eid].boundary_id;

		const auto is_collinear = [&](const Tuple &e0) {
			const size_t e0_id = e0.eid(*this);
			const size_t v2_id = e0.vid(*this);
			const size_t v3_id = e0.switch_vertex(*this).vid(*this);
			return e0_id != eid
				   && boundary_attrs[e0_id].boundary_id == boundary_id
				   && is_body_boundary_edge(e0)
				   && utils::are_edges_collinear(
					   v0, v1, vertex_attrs[v2_id].rest_position,
					   vertex_attrs[v3_id].rest_position);
		};

		const std::vector<Tuple> v0_edges = get_one_ring_edges_for_vertex(e);
		const bool is_v0_collinear = std::any_of(v0_edges.begin(), v0_edges.end(), is_collinear);

		const std::vector<Tuple> v1_edges = get_one_ring_edges_for_vertex(e.switch_vertex(*this));
		const bool is_v1_collinear = std::any_of(v1_edges.begin(), v1_edges.end(), is_collinear);

		if (!is_v0_collinear && !is_v1_collinear)
			return CollapseEdgeTo::ILLEGAL; // only collapse boundary edges that have collinear neighbors
		else if (!is_v0_collinear)
			return CollapseEdgeTo::V0;
		else if (!is_v1_collinear)
			return CollapseEdgeTo::V1;
		else
			return CollapseEdgeTo::MIDPOINT; // collapse to midpoint if both points are collinear
	}

	template <>
	WildTriRemesher::EdgeAttributes &WildTriRemesher::edge_attr(const size_t e_id)
	{
		return boundary_attrs[e_id];
	}

	template <>
	const WildTriRemesher::EdgeAttributes &WildTriRemesher::edge_attr(const size_t e_id) const
	{
		return boundary_attrs[e_id];
	}

} // namespace polyfem::mesh