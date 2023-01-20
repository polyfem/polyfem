#include "WildTriRemesher.hpp"

#include <polyfem/utils/GeometryUtils.hpp>

#include <igl/predicates/predicates.h>

namespace polyfem::mesh
{
	WildTriRemesher::WildTriRemesher(
		const State &state,
		const Eigen::MatrixXd &obstacle_displacements,
		const Eigen::MatrixXd &obstacle_vals,
		const double current_time,
		const double starting_energy)
		: Super(state, obstacle_displacements, obstacle_vals, current_time, starting_energy)
	{
	}

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

	// smooth_before/smooth_after in wild_remesh/Smooth.cpp

	// split_edge_after in wild_remesh/Split.cpp

	// collapse_edge_before/collapse_edge_after in wild_remesh/Collapse.cpp

	// swap_edge_before/swap_edge_after in wild_remesh/Swap.cpp

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

	double WildTriRemesher::element_volume(const Tuple &e) const
	{
		const std::array<size_t, 3> vids = oriented_tri_vids(e);
		return utils::triangle_area_2D(
			vertex_attrs[vids[0]].rest_position,
			vertex_attrs[vids[1]].rest_position,
			vertex_attrs[vids[2]].rest_position);
	}

	std::vector<WildTriRemesher::Tuple> WildTriRemesher::boundary_facets() const
	{
		const std::vector<Tuple> edges = get_edges();
		std::vector<Tuple> boundary_edges;
		std::copy_if(edges.begin(), edges.end(), std::back_inserter(boundary_edges), [this](const Tuple &e) {
			return is_boundary_edge(e);
		});
		return boundary_edges;
	}

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

	bool WildTriRemesher::is_edge_on_body_boundary(const Tuple &e) const
	{
		const auto adj_face = e.switch_face(*this);
		return !adj_face.has_value()
			   || element_attrs[element_id(e)].body_id
					  != element_attrs[element_id(*adj_face)].body_id;
	}

	bool WildTriRemesher::is_vertex_on_body_boundary(const Tuple &v) const
	{
		for (const auto &e : get_one_ring_edges_for_vertex(v))
			if (is_edge_on_body_boundary(e))
				return true;
		return false;
	}

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
				   && is_edge_on_body_boundary(e0)
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

	// map_edge_split_boundary_attributes/map_edge_split_element_attributes in wild_remesh/Split.cpp

} // namespace polyfem::mesh