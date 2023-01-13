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

	Eigen::MatrixXi WildTriRemesher::boundary_edges() const
	{
		const std::vector<Tuple> edges = get_edges();
		int num_boundary_edges = 0;
		Eigen::MatrixXi BE(edges.size(), 2);
		for (int i = 0; i < edges.size(); ++i)
		{
			const Tuple &e = edges[i];
			if (e.switch_face(*this).has_value()) // not a boundary edge
				continue;
			BE(num_boundary_edges, 0) = e.vid(*this);
			BE(num_boundary_edges, 1) = e.switch_vertex(*this).vid(*this);
			num_boundary_edges++;
		}
		BE.conservativeResize(num_boundary_edges, 2);
		if (obstacle().n_edges() > 0)
			utils::append_rows(BE, obstacle().e().array() + vert_capacity());
		return BE;
	}

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
		std::vector<Tuple> boundary_edges;
		for (const Tuple &e : get_edges())
			if (!e.switch_face(*this))
				boundary_edges.push_back(e);
		return boundary_edges;
	}

	bool WildTriRemesher::is_edge_on_body_boundary(const Tuple &e) const
	{
		const auto adj_face = e.switch_face(*this);
		return !adj_face.has_value()
			   || element_attrs[element_id(e)].body_id
					  != element_attrs[element_id(*adj_face)].body_id;
	}

	bool WildTriRemesher::is_vertex_on_boundary(const Tuple &v) const
	{
		for (const auto &e : get_one_ring_edges_for_vertex(v))
		{
			if (!e.switch_face(*this).has_value())
				return true;
		}
		return false;
	}

	bool WildTriRemesher::is_vertex_on_body_boundary(const Tuple &v) const
	{
		for (const auto &e : get_one_ring_edges_for_vertex(v))
		{
			if (is_edge_on_body_boundary(e))
				return true;
		}
		return false;
	}

	// map_edge_split_boundary_attributes/map_edge_split_element_attributes in wild_remesh/Split.cpp

} // namespace polyfem::mesh