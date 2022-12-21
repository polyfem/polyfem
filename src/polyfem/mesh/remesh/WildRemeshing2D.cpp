#include "WildRemeshing2D.hpp"

#include <wmtk/utils/TupleUtils.hpp>

#include <igl/predicates/predicates.h>

namespace polyfem::mesh
{
	void WildRemeshing2D::create_mesh(const size_t num_vertices, const Eigen::MatrixXi &triangles)
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

	// -------------------------------------------------------------------------

	bool WildRemeshing2D::is_inverted(const Tuple &loc) const
	{
		// Get the vertices ids
		const std::array<size_t, 3> vids = oriented_tri_vids(loc);

		igl::predicates::exactinit();

		// Use igl for checking orientation
		igl::predicates::Orientation rest_orientation = igl::predicates::orient2d(
			vertex_attrs[vids[0]].rest_position,
			vertex_attrs[vids[1]].rest_position,
			vertex_attrs[vids[2]].rest_position);
		igl::predicates::Orientation deformed_orientation = igl::predicates::orient2d(
			vertex_attrs[vids[0]].position,
			vertex_attrs[vids[1]].position,
			vertex_attrs[vids[2]].position);

		// The element is inverted if it not positive (i.e. it is negative or it is degenerate)
		return rest_orientation != igl::predicates::Orientation::POSITIVE
			   || deformed_orientation != igl::predicates::Orientation::POSITIVE;
	}

	// -------------------------------------------------------------------------

	std::vector<WildRemeshing2D::Tuple> WildRemeshing2D::new_edges_after(
		const std::vector<Tuple> &tris) const
	{
		std::vector<Tuple> new_edges;

		for (auto t : tris)
		{
			for (auto j = 0; j < 3; j++)
			{
				new_edges.push_back(tuple_from_edge(t.fid(*this), j));
			}
		}
		wmtk::unique_edge_tuples(*this, new_edges);
		return new_edges;
	}

	// -------------------------------------------------------------------------

	std::vector<WildRemeshing2D::Tuple> WildRemeshing2D::boundary_edges() const
	{
		std::vector<Tuple> boundary_edges;
		for (const Tuple &e : get_edges())
			if (!e.switch_face(*this))
				boundary_edges.push_back(e);
		return boundary_edges;
	}

} // namespace polyfem::mesh