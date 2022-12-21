#include "WildRemeshing3D.hpp"

#include <wmtk/utils/TupleUtils.hpp>

#include <igl/predicates/predicates.h>

namespace polyfem::mesh
{
	void WildRemeshing3D::create_mesh(const size_t num_vertices, const Eigen::MatrixXi &tetrahedra)
	{
		// Register attributes
		p_vertex_attrs = &vertex_attrs;
		p_face_attrs = &boundary_attrs;
		p_tet_attrs = &element_attrs;

		// Convert from eigen to internal representation
		std::vector<std::array<size_t, 4>> tets(tetrahedra.rows());
		for (int i = 0; i < tetrahedra.rows(); i++)
			for (int j = 0; j < tetrahedra.cols(); j++)
				tets[i][j] = (size_t)tetrahedra(i, j);

		// Initialize the trimesh class which handles connectivity
		wmtk::TetMesh::init(num_vertices, tets);
	}

	// -------------------------------------------------------------------------

	std::vector<WildRemeshing3D::Tuple> WildRemeshing3D::boundary_faces() const
	{
		std::vector<Tuple> boundary_faces;
		for (const Tuple &f : get_faces())
			if (!f.switch_tetrahedron(*this))
				boundary_faces.push_back(f);
		return boundary_faces;
	}

	// -------------------------------------------------------------------------
	// Remeshing operations

	bool WildRemeshing3D::is_inverted(const Tuple &loc) const
	{
		// Get the vertices ids
		const std::array<Tuple, 4> vs = oriented_tet_vertices(loc);

		igl::predicates::exactinit();

		// Use igl for checking orientation
		igl::predicates::Orientation rest_orientation = igl::predicates::orient3d(
			vertex_attrs[vs[0].vid(*this)].rest_position,
			vertex_attrs[vs[1].vid(*this)].rest_position,
			vertex_attrs[vs[2].vid(*this)].rest_position,
			vertex_attrs[vs[3].vid(*this)].rest_position);
		igl::predicates::Orientation deformed_orientation = igl::predicates::orient3d(
			vertex_attrs[vs[0].vid(*this)].position,
			vertex_attrs[vs[1].vid(*this)].position,
			vertex_attrs[vs[2].vid(*this)].position,
			vertex_attrs[vs[3].vid(*this)].position);

		// The element is inverted if it not positive (i.e. it is negative or it is degenerate)
		return rest_orientation != igl::predicates::Orientation::POSITIVE
			   || deformed_orientation != igl::predicates::Orientation::POSITIVE;
	}

	// std::vector<WildRemeshing3D::Tuple> WildRemeshing3D::new_edges_after(
	// 	const std::vector<Tuple> &tris) const
	// {
	// 	std::vector<Tuple> new_edges;

	// 	for (auto t : tris)
	// 	{
	// 		for (auto j = 0; j < 3; j++)
	// 		{
	// 			new_edges.push_back(tuple_from_edge(t.fid(*this), j));
	// 		}
	// 	}
	// 	wmtk::unique_edge_tuples(*this, new_edges);
	// 	return new_edges;
	// }

} // namespace polyfem::mesh