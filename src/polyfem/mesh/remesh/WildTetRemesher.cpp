#include "WildTetRemesher.hpp"

#include <polyfem/utils/GeometryUtils.hpp>

#include <wmtk/utils/TupleUtils.hpp>

#include <igl/predicates/predicates.h>

namespace polyfem::mesh
{
	/// @brief Construct a new WildTetRemesher object
	/// @param state Simulation current state
	WildTetRemesher::WildTetRemesher(
		const State &state,
		const Eigen::MatrixXd &obstacle_displacements,
		const Eigen::MatrixXd &obstacle_vals,
		const double current_time,
		const double starting_energy)
		: Super(state, obstacle_displacements, obstacle_vals, current_time, starting_energy)
	{
	}

	void WildTetRemesher::init_attributes_and_connectivity(
		const size_t num_vertices, const Eigen::MatrixXi &tetrahedra)
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

	// smooth_before/smooth_after in wild_remesh/Smooth.cpp

	bool WildTetRemesher::is_inverted(const Tuple &loc) const
	{
		// Get the vertices ids
		const std::array<size_t, 4> vids = oriented_tet_vids(loc);

		igl::predicates::exactinit();

		// Use igl for checking orientation
		igl::predicates::Orientation rest_orientation = igl::predicates::orient3d(
			vertex_attrs[vids[0]].rest_position, vertex_attrs[vids[1]].rest_position,
			vertex_attrs[vids[2]].rest_position, vertex_attrs[vids[3]].rest_position);
		igl::predicates::Orientation deformed_orientation = igl::predicates::orient3d(
			vertex_attrs[vids[0]].position, vertex_attrs[vids[1]].position,
			vertex_attrs[vids[2]].position, vertex_attrs[vids[3]].position);

		// neg result == pos tet (tet origin from geogram delaunay)
		return rest_orientation != igl::predicates::Orientation::NEGATIVE
			   || deformed_orientation != igl::predicates::Orientation::NEGATIVE;
	}

	double WildTetRemesher::element_volume(const Tuple &e) const
	{
		const std::array<size_t, 4> vids = oriented_tet_vids(e);
		return utils::tetrahedron_volume(
			vertex_attrs[vids[0]].rest_position,
			vertex_attrs[vids[1]].rest_position,
			vertex_attrs[vids[2]].rest_position,
			vertex_attrs[vids[3]].rest_position);
	}

	std::vector<WildTetRemesher::Tuple> WildTetRemesher::boundary_facets() const
	{
		std::vector<Tuple> boundary_faces;
		for (const Tuple &f : get_faces())
			if (!f.switch_tetrahedron(*this))
				boundary_faces.push_back(f);
		return boundary_faces;
	}

	// map_edge_split_boundary_attributes/map_edge_split_element_attributes in wild_remesh/Split.cpp

} // namespace polyfem::mesh