#include "WildTetRemesher.hpp"

#include <polyfem/utils/GeometryUtils.hpp>

#include <wmtk/utils/TupleUtils.hpp>

#include <igl/predicates/predicates.h>
#include <igl/edges.h>

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
		p_edge_attrs = &edge_attrs;
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

	// split_edge_after in wild_remesh/Split.cpp

	Eigen::MatrixXi WildTetRemesher::boundary_edges() const
	{
		const Eigen::MatrixXi BF = boundary_faces();
		Eigen::MatrixXi BE;
		igl::edges(BF, BE);
		if (obstacle().n_edges() > 0)
			utils::append_rows(BE, obstacle().e().array() + vert_capacity());
		return BE;
	}

	Eigen::MatrixXi WildTetRemesher::boundary_faces() const
	{
		const std::vector<Tuple> faces = get_faces();
		int num_boundary_faces = 0;
		Eigen::MatrixXi BF(faces.size(), 3);
		for (int i = 0; i < faces.size(); ++i)
		{
			const Tuple &f = faces[i];
			if (f.switch_tetrahedron(*this).has_value()) // not a boundary face
				continue;
			const std::array<Tuple, 3> vs = get_face_vertices(f);
			BF(num_boundary_faces, 0) = vs[0].vid(*this);
			BF(num_boundary_faces, 1) = vs[1].vid(*this);
			BF(num_boundary_faces, 2) = vs[2].vid(*this);
			num_boundary_faces++;
		}
		BF.conservativeResize(num_boundary_faces, 3);
		if (obstacle().n_faces() > 0)
			utils::append_rows(BF, obstacle().f().array() + vert_capacity());
		return BF;
	}

	bool WildTetRemesher::is_inverted(const Tuple &loc) const
	{
		// Get the vertices ids
		const std::array<size_t, 4> vids = oriented_tet_vids(loc);

		igl::predicates::exactinit();

		for (int i = 0; i < n_quantities() / 3 + 2; ++i)
		{
			// Use igl for checking orientation
			const igl::predicates::Orientation orientation = igl::predicates::orient3d(
				vertex_attrs[vids[0]].position_i(i), vertex_attrs[vids[1]].position_i(i),
				vertex_attrs[vids[2]].position_i(i), vertex_attrs[vids[3]].position_i(i));

			// neg result == pos tet (tet origin from geogram delaunay)
			if (orientation != igl::predicates::Orientation::NEGATIVE)
				return true;
		}

		return false;
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

	bool WildTetRemesher::is_edge_on_boundary(const Tuple &e) const
	{
		const size_t tid = e.tid(*this);

		std::optional<Tuple> t = e.switch_tetrahedron(*this);
		while (t && t->tid(*this) != tid)
			t = t->switch_face(*this).switch_tetrahedron(*this);

		return !t.has_value();
	}

	bool WildTetRemesher::is_edge_on_body_boundary(const Tuple &e) const
	{
		const size_t tid = e.tid(*this);
		const int body_id = element_attrs[tid].body_id;

		std::optional<Tuple> t = e.switch_tetrahedron(*this);
		while (t && element_attrs[t->tid(*this)].body_id == body_id && t->tid(*this) != tid)
			t = t->switch_face(*this).switch_tetrahedron(*this);

		return !t.has_value() || element_attrs[t->tid(*this)].body_id != body_id;
	}

	std::vector<WildTetRemesher::Tuple>
	WildTetRemesher::get_one_ring_boundary_faces_for_vertex(const Tuple &v) const
	{
		const size_t vid = v.vid(*this);

		std::vector<Tuple> faces;
		for (const Tuple &t : get_one_ring_tets_for_vertex(v))
		{
			const size_t tid = t.tid(*this);
			for (int fi = 0; fi < 4; ++fi)
			{
				const Tuple f = tuple_from_face(tid, fi);

				if (!f.is_boundary_face(*this))
					continue;

				// check if the face contains the vertex
				for (const Tuple &fv : get_face_vertices(f))
				{
					if (fv.vid(*this) == vid)
					{
						faces.push_back(f);
						break;
					}
				}
			}
		}
		unique_face_tuples(*this, faces);
		return faces;
	}

	std::vector<WildTetRemesher::Tuple>
	WildTetRemesher::get_one_ring_boundary_edges_for_vertex(const Tuple &v) const
	{
		const size_t vid = v.vid(*this);

		std::vector<Tuple> edges;
		for (const Tuple &f : get_one_ring_boundary_faces_for_vertex(v))
		{
			const size_t fid = f.fid(*this);

			const std::array<Tuple, 3> face_edges{{
				f,
				f.switch_vertex(*this).switch_edge(*this),
				f.switch_edge(*this),
			}};

			for (const Tuple &e : face_edges)
			{
				// check if the edge contains the vertex
				if (e.vid(*this) == vid || e.switch_vertex(*this).vid(*this) == vid)
				{
					edges.push_back(e);
				}
			}
		}
		unique_edge_tuples(*this, edges);
		return edges;
	}

	std::array<WildTetRemesher::Tuple, 2>
	WildTetRemesher::get_boundary_faces_for_edge(const Tuple &e) const
	{
		assert(is_edge_on_boundary(e));

		const size_t tid = e.tid(*this);

		// Find the two boundary faces that the edge belongs to
		std::array<Tuple, 2> faces{{e, e.switch_face(*this)}};
		for (Tuple &f : faces)
		{
			do
			{
				std::optional st = f.switch_tetrahedron(*this);
				if (!st)
					break;
				f = st->switch_face(*this);
			} while (f.tid(*this) != tid);
			assert(f.is_boundary_face(*this));
		}
		assert(faces[0].fid(*this) != faces[1].fid(*this));
		return faces;
	}

	CollapseEdgeTo WildTetRemesher::collapse_boundary_edge_to(const Tuple &e) const
	{
		// TODO: handle body boundary edges
		assert(is_edge_on_boundary(e));

		const std::array<Tuple, 2> boundary_faces = get_boundary_faces_for_edge(e);

		const Eigen::Vector3d &v0 = vertex_attrs[e.vid(*this)].rest_position;
		const Eigen::Vector3d &v1 = vertex_attrs[e.switch_vertex(*this).vid(*this)].rest_position;
		const Eigen::Vector3d &v2 = vertex_attrs[opposite_vertex_on_face(boundary_faces[0]).vid(*this)].rest_position;
		const Eigen::Vector3d &v3 = vertex_attrs[opposite_vertex_on_face(boundary_faces[1]).vid(*this)].rest_position;
		assert((v2.array() != v3.array()).any());

		const int boundary_id0 = boundary_attrs[boundary_faces[0].fid(*this)].boundary_id;
		const int boundary_id1 = boundary_attrs[boundary_faces[1].fid(*this)].boundary_id;

		const std::vector<Tuple> v0_boundary_faces = get_one_ring_boundary_faces_for_vertex(e);
		const std::vector<Tuple> v1_boundary_faces = get_one_ring_boundary_faces_for_vertex(e.switch_vertex(*this));

		bool is_v0_movable, is_v1_movable;
		if (boundary_id0 != boundary_id1 || !utils::are_triangles_coplanar(v0, v1, v2, v0, v1, v3))
		{
			const auto &is_collinear = [&](const Tuple &e1) {
				assert(e.is_boundary_edge(*this));
				return e.eid(*this) != e1.eid(*this)
					   && utils::are_edges_collinear(
						   v0, v1, vertex_attrs[e1.vid(*this)].rest_position,
						   vertex_attrs[e1.switch_vertex(*this).vid(*this)].rest_position);
			};

			const auto &is_coplanar = [&](const Tuple &f) {
				assert(f.is_boundary_face(*this));
				const std::array<Tuple, 3> vs = get_face_vertices(f);
				const Eigen::Vector3d &v4 = vertex_attrs[vs[0].vid(*this)].rest_position;
				const Eigen::Vector3d &v5 = vertex_attrs[vs[1].vid(*this)].rest_position;
				const Eigen::Vector3d &v6 = vertex_attrs[vs[2].vid(*this)].rest_position;
				return (boundary_attrs[f.fid(*this)].boundary_id == boundary_id0
						&& utils::are_triangles_coplanar(v0, v1, v2, v4, v5, v6))
					   || (boundary_attrs[f.fid(*this)].boundary_id == boundary_id1
						   && utils::are_triangles_coplanar(v0, v1, v3, v4, v5, v6));
			};

			const std::vector<Tuple> v0_boundary_edges = get_one_ring_boundary_edges_for_vertex(e);
			is_v0_movable = std::any_of(v0_boundary_edges.begin(), v0_boundary_edges.end(), is_collinear)
							&& std::all_of(v0_boundary_faces.begin(), v0_boundary_faces.end(), is_coplanar);

			const std::vector<Tuple> v1_boundary_edges = get_one_ring_boundary_edges_for_vertex(e.switch_vertex(*this));
			is_v1_movable = std::any_of(v1_boundary_edges.begin(), v1_boundary_edges.end(), is_collinear)
							&& std::all_of(v1_boundary_faces.begin(), v1_boundary_faces.end(), is_coplanar);
		}
		else
		{
			const auto &is_coplanar = [&](const Tuple &f) {
				const std::array<Tuple, 3> vs = get_face_vertices(f);
				const Eigen::Vector3d &v4 = vertex_attrs[vs[0].vid(*this)].rest_position;
				const Eigen::Vector3d &v5 = vertex_attrs[vs[1].vid(*this)].rest_position;
				const Eigen::Vector3d &v6 = vertex_attrs[vs[2].vid(*this)].rest_position;
				return boundary_attrs[f.fid(*this)].boundary_id == boundary_id0
					   && utils::are_triangles_coplanar(v0, v1, v2, v4, v5, v6);
			};

			is_v0_movable = std::all_of(v0_boundary_faces.begin(), v0_boundary_faces.end(), is_coplanar);
			is_v1_movable = std::all_of(v1_boundary_faces.begin(), v1_boundary_faces.end(), is_coplanar);
		}

		if (!is_v0_movable && !is_v1_movable)
			return CollapseEdgeTo::ILLEGAL;
		else if (!is_v0_movable)
			return CollapseEdgeTo::V0;
		else if (!is_v1_movable)
			return CollapseEdgeTo::V1;
		else
			return CollapseEdgeTo::MIDPOINT; // collapse to midpoint if both points are movable
	}

	// map_edge_split_boundary_attributes/map_edge_split_element_attributes in wild_remesh/Split.cpp

} // namespace polyfem::mesh