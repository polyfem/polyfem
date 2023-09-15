#include "WildRemesher.hpp"

#include <polyfem/mesh/remesh/wild_remesh/OperationCache.hpp>
#include <polyfem/utils/GeometryUtils.hpp>

#include <wmtk/utils/TupleUtils.hpp>
#include <wmtk/utils/TetraQualityUtils.hpp>

#include <igl/predicates/predicates.h>
#include <igl/edges.h>

namespace polyfem::mesh
{
	using Tuple = WildTetRemesher::Tuple;

	template <>
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

	template <>
	WildTetRemesher::BoundaryMap<int> WildTetRemesher::boundary_ids() const
	{
		const std::vector<Tuple> faces = get_faces();
		FaceMap<int> boundary_ids;
		for (const Tuple &face : faces)
		{
			const size_t f0 = face.vid(*this);
			const size_t f1 = face.switch_vertex(*this).vid(*this);
			const size_t f2 = opposite_vertex_on_face(face).vid(*this);

			boundary_ids[{{f0, f1, f2}}] = boundary_attrs[face.fid(*this)].boundary_id;
		}
		return boundary_ids;
	}

	template <>
	Eigen::MatrixXi WildTetRemesher::boundary_edges() const
	{
		const std::vector<Tuple> boundary_face_tuples = boundary_facets();
		std::vector<Tuple> boundary_edge_tuples;
		for (const Tuple &f : boundary_face_tuples)
		{
			boundary_edge_tuples.push_back(f);
			boundary_edge_tuples.push_back(f.switch_edge(*this));
			boundary_edge_tuples.push_back(f.switch_vertex(*this).switch_edge(*this));
		}
		wmtk::unique_edge_tuples(*this, boundary_edge_tuples);

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
	Eigen::MatrixXi WildTetRemesher::boundary_faces() const
	{
		const std::vector<Tuple> boundary_face_tuples = boundary_facets();
		Eigen::MatrixXi BF(boundary_face_tuples.size(), 3);
		for (int i = 0; i < BF.rows(); ++i)
		{
			const std::array<Tuple, 3> vs = get_face_vertices(boundary_face_tuples[i]);
			for (int j = 0; j < 3; ++j)
				BF(i, j) = vs[j].vid(*this);
		}
		if (obstacle().n_faces() > 0)
			utils::append_rows(BF, obstacle().f().array() + vert_capacity());
		return BF;
	}

	template <>
	std::vector<Tuple> WildTetRemesher::get_facets() const
	{
		return get_faces();
	}

	template <>
	std::vector<Tuple> WildTetRemesher::get_elements() const
	{
		return get_tets();
	}

	template <>
	bool WildTetRemesher::is_rest_inverted(const Tuple &loc) const
	{
		// Get the vertices ids
		const std::array<size_t, 4> vids = oriented_tet_vids(loc);

		igl::predicates::exactinit();

		// Use igl for checking orientation
		const igl::predicates::Orientation orientation = igl::predicates::orient3d(
			vertex_attrs[vids[0]].rest_position, vertex_attrs[vids[1]].rest_position,
			vertex_attrs[vids[2]].rest_position, vertex_attrs[vids[3]].rest_position);

		// neg result == pos tet (tet origin from geogram delaunay)
		return orientation != igl::predicates::Orientation::NEGATIVE;
	}

	template <>
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

	template <>
	double WildTetRemesher::element_volume(const Tuple &e) const
	{
		const std::array<size_t, 4> vids = oriented_tet_vids(e);
		return utils::tetrahedron_volume(
			vertex_attrs[vids[0]].rest_position,
			vertex_attrs[vids[1]].rest_position,
			vertex_attrs[vids[2]].rest_position,
			vertex_attrs[vids[3]].rest_position);
	}

	template <>
	size_t WildTetRemesher::facet_id(const Tuple &t) const
	{
		return t.fid(*this);
	}

	template <>
	size_t WildTetRemesher::element_id(const Tuple &t) const
	{
		return t.tid(*this);
	}

	template <>
	Tuple WildTetRemesher::tuple_from_facet(size_t elem_id, int local_facet_id) const
	{
		return tuple_from_face(elem_id, local_facet_id);
	}

	template <>
	Tuple WildTetRemesher::tuple_from_element(size_t elem_id) const
	{
		return tuple_from_tet(elem_id);
	}

	template <>
	bool WildTetRemesher::is_boundary_facet(const Tuple &t) const
	{
		return t.is_boundary_face(*this);
	}

	template <>
	bool WildTetRemesher::is_boundary_vertex(const Tuple &v) const
	{
		for (const Tuple &t : get_one_ring_tets_for_vertex(v))
		{
			for (int fi = 0; fi < FACETS_PER_ELEMENT; ++fi)
			{
				if (is_boundary_facet(tuple_from_facet(t.tid(*this), fi)))
					return true;
			}
		}
		return false;
	}

	template <>
	bool WildTetRemesher::is_body_boundary_vertex(const Tuple &v) const
	{
		log_and_throw_error("WildTetRemesher::is_body_boundary_vertex() not implemented!");
	}

	template <>
	bool WildTetRemesher::is_boundary_edge(const Tuple &e) const
	{
		return e.is_boundary_edge(*this);
	}

	template <>
	bool WildTetRemesher::is_body_boundary_edge(const Tuple &e) const
	{
		const size_t tid = e.tid(*this);
		const int body_id = element_attrs[tid].body_id;

		std::optional<Tuple> t = e.switch_tetrahedron(*this);
		while (t && element_attrs[t->tid(*this)].body_id == body_id && t->tid(*this) != tid)
			t = t->switch_face(*this).switch_tetrahedron(*this);

		return !t.has_value() || element_attrs[t->tid(*this)].body_id != body_id;
	}

	template <>
	bool WildTetRemesher::is_boundary_op() const
	{
		return op_cache->is_boundary_op();
	}

	template <>
	std::array<Tuple, 3> WildTetRemesher::facet_vertices(const Tuple &t) const
	{
		return get_face_vertices(t);
	}

	template <>
	std::array<size_t, 3> WildTetRemesher::facet_vids(const Tuple &t) const
	{
		return {{
			t.vid(*this),
			t.switch_vertex(*this).vid(*this),
			opposite_vertex_on_face(t).vid(*this),
		}};
	}

	template <>
	std::array<Tuple, 4> WildTetRemesher::element_vertices(const Tuple &t) const
	{
		return oriented_tet_vertices(t);
	}

	template <>
	std::array<size_t, 4> WildTetRemesher::element_vids(const Tuple &t) const
	{
		return oriented_tet_vids(t);
	}

	template <>
	std::array<size_t, 4> WildTetRemesher::orient_preserve_element_reorder(
		const std::array<size_t, 4> &conn, const size_t v0) const
	{
		return wmtk::orient_preserve_tet_reorder(conn, v0);
	}

	template <>
	std::vector<Tuple> WildTetRemesher::get_one_ring_elements_for_vertex(const Tuple &t) const
	{
		return get_one_ring_tets_for_vertex(t);
	}

	template <>
	std::vector<Tuple> WildTetRemesher::get_incident_elements_for_edge(const Tuple &t) const
	{
		return get_incident_tets_for_edge(t);
	}

	template <>
	std::vector<Tuple> WildTetRemesher::get_one_ring_boundary_faces_for_vertex(const Tuple &v) const
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

	template <>
	std::vector<Tuple> WildTetRemesher::get_one_ring_boundary_edges_for_vertex(const Tuple &v) const
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

	template <>
	std::array<Tuple, 2> WildTetRemesher::get_boundary_faces_for_edge(const Tuple &e) const
	{
		assert(e.is_boundary_edge(*this));

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

	template <>
	CollapseEdgeTo WildTetRemesher::collapse_boundary_edge_to(const Tuple &e) const
	{
		// TODO: handle body boundary edges
		assert(e.is_boundary_edge(*this));

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

	template <>
	WildTetRemesher::EdgeAttributes &WildTetRemesher::edge_attr(const size_t e_id)
	{
		return edge_attrs[e_id];
	}

	template <>
	const WildTetRemesher::EdgeAttributes &WildTetRemesher::edge_attr(const size_t e_id) const
	{
		return edge_attrs[e_id];
	}

} // namespace polyfem::mesh