#include <polyfem/mesh/remesh/WildRemesh2D.hpp>

namespace polyfem::mesh
{
	namespace
	{
		void map_edge_split_edge_attributes(
			WildRemeshing2D &m,
			const WildRemeshing2D::Tuple &new_vertex,
			const WildRemeshing2D::EdgeMap<WildRemeshing2D::EdgeAttributes> &old_edges,
			const size_t old_v0_id,
			const size_t old_v1_id)
		{
			using Tuple = WildRemeshing2D::Tuple;
			using EdgeAttributes = WildRemeshing2D::EdgeAttributes;

			const EdgeAttributes old_split_edge = old_edges.at(std::make_pair(
				std::min(old_v0_id, old_v1_id), std::max(old_v0_id, old_v1_id)));
			const EdgeAttributes interior_edge; // default

			const size_t new_vid = new_vertex.vid(m);

			const std::vector<Tuple> new_faces = m.get_one_ring_tris_for_vertex(new_vertex);
			for (const auto &new_face : new_faces)
			{
				for (int i = 0; i < 3; i++)
				{
					const Tuple e = m.tuple_from_edge(new_face.fid(m), i);

					size_t v0_id = e.vid(m);
					size_t v1_id = e.switch_vertex(m).vid(m);
					if (v0_id > v1_id)
						std::swap(v0_id, v1_id);

					assert(v0_id != new_vid); // new_vid should have a higher id than any other vertex
					if (v1_id == new_vid)
					{
						m.edge_attrs[e.eid(m)] =
							(v0_id == old_v0_id || v0_id == old_v1_id) ? old_split_edge : interior_edge;
					}
					else
					{
						m.edge_attrs[e.eid(m)] = old_edges.at(std::make_pair(v0_id, v1_id));
					}
				}
			}
		}

		void map_edge_split_face_attributes(
			WildRemeshing2D &m,
			const WildRemeshing2D::Tuple &new_vertex,
			const std::vector<WildRemeshing2D::FaceAttributes> &old_faces)
		{
			using Tuple = WildRemeshing2D::Tuple;

			Tuple nav = new_vertex.switch_face(m).value();
			m.face_attrs[nav.fid(m)] = old_faces[0];
			nav = nav.switch_face(m).value();
			m.face_attrs[nav.fid(m)] = old_faces[0];
			nav = nav.switch_edge(m);
			if (nav.switch_face(m))
			{
				nav = nav.switch_face(m).value();
				m.face_attrs[nav.fid(m)] = old_faces[1];
				nav = nav.switch_edge(m).switch_face(m).value();
				m.face_attrs[nav.fid(m)] = old_faces[1];

#ifndef NDEBUG
				nav = nav.switch_edge(m).switch_face(m).value();
				assert(m.face_attrs[nav.fid(m)].body_id == old_faces[0].body_id);
#endif
			}
		}
	} // namespace

	bool WildRemeshing2D::split_edge_before(const Tuple &e)
	{
		if (!super::split_edge_before(e))
			return false;

		// Dont split if the edge is too small
		if (edge_length(e) < 1e-6)
			return false;

		// Cache necessary local data
		edge_cache = EdgeOperationCache::split(*this, e);

		return true;
	}

	bool WildRemeshing2D::split_edge_after(const Tuple &t)
	{
		// 0) perform operation (done before this function)

		// 1a) Update rest position of new vertex
		VertexAttributes &new_vertex_attr = vertex_attrs[t.vid(*this)];
		const auto &[old_v0_id, v0] = edge_cache.v0();
		const auto &[old_v1_id, v1] = edge_cache.v1();
		new_vertex_attr.rest_position = (v0.rest_position + v1.rest_position) / 2.0;
		new_vertex_attr.fixed = v0.fixed && v1.fixed;
		new_vertex_attr.partition_id = v0.partition_id; // TODO: what should this be?

		// 1b) Assign edge attributes to the new edges
		map_edge_split_edge_attributes(*this, t, edge_cache.edges(), old_v0_id, old_v1_id);
		map_edge_split_face_attributes(*this, t, edge_cache.faces());

		// 2) Project quantaties so to minimize the L2 error
		new_vertex_attr.position = (v0.position + v1.position) / 2.0;
		new_vertex_attr.projection_quantities =
			(v0.projection_quantities + v1.projection_quantities) / 2.0;

		// 3) Perform a local relaxation of the n-ring to get an estimate of the
		//    energy decrease.
		return local_relaxation(t, n_ring_size);
	}

} // namespace polyfem::mesh