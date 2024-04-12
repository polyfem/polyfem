#include "OperationCache.hpp"

namespace polyfem::mesh
{
	namespace
	{
		void insert_edges_of_face(
			WildTriRemesher &m,
			const WildTriRemesher::Tuple &t,
			WildTriRemesher::EdgeMap<TriOperationCache::EdgeAttributes> &edge_map)
		{
			for (auto i = 0; i < 3; i++)
			{
				WildTriRemesher::Tuple e = m.tuple_from_edge(t.fid(m), i);
				const size_t v0 = e.vid(m);
				const size_t v1 = e.switch_vertex(m).vid(m);
				edge_map[{{v0, v1}}] = m.boundary_attrs[e.eid(m)];
			}
		}

		void insert_edges_of_tet(
			WildTetRemesher &m,
			const WildTetRemesher::Tuple &t,
			WildTriRemesher::EdgeMap<TetOperationCache::EdgeAttributes> &edge_map)
		{
			for (auto i = 0; i < 6; i++)
			{
				WildTetRemesher::Tuple e = m.tuple_from_edge(t.tid(m), i);
				const size_t v0 = e.vid(m);
				const size_t v1 = e.switch_vertex(m).vid(m);
				edge_map[{{v0, v1}}] = m.edge_attr(e.eid(m));
			}
		}

		void insert_faces_of_tet(
			WildTetRemesher &m,
			const WildTetRemesher::Tuple &t,
			WildTetRemesher::FaceMap<TetOperationCache::FaceAttributes> &face_map)
		{
			for (auto i = 0; i < 4; i++)
			{
				WildTetRemesher::Tuple f = m.tuple_from_face(t.tid(m), i);
				std::array<WildTetRemesher::Tuple, 3> vs = m.get_face_vertices(f);
				face_map[{{vs[0].vid(m), vs[1].vid(m), vs[2].vid(m)}}] = m.boundary_attrs[f.fid(m)];
			}
		}
	} // namespace

	std::shared_ptr<TriOperationCache> TriOperationCache::split_edge(WildTriRemesher &m, const Tuple &t)
	{
		std::shared_ptr<TriOperationCache> cache = std::make_shared<TriOperationCache>();

		cache->m_v0.first = t.vid(m);
		cache->m_v1.first = t.switch_vertex(m).vid(m);

		cache->m_v0.second = m.vertex_attrs[cache->m_v0.first];
		cache->m_v1.second = m.vertex_attrs[cache->m_v1.first];

		insert_edges_of_face(m, t, cache->m_edges);
		cache->m_faces.push_back(m.element_attrs[t.fid(m)]);

		if (t.switch_face(m))
		{
			const Tuple t1 = t.switch_face(m).value();
			insert_edges_of_face(m, t1, cache->m_edges);
			cache->m_faces.push_back(m.element_attrs[t1.fid(m)]);
		}

		cache->m_is_boundary_op = m.is_boundary_edge(t);

		return cache;
	}

	std::shared_ptr<TriOperationCache> TriOperationCache::collapse_edge(WildTriRemesher &m, const Tuple &t)
	{
		std::shared_ptr<TriOperationCache> cache = std::make_shared<TriOperationCache>();

		cache->m_v0.first = t.vid(m);
		cache->m_v1.first = t.switch_vertex(m).vid(m);

		cache->m_v0.second = m.vertex_attrs[cache->m_v0.first];
		cache->m_v1.second = m.vertex_attrs[cache->m_v1.first];

		// Cache all edges of the faces in the one-ring of the edge
		std::vector<Tuple> edge_one_ring_faces = m.get_one_ring_tris_for_vertex(t);
		const std::vector<Tuple> tmp = m.get_one_ring_tris_for_vertex(t.switch_vertex(m));
		edge_one_ring_faces.reserve(edge_one_ring_faces.size() + tmp.size());
		edge_one_ring_faces.insert(edge_one_ring_faces.end(), tmp.begin(), tmp.end());

		for (const auto &face : edge_one_ring_faces)
		{
			insert_edges_of_face(m, face, cache->m_edges);
		}

		// Cache the faces adjacent to the edge
		cache->m_faces.push_back(m.element_attrs[t.fid(m)]);
		if (t.switch_face(m))
			cache->m_faces.push_back(m.element_attrs[t.switch_face(m)->fid(m)]);

		cache->m_is_boundary_op = m.is_boundary_edge(t);

		return cache;
	}

	std::shared_ptr<TriOperationCache> TriOperationCache::swap_edge(WildTriRemesher &m, const Tuple &t)
	{
		std::shared_ptr<TriOperationCache> cache = std::make_shared<TriOperationCache>();

		cache->m_v0.first = t.vid(m);
		cache->m_v1.first = t.switch_vertex(m).vid(m);

		cache->m_v0.second = m.vertex_attrs[cache->m_v0.first];
		cache->m_v1.second = m.vertex_attrs[cache->m_v1.first];

		insert_edges_of_face(m, t, cache->m_edges);
		cache->m_faces.push_back(m.element_attrs[t.fid(m)]);

		assert(t.switch_face(m));
		const Tuple t1 = t.switch_face(m).value();
		insert_edges_of_face(m, t1, cache->m_edges);
		cache->m_faces.push_back(m.element_attrs[t1.fid(m)]);

		cache->m_is_boundary_op = m.is_boundary_edge(t);

		return cache;
	}

	std::shared_ptr<TetOperationCache> TetOperationCache::split_edge(WildTetRemesher &m, const Tuple &e)
	{
		std::shared_ptr<TetOperationCache> cache = std::make_shared<TetOperationCache>();

		cache->m_v0.first = e.vid(m);
		cache->m_v1.first = e.switch_vertex(m).vid(m);

		cache->m_v0.second = m.vertex_attrs[cache->m_v0.first];
		cache->m_v1.second = m.vertex_attrs[cache->m_v1.first];

		const std::vector<Tuple> tets = m.get_incident_tets_for_edge(e);
		assert(tets.size() >= 1);
		cache->m_tets.reserve(tets.size());
		for (const Tuple &t : tets)
		{
			insert_edges_of_tet(m, t, cache->m_edges);
			insert_faces_of_tet(m, t, cache->m_faces);
			cache->m_tets[m.oriented_tet_vids(t)] = m.element_attrs[t.tid(m)];
		}

		cache->m_is_boundary_op = m.is_boundary_edge(e);

		return cache;
	}

	std::shared_ptr<TetOperationCache> TetOperationCache::collapse_edge(WildTetRemesher &m, const Tuple &e)
	{
		std::shared_ptr<TetOperationCache> cache = std::make_shared<TetOperationCache>();

		cache->m_v0.first = e.vid(m);
		cache->m_v1.first = e.switch_vertex(m).vid(m);

		cache->m_v0.second = m.vertex_attrs[cache->m_v0.first];
		cache->m_v1.second = m.vertex_attrs[cache->m_v1.first];

		const std::vector<Tuple> tets = m.get_one_ring_tets_for_edge(e);
		assert(tets.size() >= 1);
		cache->m_tets.reserve(tets.size());
		for (const Tuple &t : tets)
		{
			insert_edges_of_tet(m, t, cache->m_edges);
			insert_faces_of_tet(m, t, cache->m_faces);
			cache->m_tets[m.oriented_tet_vids(t)] = m.element_attrs[t.tid(m)];
		}

		cache->m_is_boundary_op = m.is_boundary_edge(e);

		return cache;
	}
} // namespace polyfem::mesh