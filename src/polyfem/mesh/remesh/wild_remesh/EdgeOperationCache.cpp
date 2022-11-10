#include <polyfem/mesh/remesh/WildRemesh2D.hpp>

#include <wmtk/utils/TupleUtils.hpp>

namespace polyfem::mesh
{
	namespace
	{
		void insert_edges_of_face(
			WildRemeshing2D &m,
			const WildRemeshing2D::Tuple &t,
			WildRemeshing2D::EdgeMap<WildRemeshing2D::EdgeAttributes> &edge_map)
		{
			for (auto i = 0; i < 3; i++)
			{
				WildRemeshing2D::Tuple e = m.tuple_from_edge(t.fid(m), i);
				size_t v0 = e.vid(m);
				size_t v1 = e.switch_vertex(m).vid(m);
				if (v0 > v1)
					std::swap(v0, v1);
				edge_map[std::make_pair(v0, v1)] = m.edge_attrs[e.eid(m)];
			}
		}
	} // namespace

	WildRemeshing2D::EdgeOperationCache WildRemeshing2D::EdgeOperationCache::split(
		WildRemeshing2D &m, const Tuple &t)
	{
		EdgeOperationCache cache;

		cache.m_v0.first = t.vid(m);
		cache.m_v1.first = t.switch_vertex(m).vid(m);

		cache.m_v0.second = m.vertex_attrs[cache.m_v0.first];
		cache.m_v1.second = m.vertex_attrs[cache.m_v1.first];

		insert_edges_of_face(m, t, cache.m_edges);
		cache.m_faces.push_back(m.face_attrs[t.fid(m)]);

		if (t.switch_face(m))
		{
			const Tuple t1 = t.switch_face(m).value();
			insert_edges_of_face(m, t1, cache.m_edges);
			cache.m_faces.push_back(m.face_attrs[t1.fid(m)]);
		}

		return cache;
	}

	WildRemeshing2D::EdgeOperationCache WildRemeshing2D::EdgeOperationCache::collapse(
		WildRemeshing2D &m, const Tuple &t)
	{
		EdgeOperationCache cache;

		cache.m_v0.first = t.vid(m);
		cache.m_v1.first = t.switch_vertex(m).vid(m);

		cache.m_v0.second = m.vertex_attrs[cache.m_v0.first];
		cache.m_v1.second = m.vertex_attrs[cache.m_v1.first];

		// Cache all edges of the faces in the one-ring of the edge
		std::vector<Tuple> edge_one_ring_faces = m.get_one_ring_tris_for_vertex(t);
		const std::vector<Tuple> tmp = m.get_one_ring_tris_for_vertex(t);
		edge_one_ring_faces.reserve(edge_one_ring_faces.size() + tmp.size());
		edge_one_ring_faces.insert(edge_one_ring_faces.end(), tmp.begin(), tmp.end());

		for (const auto &face : edge_one_ring_faces)
		{
			insert_edges_of_face(m, face, cache.m_edges);
		}

		// Cache the faces adjacent to the edge
		cache.m_faces.push_back(m.face_attrs[t.fid(m)]);
		if (t.switch_face(m))
			cache.m_faces.push_back(m.face_attrs[t.switch_face(m)->fid(m)]);

		return cache;
	}

	WildRemeshing2D::EdgeOperationCache WildRemeshing2D::EdgeOperationCache::swap(
		WildRemeshing2D &m, const Tuple &t)
	{
		EdgeOperationCache cache;

		cache.m_v0.first = t.vid(m);
		cache.m_v1.first = t.switch_vertex(m).vid(m);

		cache.m_v0.second = m.vertex_attrs[cache.m_v0.first];
		cache.m_v1.second = m.vertex_attrs[cache.m_v1.first];

		insert_edges_of_face(m, t, cache.m_edges);
		cache.m_faces.push_back(m.face_attrs[t.fid(m)]);

		assert(t.switch_face(m));
		const Tuple t1 = t.switch_face(m).value();
		insert_edges_of_face(m, t1, cache.m_edges);
		cache.m_faces.push_back(m.face_attrs[t1.fid(m)]);

		return cache;
	}
} // namespace polyfem::mesh