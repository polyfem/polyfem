#include "LocalMesh.hpp"

namespace polyfem::mesh
{
	LocalMesh::LocalMesh(const WildRemeshing2D &m, const std::vector<Tuple> &triangle_tuples)
	{
		m_triangles.resize(triangle_tuples.size(), 3);

		for (int fi = 0; fi < num_triangles(); fi++)
		{
			const Tuple &t = triangle_tuples[fi];

			const std::array<Tuple, 3> face = m.oriented_tri_vertices(t);
			for (int i = 0; i < 3; ++i)
			{
				const size_t vi = face[i].vid(m);
				if (m_global_to_local.find(vi) == m_global_to_local.end())
					m_global_to_local[vi] = m_global_to_local.size();
				m_triangles(fi, i) = m_global_to_local[vi];
			}

			m_body_ids.push_back(m.face_attrs[t.fid(m)].body_id);
		}

		m_rest_positions.resize(num_vertices(), DIM);
		m_positions.resize(num_vertices(), DIM);
		m_prev_positions.resize(num_vertices(), DIM);
		m_prev_velocities.resize(num_vertices(), DIM);
		m_prev_accelerations.resize(num_vertices(), DIM);
		for (const auto &[glob_vi, loc_vi] : m_global_to_local)
		{
			m_rest_positions.row(loc_vi) = m.vertex_attrs[glob_vi].rest_position;
			m_positions.row(loc_vi) = m.vertex_attrs[glob_vi].position;

			assert(m.vertex_attrs[glob_vi].projection_quantities.cols() == 3);

			m_prev_positions.row(loc_vi) = m.vertex_attrs[glob_vi].prev_displacement();
			m_prev_velocities.row(loc_vi) = m.vertex_attrs[glob_vi].prev_velocity();
			m_prev_accelerations.row(loc_vi) = m.vertex_attrs[glob_vi].prev_acceleration();
		}

		m_vertex_boundary_ids.resize(num_vertices());
		for (const Tuple &t : triangle_tuples)
		{
			const std::array<Tuple, 3> edges{{
				t,
				t.switch_edge(m),
				t.switch_vertex(m).switch_edge(m),
			}};

			for (const Tuple &edge : edges)
			{
				const int boundary_id = m.edge_attrs[edge.eid(m)].boundary_id;
				size_t v0 = m_global_to_local[edge.vid(m)], v1 = m_global_to_local[edge.switch_vertex(m).vid(m)];
				if (v0 > v1)
					std::swap(v0, v1);

				m_vertex_boundary_ids[v0].insert(boundary_id);
				m_vertex_boundary_ids[v1].insert(boundary_id);

				m_is_edge_on_global_boundary[{v0, v1}] = !bool(edge.switch_face(m));
			}
		}

		m_local_to_global.resize(m_global_to_local.size(), -1);
		for (const auto &[glob_vi, loc_vi] : m_global_to_local)
		{
			assert(loc_vi < m_local_to_global.size());
			m_local_to_global[loc_vi] = glob_vi;
		}
	}

	LocalMesh LocalMesh::n_ring(const WildRemeshing2D &m, const WildRemeshing2D::Tuple &center, int n)
	{
		std::vector<Tuple> triangles = m.get_one_ring_tris_for_vertex(center);
		std::unordered_set<size_t> visited_vertices{{center.vid(m)}};
		std::unordered_set<size_t> visited_faces;
		for (const auto &triangle : triangles)
			visited_faces.insert(triangle.fid(m));

		std::vector<Tuple> new_triangles = triangles;

		for (int i = 1; i < n; i++)
		{
			std::vector<Tuple> new_new_triangles;
			for (const auto &t : new_triangles)
			{
				const std::array<Tuple, 3> vs = m.oriented_tri_vertices(t);
				for (int vi = 0; vi < 3; vi++)
				{
					const Tuple &v = vs[vi];
					if (visited_vertices.find(v.vid(m)) != visited_vertices.end())
						continue;
					visited_vertices.insert(v.vid(m));

					std::vector<wmtk::TriMesh::Tuple> tmp = m.get_one_ring_tris_for_vertex(v);
					for (auto &t1 : tmp)
					{
						if (visited_faces.find(t1.fid(m)) != visited_faces.end())
							continue;
						visited_faces.insert(t1.fid(m));
						triangles.push_back(t1);
						new_new_triangles.push_back(t1);
					}
				}
			}
			new_triangles = new_new_triangles;
			if (new_triangles.empty())
				break;
		}

		return LocalMesh(m, triangles);
	}

	bool LocalMesh::is_edge_on_global_boundary(size_t v0, size_t v1) const
	{
		assert(v0 < num_vertices() && v1 < num_vertices());
		if (v0 > v1)
			std::swap(v0, v1);
		return m_is_edge_on_global_boundary.at({v0, v1});
	}
} // namespace polyfem::mesh