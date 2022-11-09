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
				m_vertex_boundary_ids[m_global_to_local[edge.vid(m)]].insert(boundary_id);
				m_vertex_boundary_ids[m_global_to_local[edge.switch_vertex(m).vid(m)]].insert(boundary_id);
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
		std::vector<Tuple> triangles = {{center}};
		std::vector<int> depths{{0}};
		std::unordered_set<size_t> visited{{center.fid(m)}};

		// Loop around a vertex until we return to the starting triangle or hit a boundary.
		const auto helper = [&](const int depth, std::optional<Tuple> &nav) -> void {
			assert(nav);
			const size_t start_fid = nav->fid(m);
			do
			{
				nav = nav->switch_edge(m);
				if (visited.find(nav->fid(m)) == visited.end())
				{
					triangles.push_back(nav->switch_vertex(m));
					depths.push_back(depth + 1);
					visited.insert(nav->fid(m));
				}
				nav = nav->switch_face(m);
			} while (nav && nav->fid(m) != start_fid);
		};

		int i = 0;
		while (i < triangles.size())
		{
			const Tuple t = triangles[i];
			const int depth = depths[i];
			i++;

			if (depth > n)
				continue;

			// NOTE: This only works for manifold meshes.
			std::optional<Tuple> nav = t;
			helper(depth, nav);
			if (!nav) // Hit a boundary, so loop in the opposite direction.
			{
				// Switch edge to cause the helper loop to go the opposite direction
				nav = t.switch_edge(m);
				helper(depth, nav);
			}
		}

		return LocalMesh(m, triangles);
	}
} // namespace polyfem::mesh