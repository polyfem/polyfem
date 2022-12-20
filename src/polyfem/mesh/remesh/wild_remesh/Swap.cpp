#include <polyfem/mesh/remesh/WildRemeshing2D.hpp>

namespace polyfem::mesh
{
	bool WildRemeshing2D::swap_edge_before(const Tuple &t)
	{
		if (!wmtk::TriMesh::swap_edge_before(t))
			return false;

		const int v0i = t.vid(*this);
		const int v1i = t.switch_vertex(*this).vid(*this);

		if (vertex_attrs[v0i].fixed && vertex_attrs[v1i].fixed)
			return false;

		cache_before();
		// Cache necessary local data
		op_cache = OperationCache2D::swap(*this, t);

		return true;
	}

	bool WildRemeshing2D::swap_edge_after(const Tuple &t)
	{
		// 0) perform operation (done before this function)

		// 1a) Update rest position of new vertex
		//     No new vertex, so nothing to do.

		// 1b) Assign edge attributes to the new edges
		const auto &old_edges = op_cache.edges();
		const std::array<size_t, 2> face_ids{{t.fid(*this), t.switch_face(*this)->fid(*this)}};
		for (const size_t fid : face_ids)
		{
			for (int leid = 0; leid < 3; leid++)
			{
				const Tuple e = tuple_from_edge(fid, leid);

				size_t v0_id = e.vid(*this);
				size_t v1_id = e.switch_vertex(*this).vid(*this);
				if (v0_id > v1_id)
					std::swap(v0_id, v1_id);
				std::pair<size_t, size_t> edge(v0_id, v1_id);

				if (old_edges.find(edge) != old_edges.end())
				{
					edge_attrs[e.eid(*this)] = old_edges.at(edge);
				}
				else
				{
					assert(e.switch_face(*this));
					edge_attrs[e.eid(*this)] = EdgeAttributes(); // interior edge
				}
			}
		}

		face_attrs[t.fid(*this)] = op_cache.faces()[0];
		face_attrs[t.switch_face(*this)->fid(*this)] = op_cache.faces()[1];

		// 2) Project quantities so to minimize the L2 error
		project_quantities(); // also projects positions

		// 3) Perform a local relaxation of the n-ring to get an estimate of the
		//    energy decrease.
		return local_relaxation(t, n_ring_size);
	}

} // namespace polyfem::mesh