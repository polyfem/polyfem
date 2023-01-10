#include <polyfem/mesh/remesh/WildTriRemesher.hpp>

namespace polyfem::mesh
{
	bool WildTriRemesher::swap_edge_before(const Tuple &t)
	{
		const int v0i = t.vid(*this);
		const int v1i = t.switch_vertex(*this).vid(*this);

		if (vertex_attrs[v0i].fixed && vertex_attrs[v1i].fixed)
			return false;

		cache_before();
		// Cache necessary local data
		op_cache = TriOperationCache::swap_edge(*this, t);

		return true;
	}

	bool WildTriRemesher::swap_edge_after(const Tuple &t)
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

				const auto iter = old_edges.find({{v0_id, v1_id}});

				if (iter != old_edges.end())
				{
					boundary_attrs[e.eid(*this)] = iter->second;
				}
				else
				{
					assert(e.switch_face(*this));
					boundary_attrs[e.eid(*this)] = BoundaryAttributes(); // interior edge
				}
			}
		}

		element_attrs[t.fid(*this)] = op_cache.faces()[0];
		element_attrs[t.switch_face(*this)->fid(*this)] = op_cache.faces()[1];

		// There is no non-inversion check in project_quantities, so check it here.
		if (!invariants(get_one_ring_elements_for_vertex(t)))
			return false;

		// ~2) Project quantities so to minimize the L2 error~ (done after all operations)
		// project_quantities(); // also projects positions

		// 3) Perform a local relaxation of the n-ring to get an estimate of the
		//    energy decrease.
		assert(false); // TODO: set local_energy in _before
		return local_relaxation(
			t, op_cache.local_energy,
			swap_relative_tolerance,
			swap_absolute_tolerance);
	}

} // namespace polyfem::mesh