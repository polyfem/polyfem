#include <polyfem/mesh/remesh/WildRemeshing2D.hpp>
#include <polyfem/utils/Logger.hpp>

#include <wmtk/ExecutionScheduler.hpp>

namespace polyfem::mesh
{
	bool WildRemeshing2D::collapse_edge_before(const Tuple &t)
	{
		if (!wmtk::TriMesh::collapse_edge_before(t))
			return false;

		const int v0i = t.vid(*this);
		const int v1i = t.switch_vertex(*this).vid(*this);

		if (vertex_attrs[v0i].fixed && vertex_attrs[v1i].fixed)
			return false;

		cache_before();
		op_cache = OperationCache2D::collapse(*this, t);

		return true;
	}

	bool WildRemeshing2D::collapse_edge_after(const Tuple &t)
	{
		// 0) perform operation (done before this function)

		const auto &[old_v0_id, v0] = op_cache.v0();
		const auto &[old_v1_id, v1] = op_cache.v1();
		const auto &old_edges = op_cache.edges();
		const size_t new_vid = t.vid(*this);

		// 1a) Update rest position of new vertex
		assert(!(v0.fixed && v1.fixed));
		if (v0.fixed)
		{
			vertex_attrs[new_vid].rest_position = v0.rest_position;
			vertex_attrs[new_vid].partition_id = v0.partition_id;
			vertex_attrs[new_vid].fixed = true;
		}
		else if (v1.fixed)
		{
			vertex_attrs[new_vid].rest_position = v1.rest_position;
			vertex_attrs[new_vid].partition_id = v1.partition_id;
			vertex_attrs[new_vid].fixed = true;
		}
		else
		{
			// TODO: using an average midpoint for now
			vertex_attrs[new_vid].rest_position = (v0.rest_position + v1.rest_position) / 2.0;
			vertex_attrs[new_vid].partition_id = v0.partition_id; // TODO: what should this be?
			vertex_attrs[new_vid].fixed = false;
		}

		// 1b) Assign edge attributes to the new edges
		const std::vector<Tuple> one_ring_edges = get_one_ring_edges_for_vertex(t);
		for (const Tuple &e : one_ring_edges)
		{
			const size_t e_id = e.eid(*this);

			size_t v0_id = e.vid(*this);
			size_t v1_id = e.switch_vertex(*this).vid(*this);
			if (v0_id > v1_id)
				std::swap(v0_id, v1_id);
			assert(v1_id == new_vid); // should be the new vertex because it has a larger id

			std::pair<int, int> old_edge(std::min(v0_id, old_v0_id), std::max(v0_id, old_v0_id));
			if (old_edges.find(old_edge) == old_edges.end())
			{
				old_edge = std::make_pair(std::min(v0_id, old_v1_id), std::max(v0_id, old_v1_id));
				assert(old_edges.find(old_edge) != old_edges.end());
			}
			edge_attrs[e_id] = old_edges.at(old_edge);
		}

		// Nothing to do for the face attributes because no new faces are created.

		// 2) Project quantities so to minimize the L2 error

		// initial guess for the new vertex position
		if (v0.fixed)
		{
			vertex_attrs[new_vid].position = v0.position;
			vertex_attrs[new_vid].projection_quantities = v0.projection_quantities;
		}
		else if (v1.fixed)
		{
			vertex_attrs[new_vid].position = v1.position;
			vertex_attrs[new_vid].projection_quantities = v1.projection_quantities;
		}
		else
		{
			vertex_attrs[new_vid].position = (v0.position + v1.position) / 2.0;
			vertex_attrs[new_vid].projection_quantities = (v0.projection_quantities + v1.projection_quantities) / 2.0;
		}

		project_quantities(); // also projects positions

		// 3) Perform a local relaxation of the n-ring to get an estimate of the
		//    energy decrease.
		return local_relaxation(t, n_ring_size);
	}

} // namespace polyfem::mesh