#include <polyfem/mesh/remesh/WildTriRemesher.hpp>
#include <polyfem/utils/Logger.hpp>

#include <wmtk/ExecutionScheduler.hpp>

namespace polyfem::mesh
{
	namespace
	{
		bool is_edge_fixed(const WildTriRemesher &m, const WildTriRemesher::Tuple &e)
		{
			const int v0i = e.vid(m);
			const int v1i = e.switch_vertex(m).vid(m);
			return m.vertex_attrs[v0i].fixed && m.vertex_attrs[v1i].fixed;
		}

		bool are_edges_collinear(
			const Eigen::Vector2d &ea,
			const Eigen::Vector2d &eb,
			const double tol = 1e-6)
		{
			return abs(ea.dot(eb) / (ea.norm() * eb.norm())) > 1 - tol;
		}
	} // namespace

	bool WildTriRemesher::collapse_edge_before(const Tuple &t)
	{
		if (!wmtk::TriMesh::collapse_edge_before(t))
			return false;

		const int eid = t.eid(*this);
		const int v0i = t.vid(*this);
		const int v1i = t.switch_vertex(*this).vid(*this);

		const Eigen::Vector2d &v0 = vertex_attrs[v0i].rest_position;
		const Eigen::Vector2d &v1 = vertex_attrs[v1i].rest_position;

		cache_before();
		op_cache = TriOperationCache::collapse_edge(*this, t);

		const bool is_v0_fixed = vertex_attrs[v0i].fixed;
		const bool is_v1_fixed = vertex_attrs[v1i].fixed;

		// boundary edge
		if (is_v0_fixed && is_v1_fixed)
		{
			const int boundary_id = boundary_attrs[eid].boundary_id;

			const std::vector<Tuple> v0_edges = get_one_ring_edges_for_vertex(t);
			const bool is_v0_collinear = std::any_of(v0_edges.begin(), v0_edges.end(), [&](const Tuple &e) {
				const Eigen::Vector2d &v2 = vertex_attrs[e.switch_vertex(*this).vid(*this)].rest_position;
				const size_t other_eid = e.eid(*this);
				return other_eid != eid && boundary_attrs[other_eid].boundary_id == boundary_id
					   && is_edge_fixed(*this, e) && are_edges_collinear(v1 - v0, v2 - v0);
			});

			const std::vector<Tuple> v1_edges = get_one_ring_edges_for_vertex(t.switch_vertex(*this));
			const bool is_v1_collinear = std::any_of(v1_edges.begin(), v1_edges.end(), [&](const Tuple &e) {
				const Eigen::Vector2d &v2 = vertex_attrs[e.switch_vertex(*this).vid(*this)].rest_position;
				const size_t other_eid = e.eid(*this);
				return other_eid != eid && boundary_attrs[other_eid].boundary_id == boundary_id
					   && is_edge_fixed(*this, e) && are_edges_collinear(v0 - v1, v2 - v1);
			});

			// only collapse boundary edges that have collinear neighbors
			if (!is_v0_collinear && !is_v1_collinear)
				return false;

			// collapse to midpoint if both points are collinear
			op_cache.collapse_to =
				!is_v0_collinear ? CollapseEdgeTo::V0 : //
					(!is_v1_collinear ? CollapseEdgeTo::V1 : CollapseEdgeTo::MIDPOINT);
		}
		else
		{
			op_cache.collapse_to =
				is_v0_fixed ? CollapseEdgeTo::V0 : //
					(is_v1_fixed ? CollapseEdgeTo::V1 : CollapseEdgeTo::MIDPOINT);
		}

		switch (op_cache.collapse_to)
		{
		case CollapseEdgeTo::V0:
			op_cache.local_energy = local_mesh_energy(v0);
			break;
		case CollapseEdgeTo::V1:
			op_cache.local_energy = local_mesh_energy(v1);
			break;
		case CollapseEdgeTo::MIDPOINT:
			op_cache.local_energy = local_mesh_energy((v0 + v1) / 2);
			break;
		default:
			assert(false);
		}

		return true;
	}

	bool WildTriRemesher::collapse_edge_after(const Tuple &t)
	{
		// 0) perform operation (done before this function)

		const auto &[old_v0_id, v0] = op_cache.v0();
		const auto &[old_v1_id, v1] = op_cache.v1();
		const auto &old_edges = op_cache.edges();
		const size_t new_vid = t.vid(*this);

		// 1a) Update rest position of new vertex
		switch (op_cache.collapse_to)
		{
		case CollapseEdgeTo::V0:
		{
			vertex_attrs[new_vid].rest_position = v0.rest_position;
			vertex_attrs[new_vid].partition_id = v0.partition_id;
			vertex_attrs[new_vid].fixed = v0.fixed;
			break;
		}
		case CollapseEdgeTo::V1:
		{
			vertex_attrs[new_vid].rest_position = v1.rest_position;
			vertex_attrs[new_vid].partition_id = v1.partition_id;
			vertex_attrs[new_vid].fixed = v1.fixed;
			break;
		}
		case CollapseEdgeTo::MIDPOINT:
		{
			// TODO: using an average midpoint for now
			vertex_attrs[new_vid].rest_position = (v0.rest_position + v1.rest_position) / 2.0;
			vertex_attrs[new_vid].partition_id = v0.partition_id; // TODO: what should this be?
			vertex_attrs[new_vid].fixed = v0.fixed || v1.fixed;
			break;
		}
		default:
			assert(false);
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

			std::array<size_t, 2> old_edge{{v0_id, old_v0_id}};
			if (old_edges.find(old_edge) == old_edges.end())
			{
				old_edge = {{v0_id, old_v1_id}};
				assert(old_edges.find(old_edge) != old_edges.end());
			}
			boundary_attrs[e_id] = old_edges.at(old_edge);
		}

		// Nothing to do for the face attributes because no new faces are created.

		// 2) Interpolate quantaties for a good initialization to the local relaxation

		// initial guess for the new vertex position
		switch (op_cache.collapse_to)
		{
		case CollapseEdgeTo::V0:
		{
			vertex_attrs[new_vid].position = v0.position;
			vertex_attrs[new_vid].projection_quantities = v0.projection_quantities;
			break;
		}
		case CollapseEdgeTo::V1:
		{
			vertex_attrs[new_vid].position = v1.position;
			vertex_attrs[new_vid].projection_quantities = v1.projection_quantities;
			break;
		}
		case CollapseEdgeTo::MIDPOINT:
		{
			// TODO: using an average midpoint for now
			vertex_attrs[new_vid].position = (v0.position + v1.position) / 2.0;
			vertex_attrs[new_vid].projection_quantities = (v0.projection_quantities + v1.projection_quantities) / 2.0;
			break;
		}
		default:
			assert(false);
		}

		// Check the interpolated poisition does not violate invariants
		if (!invariants(get_one_ring_elements_for_vertex(t)))
			return false;

		// ~2) Project quantities so to minimize the L2 error~ (done after all operations)
		// project_quantities(); // also projects positions

		// 3) Perform a local relaxation of the n-ring to get an estimate of the
		//    energy decrease/increase.
		return local_relaxation(
			t, op_cache.local_energy,
			collapse_relative_tolerance,
			collapse_absolute_tolerance);
	}

} // namespace polyfem::mesh