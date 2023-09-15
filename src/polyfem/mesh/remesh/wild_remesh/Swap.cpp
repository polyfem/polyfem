#include <polyfem/mesh/remesh/PhysicsRemesher.hpp>
#include <polyfem/mesh/remesh/wild_remesh/OperationCache.hpp>
#include <polyfem/utils/GeometryUtils.hpp>

namespace polyfem::mesh
{
	template <class WMTKMesh>
	void WildRemesher<WMTKMesh>::cache_swap_edge(const Tuple &e)
	{
		if constexpr (std::is_same_v<WMTKMesh, wmtk::TriMesh>)
			op_cache = TriOperationCache::swap_edge(*this, e);
		else
			op_cache = TetOperationCache::swap_32(*this, e);
	}

	template <class WMTKMesh>
	bool WildRemesher<WMTKMesh>::swap_edge_before(const Tuple &e)
	{
		if (!WMTKMesh::swap_edge_before(e))
			return false;

		const int v0i = e.vid(*this);
		const int v1i = e.switch_vertex(*this).vid(*this);

		if (is_body_boundary_edge(e))
		{
			executor.m_cnt_fail--; // do not count this as a failed swap
			return false;
		}

		if constexpr (std::is_same_v<WMTKMesh, wmtk::TriMesh>)
		{
			const int v2i = opposite_vertex_on_face(e).vid(*this);
			const int v3i = opposite_vertex_on_face(*e.switch_face(*this)).vid(*this);

			const double f0_area = std::abs(utils::triangle_area_2D(vertex_attrs[v0i].rest_position, vertex_attrs[v1i].rest_position, vertex_attrs[v2i].rest_position));
			const double f1_area = std::abs(utils::triangle_area_2D(vertex_attrs[v0i].rest_position, vertex_attrs[v1i].rest_position, vertex_attrs[v3i].rest_position));
			const double f2_area = std::abs(utils::triangle_area_2D(vertex_attrs[v0i].rest_position, vertex_attrs[v2i].rest_position, vertex_attrs[v3i].rest_position));
			const double f3_area = std::abs(utils::triangle_area_2D(vertex_attrs[v1i].rest_position, vertex_attrs[v2i].rest_position, vertex_attrs[v3i].rest_position));

			const double total_area = f0_area + f1_area;
			if (f2_area < 1e-1 * total_area || f3_area < 1e-1 * total_area)
			{
				executor.m_cnt_fail--; // do not count this as a failed swap
				return false;
			}

			// const double current_area_ratio = f0_area < f1_area ? (f1_area / f0_area) : (f0_area / f1_area);
			// const double future_area_ratio = f2_area < f3_area ? (f3_area / f2_area) : (f2_area / f3_area);

			// if (future_area_ratio > 100 * current_area_ratio)
			// {
			// 	executor.m_cnt_fail--; // do not count this as a failed swap
			// 	return false;
			// }
		}
		// TODO: ratio of elements

		cache_swap_edge(e);

		return true;
	}

	bool PhysicsTriRemesher::swap_edge_before(const Tuple &e)
	{
		if (!Super::swap_edge_before(e)) // NOTE: also calls cache_swap_edge
			return false;

		if (this->edge_attr(e.eid(*this)).op_attempts++ >= this->max_op_attempts
			|| this->edge_attr(e.eid(*this)).op_depth >= args["swap"]["max_depth"].template get<int>())
		{
			this->executor.m_cnt_fail--; // do not count this as a failed swap
			return false;
		}

		const VectorNd &v0 = vertex_attrs[e.vid(*this)].rest_position;
		const VectorNd &v1 = vertex_attrs[e.switch_vertex(*this).vid(*this)].rest_position;
		this->op_cache->local_energy = local_mesh_energy((v0 + v1) / 2);

		return true;
	}

	template <>
	void WildTriRemesher::map_edge_swap_edge_attributes(const Tuple &e)
	{
		const auto &old_edges = op_cache->edges();
		for (const Tuple &e : get_edges_for_elements({{e, e.switch_face(*this).value()}}))
		{
			size_t v0_id = e.vid(*this);
			size_t v1_id = e.switch_vertex(*this).vid(*this);

			const auto iter = old_edges.find({{v0_id, v1_id}});
			if (iter != old_edges.end())
			{
				boundary_attrs[e.eid(*this)] = iter->second;
			}
			else
			{
				assert(e.switch_face(*this).has_value());
				// swapped interior edge
				boundary_attrs[e.eid(*this)] =
					old_edges.find({{op_cache->v0().first, op_cache->v1().first}})->second;
			}
		}
	}

	template <>
	void WildTriRemesher::map_edge_swap_element_attributes(const Tuple &e)
	{
		assert(op_cache->faces()[0].body_id == op_cache->faces()[1].body_id);
		element_attrs[e.fid(*this)] = op_cache->faces()[0];
		element_attrs[e.switch_face(*this)->fid(*this)] = op_cache->faces()[1];
	}

	template <>
	bool WildTriRemesher::swap_edge_after(const Tuple &e)
	{
		if (!wmtk::TriMesh::swap_edge_after(e))
			return false;

		// 0) perform operation (done before this function)

		// 1a) Update rest position of new vertex
		// Nothing to do because there is no new vertex.

		// 1b) Assign edge attributes to the new edge
		map_edge_swap_edge_attributes(e);

		// 1c) Assign boundary attributes to the new edges/faces
		// Nothing to do because we disallow boundary edge swaps.

		// 1c) Assign face attributes to the new faces
		map_edge_swap_element_attributes(e);

		// 2) Interpolate quantaties for a good initialization to the local relaxation
		// Nothing to do because there is no new vertex.

		// Check the new faces are not inverted
		assert(e.switch_face(*this).has_value());
		if (is_inverted(e) || element_volume(e) < 1e-16
			|| is_inverted(e.switch_face(*this).value())
			|| element_volume(e.switch_face(*this).value()) < 1e-16)
		{
			return false;
		}

		// No need to check for intersections because we disallow boundary edge swaps.

		// ~2) Project quantities so to minimize the L2 error~ (done after all operations)
		// project_quantities(); // also projects positions

		return true;
	}

	template <>
	bool WildTetRemesher::swap_edge_after(const Tuple &e)
	{
		log_and_throw_error("WildTetRemesher::swap_edge_after not implemented!");
	}

	bool PhysicsTriRemesher::swap_edge_after(const Tuple &e)
	{
		utils::Timer timer(this->timings["Swap edges after"]);
		timer.start();
		if (!Super::swap_edge_after(e))
			return false;
		// local relaxation has its own timers
		timer.stop();

		// 3) Perform a local relaxation of the n-ring to get an estimate of the
		//    energy decrease/increase.
		const VectorNd edge_midpoint =
			(vertex_attrs[e.vid(*this)].rest_position
			 + vertex_attrs[e.switch_vertex(*this).vid(*this)].rest_position)
			/ 2;
		return local_relaxation(edge_midpoint, args["swap"]["acceptance_tolerance"])
			   && invariants(std::vector<Tuple>());
	}

	// -------------------------------------------------------------------------

	double compute_valence_energy(
		const WildTriRemesher &m, std::string op, const WildTriRemesher::Tuple &t)
	{
		using Tuple = WildTriRemesher::Tuple;
		std::array<std::pair<Tuple, int>, 4> valences;
		valences[0].first = t;
		valences[1].first = t.switch_vertex(m);
		valences[2].first = t.switch_edge(m).switch_vertex(m);
		assert(t.switch_face(m).has_value());
		valences[3].first = t.switch_face(m)->switch_edge(m).switch_vertex(m);
		for (auto &[t, v] : valences)
			v = m.get_valence_for_vertex(t);

		double cost_before_swap = 0.0, cost_after_swap = 0.0;
		for (int i = 0; i < valences.size(); i++)
		{
			const auto &[vert, valence] = valences[i];
			const int ideal_val = m.is_boundary_vertex(vert) ? 4 : 6;
			cost_before_swap += std::pow(double(valence - ideal_val), 2);
			cost_after_swap +=
				std::pow(double(i < 2 ? (valence - ideal_val - 1) : (valence - ideal_val + 1)), 2);
		}
		return cost_before_swap - cost_after_swap;
	}

	double compute_area_ratio_energy(
		const WildTriRemesher &m, std::string op, const WildTriRemesher::Tuple &t)
	{
		const double f0_area = m.element_volume(t);
		assert(t.switch_face(m).has_value());
		const double f1_area = m.element_volume(*t.switch_face(m));
		return std::max(f0_area, f1_area) / std::min(f0_area, f1_area);
	}

	void PhysicsTriRemesher::swap_edges()
	{
		executor.priority = compute_valence_energy;
		executor.should_renew = [](auto val) { return (val > 0); };
		executor.renew_neighbor_tuples = [&](const WildRemesher &m, std::string op, const std::vector<Tuple> &tris) -> Operations {
			assert(tris.size() == 1);
			const Tuple &t = tris[0];
			Operations new_ops;
			for (const Tuple &t : this->get_edges_for_elements({{t, t.switch_face(*this).value()}}))
				if (!is_body_boundary_edge(t))
					new_ops.emplace_back(op, t);
			return new_ops;
		};

		// for (int i = 0; i < 20 && (i > 0 || executor.cnt_success() != 0); i++)
		{
			std::vector<Tuple> included_edges;
			{
				const std::vector<Tuple> edges = this->get_edges();
				std::copy_if(edges.begin(), edges.end(), std::back_inserter(included_edges), [this](const Tuple &e) {
					// return this->edge_attr(e.eid(*this)).energy_rank == Super::EdgeAttributes::EnergyRank::BOTTOM;
					return !is_body_boundary_edge(e);
				});
			}

			if (included_edges.empty())
				return;

			Operations swaps;
			swaps.reserve(included_edges.size());
			for (const Tuple &e : included_edges)
				swaps.emplace_back("edge_swap", e);

			executor(*this, swaps);
		}
	}

	// =========================================================================
	// Map attributes
	// =========================================================================

	// ------------------------------------------------------------------------
	// Template specializations

	template class WildRemesher<wmtk::TriMesh>;
	template class WildRemesher<wmtk::TetMesh>;
	template class PhysicsRemesher<wmtk::TriMesh>;
	template class PhysicsRemesher<wmtk::TetMesh>;

} // namespace polyfem::mesh