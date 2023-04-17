#include <polyfem/mesh/remesh/PhysicsRemesher.hpp>
#include <polyfem/mesh/remesh/wild_remesh/OperationCache.hpp>
#include <polyfem/utils/Logger.hpp>

#include <wmtk/ExecutionScheduler.hpp>
#include <ipc/ipc.hpp>

namespace polyfem::mesh
{
	template <class WMTKMesh>
	void WildRemesher<WMTKMesh>::cache_collapse_edge(const Tuple &e, const CollapseEdgeTo collapse_to)
	{
		op_cache = decltype(op_cache)::element_type::collapse_edge(*this, e);
		op_cache->collapse_to = collapse_to;
	}

	template <class WMTKMesh>
	bool WildRemesher<WMTKMesh>::collapse_edge_before(const Tuple &t)
	{
		if (!WMTKMesh::collapse_edge_before(t))
			return false;

		const double max_edge_length = std::min(
			args["collapse"]["abs_max_edge_length"].template get<double>(),
			state.starting_min_edge_length
				* args["collapse"]["rel_max_edge_length"].template get<double>());

		double vol_tol;
		if constexpr (std::is_same_v<wmtk::TriMesh, WMTKMesh>)
			vol_tol = std::pow(max_edge_length, 2) / 2;
		else
			vol_tol = std::pow(max_edge_length, 3) / (6 * sqrt(2));

		// Dont collapse if the edge is large
		if (edge_adjacent_element_volumes(t).minCoeff() > vol_tol
			|| rest_edge_length(t) > max_edge_length)
		{
			executor.m_cnt_fail--; // do not count this as a failed collapse
			return false;
		}

		const int v0i = t.vid(*this);
		const int v1i = t.switch_vertex(*this).vid(*this);

		CollapseEdgeTo collapse_to = CollapseEdgeTo::ILLEGAL;
		if (is_body_boundary_edge(t))
		{
			collapse_to = collapse_boundary_edge_to(t);
		}
		else
		{
			// NOTE: .fixed here assumed to be equal to is_vertex_on_body_boundary
			if (vertex_attrs[v0i].fixed && vertex_attrs[v1i].fixed)
				collapse_to = CollapseEdgeTo::ILLEGAL; // interior edge with both vertices fixed means it spans accross the body
			else if (vertex_attrs[v0i].fixed)
				collapse_to = CollapseEdgeTo::V0;
			else if (vertex_attrs[v1i].fixed)
				collapse_to = CollapseEdgeTo::V1;
			else
				collapse_to = CollapseEdgeTo::MIDPOINT;
		}

		if (collapse_to == CollapseEdgeTo::ILLEGAL)
		{
			executor.m_cnt_fail--; // do not count this as a failed collapse
			return false;
		}

		cache_collapse_edge(t, collapse_to);

		return true;
	}

	template <class WMTKMesh>
	bool PhysicsRemesher<WMTKMesh>::collapse_edge_before(const Tuple &t)
	{
		// POLYFEM_REMESHER_SCOPED_TIMER("Collapse edge before");

		if (!Super::collapse_edge_before(t)) // NOTE: also calls cache_collapse_edge
			return false;

		if (this->edge_attr(t.eid(*this)).op_attempts++ >= this->max_op_attempts
			|| this->edge_attr(t.eid(*this)).op_depth >= args["collapse"]["max_depth"].template get<int>())
		{
			this->executor.m_cnt_fail--; // do not count this as a failed collapse
			return false;
		}

		const VectorNd &v0 = vertex_attrs[t.vid(*this)].rest_position;
		const VectorNd &v1 = vertex_attrs[t.switch_vertex(*this).vid(*this)].rest_position;

		switch (this->op_cache->collapse_to)
		{
		case CollapseEdgeTo::V0:
			this->op_cache->local_energy = local_mesh_energy(v0);
			break;
		case CollapseEdgeTo::V1:
			this->op_cache->local_energy = local_mesh_energy(v1);
			break;
		case CollapseEdgeTo::MIDPOINT:
			this->op_cache->local_energy = local_mesh_energy((v0 + v1) / 2);
			break;
		default:
			assert(false);
		}

		return true;
	}

	// -------------------------------------------------------------------------

	template <class WMTKMesh>
	bool WildRemesher<WMTKMesh>::collapse_edge_after(const Tuple &t)
	{
		if (!WMTKMesh::collapse_edge_after(t))
			return false;

		// 0) perform operation (done before this function)

		// 1a) Update rest position of new vertex
		map_edge_collapse_vertex_attributes(t);

		if (state.is_contact_enabled() && is_boundary_vertex(t))
		{
			const double dhat = state.args["contact"]["dhat"].template get<double>();

			// only enforce this invariant if it started valid
			if (state.min_boundary_edge_length >= dhat)
			{
				for (const Tuple &e : get_one_ring_boundary_edges_for_vertex(t))
				{
					if (rest_edge_length(e) < dhat)
						return false; // produced too small edge
				}
			}
		}

		// 1b) Assign edge attributes to the new edges
		map_edge_collapse_boundary_attributes(t);
		// Nothing to do for the element attributes because no new elements are created.

		// 2) Interpolate quantaties for a good initialization to the local relaxation

		// done by map_edge_collapse_vertex_attributes(t);

		// Check the interpolated position does not cause inversions
		for (const Tuple &t1 : get_one_ring_elements_for_vertex(t))
		{
			if (is_inverted(t1) || element_volume(t1) < 1e-16)
				return false;
		}

#ifndef NDEBUG
		// Check the volume of the rest mesh is preserved
		double new_total_volume = 0;
		for (const Tuple &t : get_elements())
			new_total_volume += element_volume(t);
		assert(std::abs(new_total_volume - total_volume) < std::max(1e-12 * total_volume, 1e-12));
#endif

		// Check the interpolated position does not cause intersections
		if (state.is_contact_enabled() && is_boundary_op())
		{
			Eigen::MatrixXd V_rest = rest_positions();
			utils::append_rows(V_rest, obstacle().v());

			ipc::CollisionMesh collision_mesh = ipc::CollisionMesh::build_from_full_mesh(
				V_rest, boundary_edges(), boundary_faces());

#ifndef NDEBUG
			// This should never happen because we only collapse collinear edges on the rest mesh boundary.
			if (ipc::has_intersections(collision_mesh, collision_mesh.rest_positions()))
			{
				write_mesh(state.resolve_output_path("collapse_intersects.vtu"));
				assert(false);
			}
#endif

			Eigen::MatrixXd V = positions();
			if (obstacle().n_vertices())
				utils::append_rows(V, obstacle().v() + obstacle_displacements());
			if (ipc::has_intersections(collision_mesh, collision_mesh.vertices(V)))
			{
				return false;
			}

			Eigen::MatrixXd prev_displacements = projection_quantities().leftCols(n_quantities() / 3);
			utils::append_rows(prev_displacements, obstacle_quantities().leftCols(n_quantities() / 3));
			for (const auto &u : prev_displacements.colwise())
			{
				if (ipc::has_intersections(collision_mesh, collision_mesh.displace_vertices(utils::unflatten(u, dim()))))
				{
					return false;
				}
			}
		}

		// ~2) Project quantities so to minimize the L2 error~ (done after all operations)
		// project_quantities(); // also projects positions

		return true;
	}

	template <class WMTKMesh>
	bool PhysicsRemesher<WMTKMesh>::collapse_edge_after(const Tuple &t)
	{
		utils::Timer timer(this->timings["Collapse edges after"]);
		timer.start();
		if (!Super::collapse_edge_after(t))
			return false;
		// local relaxation has its own timers
		timer.stop();

		// 3) Perform a local relaxation of the n-ring to get an estimate of the
		//    energy decrease/increase.
		return local_relaxation(t, args["collapse"]["acceptance_tolerance"]);
	}

	// -------------------------------------------------------------------------

	template <class WMTKMesh>
	void PhysicsRemesher<WMTKMesh>::collapse_edges()
	{
		const auto collapsable = [this](const Tuple &e) {
			return this->edge_attr(e.eid(*this)).energy_rank == Super::EdgeAttributes::EnergyRank::BOTTOM
				   || this->rest_edge_length(e) < 1e-3 * this->state.starting_min_edge_length; // collapse short edges
		};

		executor.priority = [](const WildRemesher<WMTKMesh> &m, std::string op, const Tuple &t) -> double {
			// return -m.edge_elastic_energy(t); // invert the energy to get a reverse ordering
			return -m.rest_edge_length(t);
		};

		executor.renew_neighbor_tuples = [&](const WildRemesher<WMTKMesh> &m, std::string op, const std::vector<Tuple> &tris) -> Operations {
			const std::vector<Tuple> edges = m.get_edges_for_elements(
				m.get_one_ring_elements_for_vertex(tris[0]));
			Operations new_ops;
			for (auto &e : edges)
				if (collapsable(e))
					new_ops.emplace_back(op, e);
			return new_ops;
		};

		std::vector<Tuple> included_edges;
		{
			const std::vector<Tuple> edges = WMTKMesh::get_edges();
			std::copy_if(edges.begin(), edges.end(), std::back_inserter(included_edges), collapsable);
		}

		if (included_edges.empty())
			return;

		Operations collapses;
		collapses.reserve(included_edges.size());
		for (const Tuple &e : included_edges)
			collapses.emplace_back("edge_collapse", e);

		executor(*this, collapses);
	}

	// =========================================================================
	// Map attributes
	// =========================================================================

	template <class WMTKMesh>
	typename WildRemesher<WMTKMesh>::VertexAttributes
	WildRemesher<WMTKMesh>::VertexAttributes::edge_collapse(
		const VertexAttributes &v0,
		const VertexAttributes &v1,
		const CollapseEdgeTo collapse_to)
	{
		VertexAttributes v;

		switch (collapse_to)
		{
		case CollapseEdgeTo::V0:
		{
			v.rest_position = v0.rest_position;
			v.position = v0.position;
			v.projection_quantities = v0.projection_quantities;
			v.partition_id = v0.partition_id;
			v.fixed = v0.fixed;
			break;
		}
		case CollapseEdgeTo::V1:
		{
			v.rest_position = v1.rest_position;
			v.position = v1.position;
			v.projection_quantities = v1.projection_quantities;
			v.partition_id = v1.partition_id;
			v.fixed = v1.fixed;
			break;
		}
		case CollapseEdgeTo::MIDPOINT:
		{
			// TODO: using an average midpoint for now
			v.rest_position = (v0.rest_position + v1.rest_position) / 2.0;
			v.position = (v0.position + v1.position) / 2.0;
			v.projection_quantities = (v0.projection_quantities + v1.projection_quantities) / 2.0;
			v.partition_id = v0.partition_id; // TODO: what should this be?
			v.fixed = v0.fixed || v1.fixed;
			break;
		}
		default:
			assert(false);
		}

		return v;
	}

	template <>
	void WildTriRemesher::map_edge_collapse_vertex_attributes(const Tuple &t)
	{
		vertex_attrs[t.vid(*this)] = VertexAttributes::edge_collapse(
			op_cache->v0().second, op_cache->v1().second, op_cache->collapse_to);
	}

	template <>
	void WildTetRemesher::map_edge_collapse_vertex_attributes(const Tuple &t)
	{
		vertex_attrs[t.vid(*this)] = VertexAttributes::edge_collapse(
			op_cache->v0().second, op_cache->v1().second, op_cache->collapse_to);
	}

	// -------------------------------------------------------------------------

	template <>
	void WildTriRemesher::map_edge_collapse_edge_attributes(const Tuple &t) { return; }

	template <>
	void WildTetRemesher::map_edge_collapse_edge_attributes(const Tuple &t)
	{
		const auto &[old_v0_id, old_v0] = op_cache->v0();
		const auto &[old_v1_id, old_v1] = op_cache->v1();
		const auto &old_edges = op_cache->edges();

		const size_t new_vid = t.vid(*this);

		for (const Tuple &t : get_one_ring_tets_for_vertex(t))
		{
			for (const Tuple &e : tet_edges(t))
			{
				std::array<size_t, 2> vids{{e.vid(*this), e.switch_vertex(*this).vid(*this)}};
				auto iter = std::find(vids.begin(), vids.end(), new_vid);

				if (iter != vids.end())
				{
					*iter = old_v0_id;
					const auto find_old_edge0 = old_edges.find(vids);
					*iter = old_v1_id;
					const auto find_old_edge1 = old_edges.find(vids);

					if (find_old_edge0 != old_edges.end() && find_old_edge1 != old_edges.end())
					{
						edge_attr(e.eid(*this)).energy_rank = std::min(
							find_old_edge0->second.energy_rank, find_old_edge1->second.energy_rank);
						edge_attr(e.eid(*this)).op_depth =
							std::max(find_old_edge0->second.op_depth, find_old_edge1->second.op_depth);
					}
					else if (find_old_edge0 != old_edges.end())
						edge_attr(e.eid(*this)) = find_old_edge0->second;
					else if (find_old_edge1 != old_edges.end())
						edge_attr(e.eid(*this)) = find_old_edge1->second;
					else
						assert(false);

					edge_attr(e.eid(*this)).op_attempts = 0;
					edge_attr(e.eid(*this)).op_depth++;
				}
				else
				{
					edge_attr(e.eid(*this)) = old_edges.at(vids);
				}
			}
		}
	}

	// -------------------------------------------------------------------------

	template <>
	void WildTriRemesher::map_edge_collapse_boundary_attributes(const Tuple &t)
	{
		const auto &[old_v0_id, old_v0] = op_cache->v0();
		const auto &[old_v1_id, old_v1] = op_cache->v1();
		const auto &old_edges = op_cache->edges();

		const size_t new_vid = t.vid(*this);

		const std::vector<Tuple> one_ring_edges = get_one_ring_edges_for_vertex(t);
		for (const Tuple &e : one_ring_edges)
		{
			const size_t e_id = e.eid(*this);

			size_t v0_id = e.vid(*this);
			size_t v1_id = e.switch_vertex(*this).vid(*this);
			if (v0_id > v1_id)
				std::swap(v0_id, v1_id);
			assert(v1_id == new_vid); // should be the new vertex because it has a larger id

			const std::array<size_t, 2> old_edge0{{v0_id, old_v0_id}};
			const std::array<size_t, 2> old_edge1{{v0_id, old_v1_id}};
			const auto find_old_edge0 = old_edges.find(old_edge0);
			const auto find_old_edge1 = old_edges.find(old_edge1);

			if (find_old_edge0 != old_edges.end() && find_old_edge1 != old_edges.end())
			{
				// both cannot be boundary edges
				assert(!(find_old_edge0->second.boundary_id >= 0 && find_old_edge1->second.boundary_id >= 0));
				boundary_attrs[e_id].boundary_id = std::max(
					find_old_edge0->second.boundary_id, find_old_edge1->second.boundary_id);
				boundary_attrs[e_id].energy_rank = std::min(
					find_old_edge0->second.energy_rank, find_old_edge1->second.energy_rank);
				boundary_attrs[e_id].op_depth =
					std::max(find_old_edge0->second.op_depth, find_old_edge1->second.op_depth);
			}
			else if (find_old_edge0 != old_edges.end())
				boundary_attrs[e_id] = find_old_edge0->second;
			else if (find_old_edge1 != old_edges.end())
				boundary_attrs[e_id] = find_old_edge1->second;
			else
				assert(false);

			boundary_attrs[e_id].op_attempts = 0;
			boundary_attrs[e_id].op_depth++;

#ifndef NDEBUG
			if (is_boundary_facet(e))
				assert(boundary_attrs[facet_id(e)].boundary_id >= 0);
			else
				assert(boundary_attrs[facet_id(e)].boundary_id == -1);
#endif
		}
	}

	template <>
	void WildTetRemesher::map_edge_collapse_boundary_attributes(const Tuple &t)
	{
		const auto &[old_v0_id, old_v0] = op_cache->v0();
		const auto &[old_v1_id, old_v1] = op_cache->v1();
		const auto &old_faces = op_cache->faces();

		const size_t new_vid = t.vid(*this);

		for (const Tuple &f : get_one_ring_boundary_faces_for_vertex(t))
		{
			assert(f.is_boundary_face(*this));
			const size_t f_id = f.fid(*this);

			const std::array<Tuple, 3> fv = get_face_vertices(f);
			std::array<size_t, 3> vids{{fv[0].vid(*this), fv[1].vid(*this), fv[2].vid(*this)}};
			auto iter = std::find(vids.begin(), vids.end(), new_vid);
			assert(iter != vids.end());

			*iter = old_v0_id;
			const auto find_old_face0 = old_faces.find(vids);
			*iter = old_v1_id;
			const auto find_old_face1 = old_faces.find(vids);

			if (find_old_face0 != old_faces.end() && find_old_face1 != old_faces.end())
			{
				// both cannot be boundary faces
				assert(!(find_old_face0->second.boundary_id >= 0 && find_old_face1->second.boundary_id >= 0));
				boundary_attrs[f_id].boundary_id = std::max(
					find_old_face0->second.boundary_id, find_old_face1->second.boundary_id);
			}
			else if (find_old_face0 != old_faces.end())
				boundary_attrs[f_id] = find_old_face0->second;
			else if (find_old_face1 != old_faces.end())
				boundary_attrs[f_id] = find_old_face1->second;
			else
				assert(false);

#ifndef NDEBUG
			if (is_boundary_facet(f))
				assert(boundary_attrs[facet_id(f)].boundary_id >= 0);
			else
				assert(boundary_attrs[facet_id(f)].boundary_id == -1);
#endif
		}

		map_edge_collapse_edge_attributes(t);

#ifndef NDEBUG
		for (const Tuple &f : get_facets())
		{
			if (is_boundary_facet(f))
				assert(boundary_attrs[facet_id(f)].boundary_id >= 0);
			else
				assert(boundary_attrs[facet_id(f)].boundary_id == -1);
		}
#endif
	}

	// =========================================================================

	// ------------------------------------------------------------------------
	// Template specializations

	template class WildRemesher<wmtk::TriMesh>;
	template class WildRemesher<wmtk::TetMesh>;
	template class PhysicsRemesher<wmtk::TriMesh>;
	template class PhysicsRemesher<wmtk::TetMesh>;

} // namespace polyfem::mesh