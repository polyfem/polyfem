#include <polyfem/mesh/remesh/PhysicsRemesher.hpp>

namespace polyfem::mesh
{
	template <class WMTKMesh>
	void WildRemesher<WMTKMesh>::cache_split_edge(const Tuple &e)
	{
		op_cache = decltype(op_cache)::element_type::split_edge(*this, e);
	}

	template <class WMTKMesh>
	bool WildRemesher<WMTKMesh>::split_edge_before(const Tuple &e)
	{
		// POLYFEM_REMESHER_SCOPED_TIMER("Split edges before");

		if (!WMTKMesh::split_edge_before(e))
			return false;

		// Dont split if the edge is too small
		double min_edge_length = args["split"]["min_edge_length"];
		if (is_boundary_facet(e) && state.is_contact_enabled())
			min_edge_length = std::max(min_edge_length, 2.0 * state.args["contact"]["dhat"].template get<double>());

		if (rest_edge_length(e) < min_edge_length)
		{
			executor.m_cnt_fail--; // do not count this as a failed split
			return false;
		}

		// Cache necessary local data
		cache_split_edge(e);

		return true;
	}

	template <class WMTKMesh>
	bool PhysicsRemesher<WMTKMesh>::split_edge_before(const Tuple &e)
	{
		// POLYFEM_REMESHER_SCOPED_TIMER("Split edges before");

		if (!Super::split_edge_before(e)) // NOTE: also calls cache_split_edge
			return false;

		if (this->edge_attr(e.eid(*this)).op_attempts++ >= this->max_op_attempts
			|| this->edge_attr(e.eid(*this)).op_depth >= args["split"]["max_depth"].template get<int>())
		{
			this->executor.m_cnt_fail--; // do not count this as a failed split
			return false;
		}

		const auto &v0 = this->vertex_attrs[e.vid(*this)].rest_position;
		const auto &v1 = this->vertex_attrs[e.switch_vertex(*this).vid(*this)].rest_position;
		this->op_cache->local_energy = local_mesh_energy((v0 + v1) / 2);
		// assert(this->op_cache->local_energy >= 0);
		// Do not split if the energy of the local mesh is too small
		// if (this->op_cache->local_energy < args["split"]["acceptance_tolerance"].template get<double>())
		// 	return false;

		return true;
	}

	// -------------------------------------------------------------------------

	namespace
	{
		template <typename T>
		inline T lerp(const T &a, const T &b, double t)
		{
			return a + t * (b - a);
		}
	} // namespace

	template <class WMTKMesh>
	bool WildRemesher<WMTKMesh>::split_edge_after(const Tuple &t)
	{
		if (!WMTKMesh::split_edge_after(t))
			return false;

		// 0) perform operation (done before this function)

		// 1a) Update rest position of new vertex
		Tuple new_vertex;
		if constexpr (std::is_same_v<WMTKMesh, wmtk::TriMesh>)
			new_vertex = t.switch_vertex(*this);
		else
			new_vertex = t;

		const auto &[old_v0_id, v0] = op_cache->v0();
		const auto &[old_v1_id, v1] = op_cache->v1();

		VertexAttributes &new_vertex_attr = vertex_attrs[new_vertex.vid(*this)];
		constexpr double alpha = 0.5; // TODO: maybe we want to use a different barycentric coordinate?
		new_vertex_attr.rest_position = lerp(v0.rest_position, v1.rest_position, alpha);
		new_vertex_attr.fixed = v0.fixed && v1.fixed;
		new_vertex_attr.partition_id = v0.partition_id; // TODO: what should this be?

		if (state.is_contact_enabled() && is_boundary_vertex(new_vertex))
		{
			const double dhat = state.args["contact"]["dhat"].template get<double>();

			// only enforce this invariant if it started valid
			if (state.min_boundary_edge_length >= dhat)
			{
				for (const Tuple &e : get_one_ring_boundary_edges_for_vertex(new_vertex))
				{
					if (rest_edge_length(e) < dhat)
						return false; // produced too small edge
				}
			}
		}

		// 1b) Assign edge attributes to the new edges
		map_edge_split_edge_attributes(new_vertex);
		map_edge_split_boundary_attributes(new_vertex);
		map_edge_split_element_attributes(t);

		// 2) Project quantities so to minimize the L2 error
		new_vertex_attr.position = lerp(v0.position, v1.position, alpha);
		new_vertex_attr.projection_quantities =
			lerp(v0.projection_quantities, v1.projection_quantities, alpha);

		// 3) No need to check for intersections because we are not moving the vertices
		return true;
	}

	template <class WMTKMesh>
	bool PhysicsRemesher<WMTKMesh>::split_edge_after(const Tuple &t)
	{
		utils::Timer timer(this->timings["Split edges after"]);
		timer.start();
		if (!Super::split_edge_after(t))
			return false;
		// local relaxation has its own timers
		timer.stop();

		Tuple new_vertex;
		if constexpr (std::is_same_v<WMTKMesh, wmtk::TriMesh>)
			new_vertex = t.switch_vertex(*this);
		else
			new_vertex = t;

		std::vector<Tuple> local_mesh_tuples = this->local_mesh_tuples(new_vertex);

		// Perform a local relaxation of the n-ring to get an estimate of the energy decrease.
		if (!local_relaxation(local_mesh_tuples, args["split"]["acceptance_tolerance"]))
			return false;

		// Increase the hash of the triangles that have been modified
		// to invalidate all tuples that point to them.
		this->extend_local_patch(local_mesh_tuples);
		for (Tuple &t : local_mesh_tuples)
		{
			assert(t.is_valid(*this));
			if constexpr (std::is_same_v<wmtk::TriMesh, WMTKMesh>)
				this->m_tri_connectivity[t.fid(*this)].hash++;
			else
				this->m_tet_connectivity[t.tid(*this)].hash++;
			assert(!t.is_valid(*this));
			t.update_hash(*this);
			assert(t.is_valid(*this));
		}

		return true;
	}

	// -------------------------------------------------------------------------

	template <class WMTKMesh>
	void PhysicsRemesher<WMTKMesh>::split_edges()
	{
		std::vector<Tuple> included_edges;
		{
			const std::vector<Tuple> edges = WMTKMesh::get_edges();
			std::copy_if(edges.begin(), edges.end(), std::back_inserter(included_edges), [this](const Tuple &e) {
				return this->edge_attr(e.eid(*this)).energy_rank == Super::EdgeAttributes::EnergyRank::TOP;
			});
		}

		if (included_edges.empty())
			return;

		Operations splits;
		splits.reserve(included_edges.size());
		for (const Tuple &e : included_edges)
			splits.emplace_back("edge_split", e);

		executor.priority = [&](const WildRemesher<WMTKMesh> &, std::string op, const Tuple &t) -> double {
			return this->edge_elastic_energy(t);
		};

		executor(*this, splits);
	}

	// =========================================================================
	// Map attributes
	// =========================================================================

	template <>
	void WildTriRemesher::map_edge_split_edge_attributes(const Tuple &new_vertex) { return; }

	template <>
	void WildTetRemesher::map_edge_split_edge_attributes(const Tuple &new_vertex)
	{
		const auto &[old_v0_id, old_v0] = op_cache->v0();
		const auto &[old_v1_id, old_v1] = op_cache->v1();
		const auto &old_edges = op_cache->edges();

		EdgeAttributes old_split_edge = old_edges.at({{old_v0_id, old_v1_id}});
		old_split_edge.op_attempts = 0;
		EdgeAttributes interior_edge; // default
		interior_edge.op_depth = old_split_edge.op_depth;
		interior_edge.energy_rank = old_split_edge.energy_rank;

		const size_t new_vid = new_vertex.vid(*this);

		const std::vector<Tuple> new_tets = get_one_ring_tets_for_vertex(new_vertex);
		for (const auto &t : new_tets)
		{
			for (int i = 0; i < 6; i++)
			{
				const Tuple e = tuple_from_edge(t.tid(*this), i);

				size_t v0_id = e.vid(*this);
				size_t v1_id = e.switch_vertex(*this).vid(*this);
				if (v0_id > v1_id)
					std::swap(v0_id, v1_id);

				assert(v0_id != new_vid); // new_vid should have a higher id than any other vertex
				if (v1_id == new_vid)
				{
					edge_attrs[e.eid(*this)] =
						(v0_id == old_v0_id || v0_id == old_v1_id) ? old_split_edge : interior_edge;
					edge_attrs[e.eid(*this)].op_depth++;
				}
				else
				{
					edge_attrs[e.eid(*this)] = old_edges.at({{v0_id, v1_id}});
				}
			}
		}
	}

	// -------------------------------------------------------------------------

	namespace
	{
		template <typename Container>
		inline bool contains(const Container &container, const typename Container::value_type &val)
		{
			return std::find(container.begin(), container.end(), val) != container.end();
		}
	} // namespace

	template <>
	void WildTriRemesher::map_edge_split_boundary_attributes(const Tuple &new_vertex)
	{
		const auto &[old_v0_id, v0] = op_cache->v0();
		const auto &[old_v1_id, v1] = op_cache->v1();
		const auto &old_edges = op_cache->edges();

		BoundaryAttributes old_split_edge = old_edges.at({{old_v0_id, old_v1_id}});
		old_split_edge.op_attempts = 0;
		BoundaryAttributes interior_edge; // default
		interior_edge.op_depth = old_split_edge.op_depth;
		interior_edge.energy_rank = old_split_edge.energy_rank;

		const size_t new_vid = new_vertex.vid(*this);

		for (const auto &new_face : get_one_ring_tris_for_vertex(new_vertex))
		{
			for (int i = 0; i < 3; i++)
			{
				const Tuple e = tuple_from_edge(new_face.fid(*this), i);

				size_t v0_id = e.vid(*this);
				size_t v1_id = e.switch_vertex(*this).vid(*this);
				if (v0_id > v1_id)
					std::swap(v0_id, v1_id);

				assert(v0_id != new_vid); // new_vid should have a higher id than any other vertex
				if (v1_id == new_vid)
				{
					boundary_attrs[e.eid(*this)] =
						(v0_id == old_v0_id || v0_id == old_v1_id) ? old_split_edge : interior_edge;
					boundary_attrs[e.eid(*this)].op_depth++;
				}
				else
				{
					boundary_attrs[e.eid(*this)] = old_edges.at({{v0_id, v1_id}});
				}

#ifndef NDEBUG
				if (is_boundary_facet(e))
					assert(boundary_attrs[facet_id(e)].boundary_id >= 0);
				else
					assert(boundary_attrs[facet_id(e)].boundary_id == -1);
#endif
			}
		}
	}

	template <>
	void WildTetRemesher::map_edge_split_boundary_attributes(const Tuple &new_vertex)
	{
		const auto &[old_v0_id, old_v0] = op_cache->v0();
		const auto &[old_v1_id, old_v1] = op_cache->v1();
		const auto &old_faces = op_cache->faces();

		const size_t new_vid = new_vertex.vid(*this);
		for (const auto &t : get_one_ring_tets_for_vertex(new_vertex))
		{
			for (int i = 0; i < 4; i++)
			{
				const Tuple f = tuple_from_face(t.tid(*this), i);

				std::array<size_t, 3> vids = facet_vids(f);

				auto new_v_itr = std::find(vids.begin(), vids.end(), new_vid);

				if (new_v_itr != vids.end())
				{
					const bool contains_old_v0 = contains(vids, old_v0_id);
					const bool contains_old_v1 = contains(vids, old_v1_id);
					assert(!(contains_old_v0 && contains_old_v1));

					// New interior face, use default boundary attributes
					if (!contains_old_v0 && !contains_old_v1)
					{
						assert(!is_boundary_facet(f));
						boundary_attrs[facet_id(f)].boundary_id = -1;
						continue;
					}

					*new_v_itr = contains_old_v0 ? old_v1_id : old_v0_id;
				}
				// else: new vertex is not part of this face, so retain the old boundary attributes

				boundary_attrs[f.fid(*this)] = old_faces.at(vids);

#ifndef NDEBUG
				if (is_boundary_facet(f))
					assert(boundary_attrs[facet_id(f)].boundary_id >= 0);
				else
					assert(boundary_attrs[facet_id(f)].boundary_id == -1);
#endif
			}
		}
	}

	// -------------------------------------------------------------------------

	template <>
	void WildTriRemesher::map_edge_split_element_attributes(const Tuple &t)
	{
		const auto &old_faces = op_cache->faces();

		Tuple nav = t.switch_vertex(*this);
		element_attrs[nav.fid(*this)] = old_faces[0];
		nav = nav.switch_edge(*this).switch_face(*this).value();
		element_attrs[nav.fid(*this)] = old_faces[0];
		nav = nav.switch_edge(*this);
		if (nav.switch_face(*this))
		{
			nav = nav.switch_face(*this).value();
			element_attrs[nav.fid(*this)] = old_faces[1];
			nav = nav.switch_edge(*this).switch_face(*this).value();
			element_attrs[nav.fid(*this)] = old_faces[1];
		}
	}

	template <>
	void WildTetRemesher::map_edge_split_element_attributes(const Tuple &new_vertex)
	{
		const auto &[old_v0_id, old_v0] = op_cache->v0();
		const auto &[old_v1_id, old_v1] = op_cache->v1();
		const auto &old_tets = op_cache->tets();

		const size_t new_vid = new_vertex.vid(*this);
		const std::vector<Tuple> new_tets = get_one_ring_tets_for_vertex(new_vertex);

		for (const auto &t : new_tets)
		{
			std::array<size_t, 4> vids = oriented_tet_vids(t);
			auto new_v_itr = std::find(vids.begin(), vids.end(), new_vid);
			assert(new_v_itr != vids.end());

			const bool contains_old_v0 = contains(vids, old_v0_id);
			assert(contains_old_v0 ^ contains(vids, old_v1_id));
			*new_v_itr = contains_old_v0 ? old_v1_id : old_v0_id;

			element_attrs[t.tid(*this)] = old_tets.at(vids);
		}
	}

	// -------------------------------------------------------------------------
	// Template specializations

	template class WildRemesher<wmtk::TriMesh>;
	template class WildRemesher<wmtk::TetMesh>;
	template class PhysicsRemesher<wmtk::TriMesh>;
	template class PhysicsRemesher<wmtk::TetMesh>;

} // namespace polyfem::mesh