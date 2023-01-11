#include <polyfem/mesh/remesh/WildTriRemesher.hpp>
#include <polyfem/mesh/remesh/WildTetRemesher.hpp>

namespace polyfem::mesh
{
	namespace
	{
		template <typename Container>
		inline bool contains(const Container &container, const typename Container::value_type &val)
		{
			return std::find(container.begin(), container.end(), val) != container.end();
		}

		template <typename T>
		inline T lerp(const T &a, const T &b, double t)
		{
			return a + t * (b - a);
		}
	} // namespace

	template <class WMTKMesh>
	bool WildRemesher<WMTKMesh>::split_edge_before(const Tuple &e)
	{
#ifndef NDEBUG
		write_priority_queue_mesh(
			state.resolve_output_path(
				fmt::format("split_edge_before_{:04d}.vtu", vis_counter++)),
			e);
#endif

		if (edge_attr(e.eid(*this)).split_attempts++ >= max_split_attempts)
			return false;

		if (edge_attr(e.eid(*this)).split_depth >= max_split_depth)
			return false;

		// ---------------------------------------------------------------------

		// Dont split if the edge is too small
		double min_edge_length = this->min_edge_length;
		if (is_on_boundary(e) && state.is_contact_enabled())
			min_edge_length = std::max(min_edge_length, 2.01 * state.args["contact"]["dhat"].get<double>());
		if (edge_length(e) < min_edge_length)
			return false;

		// ---------------------------------------------------------------------

		const auto &v0 = vertex_attrs[e.vid(*this)].rest_position;
		const auto &v1 = vertex_attrs[e.switch_vertex(*this).vid(*this)].rest_position;
		const double local_energy = local_mesh_energy((v0 + v1) / 2);
		// Do not split if the energy of the local mesh is too small
		if (local_energy < split_absolute_tolerance)
			return false;

		// ---------------------------------------------------------------------

		// Cache necessary local data
		cache_split_edge(e, local_energy);

		return true;
	}

	// -------------------------------------------------------------------------
	// 2D

	bool WildTriRemesher::split_edge_after(const Tuple &t)
	{
		// 0) perform operation (done before this function)

		// 1a) Update rest position of new vertex
		const Tuple new_vertex = t.switch_vertex(*this);
		VertexAttributes &new_vertex_attr = vertex_attrs[new_vertex.vid(*this)];
		const auto &[old_v0_id, v0] = op_cache.v0();
		const auto &[old_v1_id, v1] = op_cache.v1();
		// TODO: maybe we want to use a different barycentric coordinate?
		const double alpha = 0.5;
		new_vertex_attr.rest_position = lerp(v0.rest_position, v1.rest_position, alpha);
		new_vertex_attr.fixed = v0.fixed && v1.fixed;
		new_vertex_attr.partition_id = v0.partition_id; // TODO: what should this be?

		// 1b) Assign edge attributes to the new edges
		map_edge_split_boundary_attributes(new_vertex, op_cache.edges(), old_v0_id, old_v1_id);
		map_edge_split_element_attributes(t, op_cache.faces());

		// 2) Project quantities so to minimize the L2 error
		new_vertex_attr.position = lerp(v0.position, v1.position, alpha);
		new_vertex_attr.projection_quantities =
			lerp(v0.projection_quantities, v1.projection_quantities, alpha);
		if (!v0.fixed || !v1.fixed)
		{
			// NOTE: this assumes friction gradient is the last column of the projection matrix,
			// so internal points have a gradient of zero.
			new_vertex_attr.projection_quantities.rightCols(1).setZero();
		}

		// 3) Perform a local relaxation of the n-ring to get an estimate of the
		//    energy decrease.
		return local_relaxation(
			new_vertex, op_cache.local_energy,
			split_relative_tolerance,
			split_absolute_tolerance);
	}

	void WildTriRemesher::map_edge_split_boundary_attributes(
		const Tuple &new_vertex,
		const EdgeMap<BoundaryAttributes> &old_edges,
		const size_t old_v0_id,
		const size_t old_v1_id)
	{
		BoundaryAttributes old_split_edge = old_edges.at({{old_v0_id, old_v1_id}});
		old_split_edge.split_attempts = 0;
		BoundaryAttributes interior_edge; // default
		interior_edge.split_depth = old_split_edge.split_depth;

		const size_t new_vid = new_vertex.vid(*this);

		const std::vector<Tuple> new_faces = get_one_ring_tris_for_vertex(new_vertex);
		for (const auto &new_face : new_faces)
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
					boundary_attrs[e.eid(*this)].split_depth++;
				}
				else
				{
					boundary_attrs[e.eid(*this)] = old_edges.at({{v0_id, v1_id}});
				}
			}
		}
	}

	void WildTriRemesher::map_edge_split_element_attributes(
		const Tuple &t,
		const std::vector<ElementAttributes> &old_elements)
	{
		WildTriRemesher::Tuple nav = t.switch_vertex(*this);
		element_attrs[nav.fid(*this)] = old_elements[0];
		nav = nav.switch_edge(*this).switch_face(*this).value();
		element_attrs[nav.fid(*this)] = old_elements[0];
		nav = nav.switch_edge(*this);
		if (nav.switch_face(*this))
		{
			nav = nav.switch_face(*this).value();
			element_attrs[nav.fid(*this)] = old_elements[1];
			nav = nav.switch_edge(*this).switch_face(*this).value();
			element_attrs[nav.fid(*this)] = old_elements[1];
		}
	}

	// -------------------------------------------------------------------------
	// 3D

	bool WildTetRemesher::split_edge_after(const Tuple &t)
	{
		// 0) perform operation (done before this function)

		// 1a) Update rest position of new vertex
		VertexAttributes &new_vertex_attr = vertex_attrs[t.vid(*this)];
		const auto &[old_v0_id, v0] = op_cache.v0();
		const auto &[old_v1_id, v1] = op_cache.v1();
		// TODO: maybe we want to use a different barycentric coordinate?
		const double alpha = 0.5;
		new_vertex_attr.rest_position = lerp(v0.rest_position, v1.rest_position, alpha);
		new_vertex_attr.fixed = v0.fixed && v1.fixed;
		new_vertex_attr.partition_id = v0.partition_id; // TODO: what should this be?

		// 1b) Assign edge attributes to the new edges
		map_edge_split_edge_attributes(t, op_cache.edges(), old_v0_id, old_v1_id);
		map_edge_split_boundary_attributes(t, op_cache.faces(), old_v0_id, old_v1_id);
		map_edge_split_element_attributes(t, op_cache.tets(), old_v0_id, old_v1_id);

		// 2) Project quantities so to minimize the L2 error
		new_vertex_attr.position = lerp(v0.position, v1.position, alpha);
		new_vertex_attr.projection_quantities =
			lerp(v0.projection_quantities, v1.projection_quantities, alpha);
		if (!v0.fixed || !v1.fixed)
		{
			// NOTE: this assumes friction gradient is the last column of the projection matrix,
			// so internal points have a gradient of zero.
			new_vertex_attr.projection_quantities.rightCols(1).setZero();
		}

		// 3) Perform a local relaxation of the n-ring to get an estimate of the
		//    energy decrease.
		return local_relaxation(
			t, op_cache.local_energy,
			split_relative_tolerance,
			split_absolute_tolerance);
	}

	void WildTetRemesher::map_edge_split_edge_attributes(
		const Tuple &new_vertex,
		const EdgeMap<EdgeAttributes> &old_edges,
		const size_t old_v0_id,
		const size_t old_v1_id)
	{
		EdgeAttributes old_split_edge = old_edges.at({{old_v0_id, old_v1_id}});
		old_split_edge.split_attempts = 0;
		EdgeAttributes interior_edge; // default
		interior_edge.split_depth = old_split_edge.split_depth;

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
					edge_attrs[e.eid(*this)].split_depth++;
				}
				else
				{
					edge_attrs[e.eid(*this)] = old_edges.at({{v0_id, v1_id}});
				}
			}
		}
	}

	void WildTetRemesher::map_edge_split_boundary_attributes(
		const Tuple &new_vertex,
		const FaceMap<BoundaryAttributes> &old_faces,
		const size_t old_v0_id,
		const size_t old_v1_id)
	{
		const size_t new_vid = new_vertex.vid(*this);
		const std::vector<Tuple> new_tets = get_one_ring_tets_for_vertex(new_vertex);

		for (const auto &t : new_tets)
		{
			for (int i = 0; i < 4; i++)
			{
				const Tuple f = tuple_from_face(t.tid(*this), i);

				std::array<Tuple, 3> vs = get_face_vertices(f);
				std::array<size_t, 3> vids =
					{{vs[0].vid(*this), vs[1].vid(*this), vs[2].vid(*this)}};

				auto new_v_itr = std::find(vids.begin(), vids.end(), new_vid);

				if (new_v_itr != vids.end())
				{
					const bool contains_old_v0 = contains(vids, old_v0_id);
					const bool contains_old_v1 = contains(vids, old_v1_id);
					assert(!(contains_old_v0 && contains_old_v1));

					// New inerior face, use default boundary attributes
					if (!contains_old_v0 && !contains_old_v1)
						continue;

					*new_v_itr = contains_old_v0 ? old_v1_id : old_v0_id;
				}
				// else: new vertex is not part of this face, so retain the old boundary attributes

				boundary_attrs[f.fid(*this)] = old_faces.at(vids);
			}
		}
	}

	void WildTetRemesher::map_edge_split_element_attributes(
		const Tuple &new_vertex,
		const TetMap<ElementAttributes> &old_elements,
		const size_t old_v0_id,
		const size_t old_v1_id)
	{
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

			element_attrs[t.tid(*this)] = old_elements.at(vids);
		}
	}

	template <class WMTKMesh>
	void WildRemesher<WMTKMesh>::split_edges()
	{
		using Operations = std::vector<std::pair<std::string, Tuple>>;

		std::vector<Tuple> included_edges;
		{
			const std::vector<Tuple> starting_elements = get_elements();
			for (const Tuple &t : starting_elements)
			{
				const size_t t_id = element_id(t);

				if (element_attrs[t_id].energy_rank != ElementAttributes::EnergyRank::TOP)
					continue;

				for (int ei = 0; ei < EDGES_IN_ELEMENT; ++ei)
				{
					included_edges.push_back(WMTKMesh::tuple_from_edge(t_id, ei));
				}
			}
			wmtk::unique_edge_tuples(*this, included_edges);
		}

		if (included_edges.empty())
			return;

		Operations splits;
		splits.reserve(included_edges.size());
		for (const Tuple &e : included_edges)
			splits.emplace_back("edge_split", e);

		executor.priority = [](const WildRemesher &m, std::string op, const Tuple &t) -> double {
			// NOTE: this code compute the edge length
			// return m.edge_length(t);
			return m.edge_elastic_energy(t);
		};

		executor(*this, splits);
	}

	// ------------------------------------------------------------------------
	// Template specializations
	template class WildRemesher<wmtk::TriMesh>;
	template class WildRemesher<wmtk::TetMesh>;

} // namespace polyfem::mesh