#include "LocalMesh.hpp"

#include <polyfem/mesh/remesh/PhysicsRemesher.hpp>
#include <polyfem/utils/Timer.hpp>
#include <polyfem/utils/GeometryUtils.hpp>
#include <polyfem/utils/MatrixUtils.hpp>

#include <paraviewo/VTUWriter.hpp>

#include <wmtk/utils/TupleUtils.hpp>

#include <igl/boundary_facets.h>
#include <igl/edges.h>
#include <igl/PI.h>

#include <BVH.hpp>

namespace polyfem::mesh
{
	namespace
	{
		// Assert there are no duplicates in elements
		template <typename M>
		bool has_duplicate_elements(const M &m, const std::vector<typename M::Tuple> &elements)
		{
			std::vector<size_t> ids;
			for (const auto &t : elements)
				ids.push_back(m.element_id(t));
			std::sort(ids.begin(), ids.end());
			ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
			return ids.size() != elements.size();
		}
	} // namespace

	using TriMesh = WildRemesher<wmtk::TriMesh>;
	using TetMesh = WildRemesher<wmtk::TetMesh>;

	void unique_facet_tuples(const wmtk::TriMesh &m, std::vector<wmtk::TriMesh::Tuple> &tuples)
	{
		wmtk::unique_edge_tuples(m, tuples);
	}

	void unique_facet_tuples(const wmtk::TetMesh &m, std::vector<wmtk::TetMesh::Tuple> &tuples)
	{
		wmtk::unique_face_tuples(m, tuples);
	}

	template <typename M>
	LocalMesh<M>::LocalMesh(
		const M &m,
		const std::vector<Tuple> &element_tuples,
		const bool include_global_boundary)
	{
		POLYFEM_SCOPED_TIMER("LocalMesh::LocalMesh");

		std::unordered_set<size_t> global_element_ids;
		{
			POLYFEM_REMESHER_SCOPED_TIMER("LocalMesh::LocalMesh -> init m_elements");
			m_elements.resize(element_tuples.size(), m.dim() + 1);
			m_body_ids.reserve(element_tuples.size());
			for (int fi = 0; fi < num_elements(); fi++)
			{
				const Tuple &elem = element_tuples[fi];
				global_element_ids.insert(m.element_id(elem));

				const auto vids = m.element_vids(elem);

				for (int i = 0; i < vids.size(); ++i)
				{
					if (m_global_to_local.find(vids[i]) == m_global_to_local.end())
						m_global_to_local[vids[i]] = m_global_to_local.size();
					m_elements(fi, i) = m_global_to_local[vids[i]];
				}

				m_body_ids.push_back(m.element_attrs[m.element_id(elem)].body_id);
			}
		}
		// The above puts local vertices at front
		m_num_local_vertices = m_global_to_local.size();

		// ---------------------------------------------------------------------

		std::vector<Tuple> local_boundary_facets;
		{
			POLYFEM_REMESHER_SCOPED_TIMER("LocalMesh::LocalMesh -> init m_fixed_vertices");
			for (int fi = 0; fi < num_elements(); fi++)
			{
				const Tuple &elem = element_tuples[fi];

				for (int i = 0; i < M::FACETS_PER_ELEMENT; ++i)
				{
					const Tuple facet = m.tuple_from_facet(m.element_id(elem), i);

					// Only fix internal facets
					if (m.is_boundary_facet(facet))
					{
						local_boundary_facets.push_back(facet);
						continue;
					}

					size_t adjacent_tid;
					if constexpr (std::is_same_v<M, TriMesh>)
						adjacent_tid = facet.switch_face(m)->fid(m);
					else
						adjacent_tid = facet.switch_tetrahedron(m)->tid(m);

					// Only fix internal facets that do not have a neighbor in the local mesh
					if (global_element_ids.find(adjacent_tid) != global_element_ids.end())
						continue;

					for (const size_t &vid : m.facet_vids(facet))
						m_fixed_vertices.push_back(m_global_to_local[vid]);
				}
			}
		}

		// ---------------------------------------------------------------------
		{
			// Only build boundary ids for the local facets
			std::vector<Tuple> facets_tuples;
			facets_tuples.reserve(num_elements() * M::FACETS_PER_ELEMENT);
			std::vector<Tuple> local_boundary_facets;
			for (const Tuple &elem : element_tuples)
				for (int i = 0; i < M::FACETS_PER_ELEMENT; ++i)
					facets_tuples.push_back(m.tuple_from_facet(m.element_id(elem), i));
			unique_facet_tuples(m, facets_tuples);
			if constexpr (std::is_same_v<M, TriMesh>)
				m_boundary_ids = Remesher::EdgeMap<int>();
			else
				m_boundary_ids = Remesher::FaceMap<int>();
			for (const Tuple &t : facets_tuples)
			{
				auto vids = m.facet_vids(t);
				for (auto &v : vids)
					v = m_global_to_local[v];

				const int boundary_id = m.boundary_attrs[m.facet_id(t)].boundary_id;

				if constexpr (std::is_same_v<M, TriMesh>)
					std::get<Remesher::EdgeMap<int>>(m_boundary_ids)[vids] = boundary_id;
				else
					std::get<Remesher::FaceMap<int>>(m_boundary_ids)[vids] = boundary_id;
			}
		}
		// ---------------------------------------------------------------------

		if (include_global_boundary)
		{
			POLYFEM_REMESHER_SCOPED_TIMER("LocalMesh::LocalMesh -> include_global_boundary");
			// Copy the global to local map so we can check if a vertex is new
			const std::unordered_map<int, int> prev_global_to_local = m_global_to_local;

			const std::vector<Tuple> global_boundary_facets = m.boundary_facets();

			boundary_facets().resize(global_boundary_facets.size(), m_elements.cols() - 1);
			for (int i = 0; i < global_boundary_facets.size(); i++)
			{
				const Tuple &facet = global_boundary_facets[i];
				const auto vids = m.facet_vids(facet);

				const bool is_new_facet = std::any_of(vids.begin(), vids.end(), [&](size_t vid) {
					return prev_global_to_local.find(vid) == prev_global_to_local.end();
				});

				for (int j = 0; j < vids.size(); ++j)
				{
					if (m_global_to_local.find(vids[j]) == m_global_to_local.end())
						m_global_to_local[vids[j]] = m_global_to_local.size();

					boundary_facets()(i, j) = m_global_to_local[vids[j]];

					if (is_new_facet)
						m_fixed_vertices.push_back(boundary_facets()(i, j));
				}
			}
		}
		else
		{
			POLYFEM_REMESHER_SCOPED_TIMER("LocalMesh::LocalMesh -> !include_global_boundary");

			unique_facet_tuples(m, local_boundary_facets);
			boundary_facets().resize(local_boundary_facets.size(), m_elements.cols() - 1);
			for (int i = 0; i < local_boundary_facets.size(); i++)
			{
				const auto vids = m.facet_vids(local_boundary_facets[i]);
				for (int j = 0; j < vids.size(); ++j)
					boundary_facets()(i, j) = m_global_to_local[vids[j]];
			}
		}

		if (m_boundary_faces.rows() > 0)
			igl::edges(m_boundary_faces, m_boundary_edges);

		remove_duplicate_fixed_vertices();

		// ---------------------------------------------------------------------

		init_vertex_attributes(m);

		init_local_to_global();

		const int tmp_num_vertices = num_vertices();

		// Include the obstacle as part of the local mesh if including the global boundary
		if (include_global_boundary && m.obstacle().n_vertices() > 0)
		{
			POLYFEM_REMESHER_SCOPED_TIMER("LocalMesh::LocalMesh -> append obstacles");
			const Obstacle &obstacle = m.obstacle();
			utils::append_rows(m_rest_positions, obstacle.v());
			utils::append_rows(m_positions, obstacle.v() + m.obstacle_displacements());
			utils::append_rows(m_projection_quantities, m.obstacle_quantities());
			if (obstacle.n_edges() > 0)
				utils::append_rows(m_boundary_edges, obstacle.e().array() + tmp_num_vertices);
			if (obstacle.n_faces() > 0)
				utils::append_rows(m_boundary_faces, obstacle.f().array() + tmp_num_vertices);

			for (int i = 0; i < obstacle.n_vertices(); i++)
				m_fixed_vertices.push_back(i + tmp_num_vertices);
		}
	}

	template <typename M>
	std::vector<typename M::Tuple> LocalMesh<M>::n_ring(
		const M &m, const Tuple &center, const int n)
	{
		return n_ring(m, m.get_one_ring_elements_for_vertex(center), n);
	}

	template <typename M>
	std::vector<typename M::Tuple> LocalMesh<M>::n_ring(
		const M &m, const std::vector<Tuple> &one_ring, const int n)
	{
		POLYFEM_REMESHER_SCOPED_TIMER("LocalMesh::n_ring");
		assert(!has_duplicate_elements(m, one_ring));

		std::vector<Tuple> elements = one_ring;
		std::unordered_set<size_t> visited_vertices;
		std::unordered_set<size_t> visited_elements;
		for (const auto &element : elements)
		{
			visited_elements.insert(m.element_id(element));
		}

		std::vector<Tuple> new_elements = elements;

		for (int i = 1; i < n; i++)
		{
			std::vector<Tuple> new_new_elements;
			for (const auto &elem : new_elements)
			{
				for (const Tuple &v : m.element_vertices(elem))
				{
					if (visited_vertices.find(v.vid(m)) != visited_vertices.end())
						continue;
					visited_vertices.insert(v.vid(m));

					std::vector<Tuple> tmp = m.get_one_ring_elements_for_vertex(v);
					for (auto &t1 : tmp)
					{
						if (visited_elements.find(m.element_id(t1)) != visited_elements.end())
							continue;
						visited_elements.insert(m.element_id(t1));
						elements.push_back(t1);
						new_new_elements.push_back(t1);
					}
				}
			}
			new_elements = new_new_elements;
			if (new_elements.empty())
				break;
		}

		assert(!has_duplicate_elements(m, elements));

		return elements;
	}

	template <typename M>
	std::vector<typename M::Tuple> LocalMesh<M>::flood_fill_n_ring(
		const M &m, const Tuple &center, const double area)
	{
		POLYFEM_REMESHER_SCOPED_TIMER("LocalMesh::flood_fill_n_ring");

		double current_area = 0;

		std::vector<Tuple> elements = m.get_one_ring_elements_for_vertex(center);
		std::unordered_set<size_t> visited_vertices{{center.vid(m)}};
		std::unordered_set<size_t> visited_faces;
		for (const auto &element : elements)
			visited_faces.insert(m.element_id(element));

		std::vector<Tuple> new_elements = elements;

		int n_ring = 0;
		while (current_area < area)
		{
			n_ring++;
			std::vector<Tuple> new_new_elements;
			for (const auto &elem : new_elements)
			{
				current_area += m.element_volume(elem);
				const auto vs = m.element_vertices(elem);
				for (int vi = 0; vi < 3; vi++)
				{
					const Tuple &v = vs[vi];
					if (visited_vertices.find(v.vid(m)) != visited_vertices.end())
						continue;
					visited_vertices.insert(v.vid(m));

					std::vector<Tuple> tmp = m.get_one_ring_elements_for_vertex(v);
					for (auto &t1 : tmp)
					{
						if (visited_faces.find(m.element_id(t1)) != visited_faces.end())
							continue;
						visited_faces.insert(m.element_id(t1));
						elements.push_back(t1);
						new_new_elements.push_back(t1);
					}
				}
			}
			new_elements = new_new_elements;
			if (new_elements.empty())
				break;
		}
		// logger().critical("target_area={:g} area={:g} n_ring={}", area, current_area, n_ring);

		return elements;
	}

	template <typename M>
	std::vector<typename M::Tuple> LocalMesh<M>::ball_selection(
		const M &m, const VectorNd &center, const double volume, const int n_ring_size)
	{
		POLYFEM_REMESHER_SCOPED_TIMER("LocalMesh::ball_selection");

		const int dim = m.dim();
		const double radius = dim == 2 ? std::sqrt(volume / igl::PI) : std::cbrt(0.75 * volume / igl::PI);
		constexpr double eps_radius = 1e-10;
		const VectorNd sphere_min = center.array() - radius;
		const VectorNd sphere_max = center.array() + radius;

		const std::vector<Tuple> elements = m.get_elements();

		// ---------------------------------------------------------------------

		// Loop over all elements and find those that intersect with the ball
		std::vector<Tuple> intersecting_elements;
		std::unordered_set<size_t> intersecting_fid;
		double intersecting_volume = 0;
		std::vector<Tuple> one_ring;
		for (const Tuple &element : elements)
		{
			VectorNd el_min, el_max;
			m.element_aabb(element, el_min, el_max);

			// Quick AABB check to see if the element intersects the sphere
			if (!utils::are_aabbs_intersecting(sphere_min, sphere_max, el_min, el_max))
				continue;

			// Accurate check to see if the element intersects the sphere
			const auto vids = m.element_vids(element);
			if constexpr (std::is_same_v<M, TriMesh>)
			{
				if (!utils::triangle_intersects_disk(
						m.vertex_attrs[vids[0]].rest_position,
						m.vertex_attrs[vids[1]].rest_position,
						m.vertex_attrs[vids[2]].rest_position,
						center, radius))
				{
					continue;
				}
			}
			else
			{
				static_assert(std::is_same_v<M, TetMesh>);
				if (!utils::tetrahedron_intersects_ball(
						m.vertex_attrs[vids[0]].rest_position,
						m.vertex_attrs[vids[1]].rest_position,
						m.vertex_attrs[vids[2]].rest_position,
						m.vertex_attrs[vids[3]].rest_position,
						center, radius))
				{
					continue;
				}
			}

			intersecting_elements.push_back(element);
			intersecting_fid.insert(m.element_id(element));
			intersecting_volume += m.element_volume(element);

			// Accurate check to see if the element intersects the center point
			if constexpr (std::is_same_v<M, TriMesh>)
			{
				if (!utils::triangle_intersects_disk(
						m.vertex_attrs[vids[0]].rest_position,
						m.vertex_attrs[vids[1]].rest_position,
						m.vertex_attrs[vids[2]].rest_position,
						center, eps_radius))
				{
					continue;
				}
			}
			else
			{
				static_assert(std::is_same_v<M, TetMesh>);
				if (!utils::tetrahedron_intersects_ball(
						m.vertex_attrs[vids[0]].rest_position,
						m.vertex_attrs[vids[1]].rest_position,
						m.vertex_attrs[vids[2]].rest_position,
						m.vertex_attrs[vids[3]].rest_position,
						center, eps_radius))
				{
					continue;
				}
			}

			one_ring.push_back(element);
		}
		assert(!intersecting_elements.empty());

		// ---------------------------------------------------------------------

		// Flood fill to fill out the desired volume
		for (int i = 0; intersecting_volume < volume && i < intersecting_elements.size(); ++i)
		{
			const size_t element_id = m.element_id(intersecting_elements[i]);

			for (int j = 0; j < M::FACETS_PER_ELEMENT; ++j)
			{
				const Tuple facet = m.tuple_from_facet(element_id, j);

				std::optional<Tuple> neighbor;
				if constexpr (std::is_same_v<M, TriMesh>)
					neighbor = facet.switch_face(m);
				else
					neighbor = facet.switch_tetrahedron(m);

				if (!neighbor.has_value() || intersecting_fid.find(m.element_id(neighbor.value())) != intersecting_fid.end())
					continue;

				intersecting_elements.push_back(neighbor.value());
				intersecting_fid.insert(m.element_id(neighbor.value()));
				intersecting_volume += m.element_volume(neighbor.value());
			}
		}
		assert(intersecting_volume >= volume);

		// ---------------------------------------------------------------------

		// Expand the ball to include the n-ring
		for (const auto &e : n_ring(m, one_ring, n_ring_size))
		{
			if (intersecting_fid.find(m.element_id(e)) == intersecting_fid.end())
			{
				intersecting_elements.push_back(e);
				intersecting_fid.insert(m.element_id(e));
				intersecting_volume += m.element_volume(e);
			}
		}

		assert(!has_duplicate_elements(m, intersecting_elements));

		return intersecting_elements;
	}

	template <typename M>
	Eigen::MatrixXi &LocalMesh<M>::boundary_facets()
	{
		if constexpr (std::is_same_v<M, TriMesh>)
			return m_boundary_edges;
		else
			return m_boundary_faces;
	}

	template <typename M>
	const Eigen::MatrixXi &LocalMesh<M>::boundary_facets() const
	{
		if constexpr (std::is_same_v<M, TriMesh>)
			return m_boundary_edges;
		else
			return m_boundary_faces;
	}

	template <typename M>
	void LocalMesh<M>::remove_duplicate_fixed_vertices()
	{
		std::sort(m_fixed_vertices.begin(), m_fixed_vertices.end());
		auto new_end = std::unique(m_fixed_vertices.begin(), m_fixed_vertices.end());
		m_fixed_vertices.erase(new_end, m_fixed_vertices.end());
	}

	template <typename M>
	void LocalMesh<M>::init_local_to_global()
	{
		m_local_to_global.resize(m_global_to_local.size(), -1);
		for (const auto &[glob_vi, loc_vi] : m_global_to_local)
		{
			assert(loc_vi < m_local_to_global.size());
			m_local_to_global[loc_vi] = glob_vi;
		}
	}

	template <typename M>
	void LocalMesh<M>::init_vertex_attributes(const M &m)
	{
		const int num_vertices = m_global_to_local.size();
		const int dim = m.dim();

		m_rest_positions.resize(num_vertices, dim);
		m_positions.resize(num_vertices, dim);
		m_projection_quantities.resize(num_vertices * dim, m.n_quantities());

		for (const auto &[glob_vi, loc_vi] : m_global_to_local)
		{
			m_rest_positions.row(loc_vi) = m.vertex_attrs[glob_vi].rest_position;
			m_positions.row(loc_vi) = m.vertex_attrs[glob_vi].position;
			m_projection_quantities.middleRows(dim * loc_vi, dim) =
				m.vertex_attrs[glob_vi].projection_quantities;
		}
	}

	template <typename M>
	void LocalMesh<M>::reorder_vertices(const Eigen::VectorXi &permutation)
	{
		assert(permutation.size() == num_vertices());
		m_rest_positions = utils::reorder_matrix(m_rest_positions, permutation);
		m_positions = utils::reorder_matrix(m_positions, permutation);

		m_projection_quantities = utils::reorder_matrix(
			m_projection_quantities, permutation, /*out_blocks=*/-1,
			/*block_size=*/M::DIM);

		m_elements = utils::map_index_matrix(m_elements, permutation);
		m_boundary_edges = utils::map_index_matrix(m_boundary_edges, permutation);
		m_boundary_faces = utils::map_index_matrix(m_boundary_faces, permutation);

		for (auto &[glob_vi, loc_vi] : m_global_to_local)
			loc_vi = permutation[loc_vi];

		std::vector<int> new_local_to_global(m_local_to_global.size());
		for (int i = 0; i < m_local_to_global.size(); ++i)
			new_local_to_global[permutation[i]] = m_local_to_global[i];
		m_local_to_global = new_local_to_global;

		for (int &vi : m_fixed_vertices)
			vi = permutation[vi];

		if constexpr (std::is_same_v<M, TriMesh>)
		{
			const Remesher::EdgeMap<int> &old_boundary_ids = std::get<Remesher::EdgeMap<int>>(m_boundary_ids);
			Remesher::EdgeMap<int> new_boundary_ids;
			for (const auto &[e, id] : old_boundary_ids)
			{
				const size_t v0 = permutation[e[0]], v1 = permutation[e[1]];
				new_boundary_ids[{{v0, v1}}] = id;
			}
			m_boundary_ids = new_boundary_ids;
		}
		else
		{
			const Remesher::FaceMap<int> &old_boundary_ids = std::get<Remesher::FaceMap<int>>(m_boundary_ids);
			Remesher::FaceMap<int> new_boundary_ids;
			for (const auto &[f, id] : old_boundary_ids)
			{
				const size_t v0 = permutation[f[0]], v1 = permutation[f[1]], v2 = permutation[f[2]];
				new_boundary_ids[{{v0, v1, v2}}] = id;
			}
			m_boundary_ids = new_boundary_ids;
		}
	}

	template <typename M>
	std::vector<polyfem::basis::ElementBases> LocalMesh<M>::build_bases(const std::string &formulation)
	{
		POLYFEM_REMESHER_SCOPED_TIMER("LocalMesh::build_bases");

		std::vector<polyfem::basis::ElementBases> bases;

		std::vector<LocalBoundary> local_boundary;
		Eigen::VectorXi vertex_to_basis;
		std::unique_ptr<Mesh> mesh = Mesh::create(rest_positions(), elements());
		int n_bases = Remesher::build_bases(
			*mesh, formulation, bases, local_boundary, vertex_to_basis);

		assert(n_bases == num_local_vertices());
		assert(vertex_to_basis.size() == num_local_vertices());
		n_bases = num_vertices();
		vertex_to_basis.conservativeResize(n_bases);

		const int start_i = num_local_vertices();
		if (start_i < n_bases)
		{
			// set tail to range [start_i, n_bases)
			std::iota(vertex_to_basis.begin() + start_i, vertex_to_basis.end(), start_i);
		}

		assert(std::all_of(vertex_to_basis.begin(), vertex_to_basis.end(), [](const int basis_id) {
			return basis_id >= 0;
		}));

		reorder_vertices(vertex_to_basis);

		return bases;
	}

	template <typename M>
	void LocalMesh<M>::write_mesh(
		const std::string &path, const Eigen::MatrixXd &sol) const
	{
		Eigen::VectorXd is_free = Eigen::VectorXd::Ones(num_vertices());
		is_free(fixed_vertices()) = Eigen::VectorXd::Zero(fixed_vertices().size());

		paraviewo::VTUWriter writer;
		writer.add_field("is_free", is_free);
		writer.add_field("displacement", utils::unflatten(sol, M::DIM));
		writer.write_mesh(path, rest_positions(), elements());
	}

	// -------------------------------------------------------------------------
	// Template instantiations
	template class LocalMesh<TriMesh>;
	template class LocalMesh<TetMesh>;
} // namespace polyfem::mesh