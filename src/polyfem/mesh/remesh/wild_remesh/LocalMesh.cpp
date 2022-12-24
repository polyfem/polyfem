#include "LocalMesh.hpp"

#include <polyfem/mesh/remesh/WildRemesher.hpp>
#include <polyfem/utils/Timer.hpp>
#include <polyfem/utils/GeometryUtils.hpp>
#include <polyfem/utils/MatrixUtils.hpp>
#include <polyfem/io/VTUWriter.hpp>

#include <igl/boundary_facets.h>
#include <igl/edges.h>

#include <BVH.hpp>

namespace polyfem::mesh
{
	using TriMesh = WildRemesher<wmtk::TriMesh>;
	using TetMesh = WildRemesher<wmtk::TetMesh>;

	template <typename M>
	LocalMesh<M>::LocalMesh(
		const M &m,
		const std::vector<Tuple> &element_tuples,
		const bool include_global_boundary)
	{
		std::unordered_set<size_t> global_element_ids;

		m_elements.resize(element_tuples.size(), m.dim() + 1);
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
		// The above puts local vertices at front
		m_num_local_vertices = m_global_to_local.size();

		for (int fi = 0; fi < num_elements(); fi++)
		{
			const Tuple &elem = element_tuples[fi];

			for (int i = 0; i < M::DIM + 1; ++i)
			{
				const Tuple facet = m.tuple_from_facet(m.element_id(elem), i);

				// Only fix internal facets
				if (m.is_on_boundary(facet))
					continue;

				size_t adjacent_eid;
				if constexpr (std::is_same_v<M, TriMesh>)
					adjacent_eid = facet.switch_face(m)->fid(m);
				else
					adjacent_eid = facet.switch_tetrahedron(m)->tid(m);

				// Only fix internal facets that do not have a neighbor in the local mesh
				if (global_element_ids.find(adjacent_eid) != global_element_ids.end())
					continue;

				for (const size_t &vid : m.boundary_facet_vids(facet))
					m_fixed_vertices.push_back(m_global_to_local[vid]);
			}
		}

		// ---------------------------------------------------------------------

		if (include_global_boundary)
		{
			const std::unordered_map<int, int> prev_global_to_local = m_global_to_local;

			const std::vector<Tuple> global_boundary_facets = m.boundary_facets();
			boundary_facets().resize(global_boundary_facets.size(), 2);
			for (int i = 0; i < global_boundary_facets.size(); i++)
			{
				const Tuple &facet = global_boundary_facets[i];
				const auto vids = m.boundary_facet_vids(facet);

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

				m_boundary_ids.push_back(m.boundary_attrs[m.facet_id(facet)].boundary_id);
			}
		}
		else
		{
			igl::boundary_facets(m_elements, boundary_facets());
			m_fixed_vertices.insert(
				m_fixed_vertices.end(), boundary_facets().data(),
				boundary_facets().data() + boundary_facets().size());
			// TODO:
			// m_boundary_ids.push_back(m.boundary_attrs[e.eid(m)].boundary_id);
		}

		if (m_boundary_faces.rows() > 0)
			igl::edges(m_boundary_faces, m_boundary_edges);

		remove_duplicate_fixed_vertices();

		// ---------------------------------------------------------------------

		init_vertex_attributes(m);

		init_local_to_global();

		const int tmp_num_vertices = num_vertices();

		if (include_global_boundary && m.obstacle().n_vertices() > 0)
		{
			const Obstacle &obstacle = m.obstacle();
			utils::append_rows(m_rest_positions, obstacle.v());
			utils::append_rows(m_positions, obstacle.v() + m.obstacle_displacements());
			utils::append_rows(m_projection_quantities, m.obstacle_quantities());
			utils::append_rows(m_boundary_edges, obstacle.e().array() + tmp_num_vertices);
			utils::append_rows(m_boundary_faces, obstacle.f().array() + tmp_num_vertices);

			for (int i = 0; i < obstacle.n_vertices(); i++)
				m_fixed_vertices.push_back(i + tmp_num_vertices);

			if constexpr (std::is_same_v<M, TriMesh>)
				for (int i = 0; i < obstacle.n_edges(); i++)
					m_boundary_ids.push_back(std::numeric_limits<int>::max());
			else
				for (int i = 0; i < obstacle.n_faces(); i++)
					m_boundary_ids.push_back(std::numeric_limits<int>::max());
		}
	}

	template <typename M>
	LocalMesh<M> LocalMesh<M>::n_ring(
		const M &m,
		const Tuple &center,
		const int n,
		const bool include_global_boundary)
	{
		std::vector<Tuple> elements = m.get_one_ring_elements_for_vertex(center);
		std::unordered_set<size_t> visited_vertices{{center.vid(m)}};
		std::unordered_set<size_t> visited_faces;
		for (const auto &element : elements)
		{
			visited_faces.insert(m.element_id(element));
		}

		std::vector<Tuple> new_elements = elements;

		for (int i = 1; i < n; i++)
		{
			std::vector<Tuple> new_new_elements;
			for (const auto &elem : new_elements)
			{
				const auto vs = m.element_vertices(elem);
				for (const Tuple &v : vs)
				{
					if (visited_vertices.find(v.vid(m)) != visited_vertices.end())
						continue;
					visited_vertices.insert(v.vid(m));

					std::vector<Tuple> tmp = m.get_one_ring_elements_for_vertex(v);
					for (auto &t1 : tmp)
					{
						if (visited_faces.find(t1.fid(m)) != visited_faces.end())
							continue;
						visited_faces.insert(t1.fid(m));
						elements.push_back(t1);
						new_new_elements.push_back(t1);
					}
				}
			}
			new_elements = new_new_elements;
			if (new_elements.empty())
				break;
		}

		return LocalMesh(m, elements, include_global_boundary);
	}

	template <typename M>
	LocalMesh<M> LocalMesh<M>::flood_fill_n_ring(
		const M &m,
		const Tuple &center,
		const double area,
		const bool include_global_boundary)
	{
		POLYFEM_SCOPED_TIMER(m.timings.create_local_mesh);

		double current_area = 0;

		std::vector<Tuple> elements = m.get_one_ring_elements_for_vertex(center);
		std::unordered_set<size_t> visited_vertices{{center.vid(m)}};
		std::unordered_set<size_t> visited_faces;
		for (const auto &element : elements)
			visited_faces.insert(element.fid(m));

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
						if (visited_faces.find(t1.fid(m)) != visited_faces.end())
							continue;
						visited_faces.insert(t1.fid(m));
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

		return LocalMesh(m, elements, include_global_boundary);
	}

	template <typename M>
	LocalMesh<M> LocalMesh<M>::ball_selection(
		const M &m,
		const VectorNd &center,
		const double rel_radius,
		const bool include_global_boundary)
	{
		POLYFEM_SCOPED_TIMER(m.timings.create_local_mesh);

		const int dim = m.dim();

		Eigen::Array3d center3D = Eigen::Array3d::Zero();
		center3D.head(dim) = center;

		Eigen::MatrixXd V = m.rest_positions();
		const double radius = rel_radius * (V.colwise().maxCoeff() - V.colwise().minCoeff()).norm();

		const std::vector<Tuple> elements = m.get_elements();

		// Use a AABB tree to find all intersecting elements then loop over only those pairs
		std::vector<std::array<Eigen::Vector3d, 2>> boxes(
			elements.size(), {{Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()}});
		for (int i = 0; i < elements.size(); i++)
		{
			const auto vids = m.element_vids(elements[i]);
			boxes[i][0].head(dim) = V(vids, Eigen::all).colwise().minCoeff();
			boxes[i][1].head(dim) = V(vids, Eigen::all).colwise().maxCoeff();
		}

		BVH::BVH bvh;
		bvh.init(boxes);

		std::vector<unsigned int> candidates;
		bvh.intersect_box(center3D - radius, center3D + radius, candidates);

		std::vector<Tuple> intersecting_elements;
		for (unsigned int fi : candidates)
		{
			bool is_intersecting;
			const auto vids = m.element_vids(elements[fi]);
			if constexpr (std::is_same_v<M, TriMesh>)
			{
				is_intersecting = utils::triangle_intersects_disk(
					V.row(vids[0]), V.row(vids[1]), V.row(vids[2]),
					center, radius);
			}
			else
			{
				static_assert(std::is_same_v<M, TetMesh>);
				is_intersecting = utils::tetrahedron_intersects_ball(
					V.row(vids[0]), V.row(vids[1]), V.row(vids[2]), V.row(vids[3]),
					center, radius);
			}

			if (is_intersecting)
				intersecting_elements.push_back(elements[fi]);
		}

		return LocalMesh<M>(m, intersecting_elements, include_global_boundary);
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
	}

	template <typename M>
	void LocalMesh<M>::write_mesh(
		const std::string &path, const Eigen::MatrixXd &sol) const
	{
		Eigen::VectorXd is_free = Eigen::VectorXd::Ones(num_vertices());
		is_free(fixed_vertices()) = Eigen::VectorXd::Zero(fixed_vertices().size());

		io::VTUWriter writer;
		writer.add_field("is_free", is_free);
		writer.add_field("displacement", utils::unflatten(sol, M::DIM));
		writer.write_mesh(path, rest_positions(), elements());
	}

	// -------------------------------------------------------------------------
	// Template specializations

	template class LocalMesh<TriMesh>;
	template class LocalMesh<TetMesh>;
} // namespace polyfem::mesh