#include "WildRemesh2D.hpp"

#include <wmtk/ExecutionScheduler.hpp>
#include <wmtk/utils/TupleUtils.hpp>

namespace polyfem::mesh
{
	WildRemeshing2D::WildRemeshing2D(
		Eigen::MatrixXd rest_positions,
		Eigen::MatrixXd displacements,
		Eigen::MatrixXd velocities,
		Eigen::MatrixXd accelerations,
		int num_threads)
	{
		NUM_THREADS = num_threads;
		p_vertex_attrs = &vertex_attrs;

		const size_t n_nodes = rest_positions.rows();
		assert(n_nodes == displacements.rows());
		assert(n_nodes == velocities.rows());
		assert(n_nodes == accelerations.rows());

		vertex_attrs.resize(n_nodes);
		for (auto i = 0; i < n_nodes; i++)
			vertex_attrs[i] = {
				rest_positions.row(i),
				displacements.row(i),
				velocities.row(i),
				accelerations.row(i),
				0,
				false,
			};
	}

	void WildRemeshing2D::freeze_vertex(TriMesh::Tuple &v)
	{
		for (auto e : get_one_ring_edges_for_vertex(v))
		{
			if (is_boundary_edge(e))
			{
				vertex_attrs[v.vid(*this)].frozen = true;
				continue;
			}
		}
	}

	void WildRemeshing2D::create_mesh(
		size_t n_vertices,
		const std::vector<std::array<size_t, 3>> &tris)
	{
		super::create_mesh(n_vertices, tris);
		partition_mesh();
		// the better way is to iterate through edges.
		for (auto v : get_vertices())
		{
			freeze_vertex(v);
		}
	}

	void WildRemeshing2D::partition_mesh()
	{
		// auto m_vertex_partition_id = partition_TriMesh(*this, NUM_THREADS);
		// for (auto i = 0; i < m_vertex_partition_id.size(); i++)
		// 	vertex_attrs[i].partition_id = m_vertex_partition_id[i];
	}

	bool WildRemeshing2D::invariants(const std::vector<Tuple> &new_tris)
	{
		return true;
	}

	void WildRemeshing2D::build_mesh_matrices(Eigen::MatrixXd &V, Eigen::MatrixXi &F)
	{
		V.setZero(vert_capacity(), 3);
		for (auto &t : get_vertices())
		{
			auto i = t.vid(*this);
			V.row(i) = vertex_attrs[i].rest_position;
		}

		F.setConstant(tri_capacity(), 3, -1);
		for (auto &t : get_faces())
		{
			auto i = t.fid(*this);
			auto vs = oriented_tri_vertices(t);
			for (int j = 0; j < 3; j++)
			{
				F(i, j) = vs[j].vid(*this);
			}
		}
	}

	bool WildRemeshing2D::collapse_edge_before(const Tuple &t)
	{
		if (!super::collapse_edge_before(t))
			return false;
		if (is_vertex_frozen(t) && is_vertex_frozen(t.switch_vertex(*this)))
			return false;
		position_cache.local().v1_attr = vertex_attrs[t.vid(*this)];
		position_cache.local().v2_attr = vertex_attrs[t.switch_vertex(*this).vid(*this)];
		return true;
	}

	bool WildRemeshing2D::collapse_edge_after(const TriMesh::Tuple &t)
	{
		const auto &[v1_attr, v2_attr] = position_cache.local();
		VertexAttributes &v_attr = vertex_attrs[t.vid(*this)];
		const int partition_id = v_attr.partition_id;

		if (v1_attr.frozen)
			v_attr = v1_attr;
		else if (v2_attr.frozen)
			v_attr = v2_attr;
		else
		{
			vertex_attrs[t.vid(*this)].rest_position = (v1_attr.rest_position + v2_attr.rest_position) / 2.0;
			vertex_attrs[t.vid(*this)].displacement = (v1_attr.displacement + v2_attr.displacement) / 2.0;
			vertex_attrs[t.vid(*this)].velocity = (v1_attr.velocity + v2_attr.velocity) / 2.0;
			vertex_attrs[t.vid(*this)].acceleration = (v1_attr.acceleration + v2_attr.acceleration) / 2.0;
			vertex_attrs[t.vid(*this)].frozen = false;
		}
		v_attr.partition_id = partition_id;

		return true;
	}

	std::vector<wmtk::TriMesh::Tuple> WildRemeshing2D::new_edges_after(
		const std::vector<wmtk::TriMesh::Tuple> &tris) const
	{
		std::vector<wmtk::TriMesh::Tuple> new_edges;
		std::vector<size_t> one_ring_fid;

		for (auto t : tris)
		{
			for (auto j = 0; j < 3; j++)
			{
				new_edges.push_back(tuple_from_edge(t.fid(*this), j));
			}
		}
		wmtk::unique_edge_tuples(*this, new_edges);
		return new_edges;
	}

	bool WildRemeshing2D::collapse_shortest(int target_vert_number)
	{
		size_t initial_size = get_vertices().size();
		auto collect_all_ops = std::vector<std::pair<std::string, Tuple>>();
		for (auto &loc : get_edges())
			collect_all_ops.emplace_back("edge_collapse", loc);

		auto renew = [](auto &m, auto op, auto &tris) {
			auto edges = m.new_edges_after(tris);
			auto optup = std::vector<std::pair<std::string, Tuple>>();
			for (auto &e : edges)
				optup.emplace_back("edge_collapse", e);
			return optup;
		};
		auto measure_len2 = [](auto &m, auto op, const Tuple &new_e) {
			auto len2 =
				(m.vertex_attrs[new_e.vid(m)].pos - m.vertex_attrs[new_e.switch_vertex(m).vid(m)].pos)
					.squaredNorm();
			return -len2;
		};
		auto setup_and_execute = [&](auto executor) {
			executor.num_threads = NUM_THREADS;
			executor.renew_neighbor_tuples = renew;
			executor.priority = measure_len2;
			executor.stopping_criterion_checking_frequency =
				target_vert_number > 0 ? (initial_size - target_vert_number - 1)
									   : std::numeric_limits<int>::max();
			executor.stopping_criterion = [](auto &m) { return true; };
			executor(*this, collect_all_ops);
		};

		if (NUM_THREADS > 0)
		{
			auto executor = wmtk::ExecutePass<WildRemeshing2D, wmtk::ExecutionPolicy::kPartition>();
			executor.lock_vertices = [](auto &m, const auto &e, int task_id) {
				return m.try_set_edge_mutex_two_ring(e, task_id);
			};
			setup_and_execute(executor);
		}
		else
		{
			auto executor = wmtk::ExecutePass<WildRemeshing2D, wmtk::ExecutionPolicy::kSeq>();
			setup_and_execute(executor);
		}
		return true;
	}

} // namespace polyfem::mesh