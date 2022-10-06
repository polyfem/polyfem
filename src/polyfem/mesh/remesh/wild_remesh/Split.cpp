#include <polyfem/mesh/remesh/WildRemesh2D.hpp>

#include <wmtk/utils/ExecutorUtils.hpp>

namespace polyfem::mesh
{
	bool WildRemeshing2D::split_edge_before(const Tuple &t)
	{
		if (!super::split_edge_before(t))
			return false;
		const size_t vi = t.vid(*this);
		const size_t vj = t.switch_vertex(*this).vid(*this);

		// Dont split if the edge is too small
		// if ((vertex_attrs[vj].position - vertex_attrs[vi].position).norm() < 5e-3)
		// 	return false;

		// if (vertex_attrs[vi].frozen && vertex_attrs[vj].frozen)
		// 	return false;

		cache = {{vertex_attrs[vi], vertex_attrs[vj]}};
		energy_before = compute_global_energy();
		// energy_before = compute_global_wicke_measure();

		return true;
	}

	bool WildRemeshing2D::split_edge_after(const Tuple &t)
	{
		size_t vid = t.vid(*this);

		vertex_attrs[vid].rest_position = (cache[0].rest_position + cache[1].rest_position) / 2.0;
		vertex_attrs[vid].position = (cache[0].position + cache[1].position) / 2.0;
		vertex_attrs[vid].velocity = (cache[0].velocity + cache[1].velocity) / 2.0;
		vertex_attrs[vid].acceleration = (cache[0].acceleration + cache[1].acceleration) / 2.0;
		vertex_attrs[vid].partition_id = cache[0].partition_id;
		vertex_attrs[vid].frozen = cache[0].frozen && cache[1].frozen;

		double energy_after = compute_global_energy();
		// double energy_after = compute_global_wicke_measure();

		// logger().critical("energy_before={} energy_after={} accept={}", energy_before, energy_after, energy_after < energy_before);
		// return energy_after < energy_before;
		// return energy_after < energy_before - 1e-14;
		return true;
	}

	void WildRemeshing2D::split_all_edges()
	{
		using Operations = std::vector<std::pair<std::string, Tuple>>;

		write_rest_obj("rest_mesh_before.obj");
		write_deformed_obj("deformed_mesh_before.obj");

		Operations collect_all_ops;
		for (const Tuple &e : get_edges())
		{
			collect_all_ops.emplace_back("edge_split", e);
		}

		wmtk::ExecutePass<WildRemeshing2D, EXECUTION_POLICY> executor;
		// if (NUM_THREADS > 0)
		// {
		// 	executor.lock_vertices = [&](WildRemeshing2D &m, const Tuple &e, int task_id) -> bool {
		// 		return m.try_set_edge_mutex_two_ring(e, task_id);
		// 	};
		// 	executor.num_threads = NUM_THREADS;
		// }

		executor.priority = [](const WildRemeshing2D &m, std::string op, const Tuple &t) -> double {
			return (m.vertex_attrs[t.vid(m)].position
					- m.vertex_attrs[t.switch_vertex(m).vid(m)].position)
				.squaredNorm();
		};

		executor.renew_neighbor_tuples = [](const WildRemeshing2D &m, std::string op, const std::vector<Tuple> &tris) -> Operations {
			auto edges = m.new_edges_after(tris);
			Operations new_ops;
			for (auto &e : edges)
				new_ops.emplace_back("edge_split", e);
			return new_ops;
		};

		// Split 25% of edges
		int num_splits = 0;
		const int max_splits = std::round(0.25 * collect_all_ops.size());
		executor.stopping_criterion = [&](const WildRemeshing2D &m) -> bool {
			return (++num_splits) > max_splits;
		};
		executor.stopping_criterion_checking_frequency = 1;

		executor(*this, collect_all_ops);

		write_rest_obj("rest_mesh_after.obj");
		write_deformed_obj("deformed_mesh_after.obj");
	}

} // namespace polyfem::mesh