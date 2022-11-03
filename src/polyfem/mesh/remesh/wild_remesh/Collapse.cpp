#include <polyfem/mesh/remesh/WildRemesh2D.hpp>
#include <polyfem/utils/Logger.hpp>

#include <wmtk/ExecutionScheduler.hpp>

namespace polyfem::mesh
{
	bool WildRemeshing2D::collapse_edge_before(const Tuple &t)
	{
		if (!super::collapse_edge_before(t))
			return false;

		if (vertex_attrs[t.vid(*this)].fixed
			|| vertex_attrs[t.switch_vertex(*this).vid(*this)].fixed)
		{
			return false;
		}

		cache_before();
		edge_cache = EdgeCache(*this, t);

		return true;
	}

	bool WildRemeshing2D::collapse_edge_after(const Tuple &t)
	{
		size_t vid = t.vid(*this);

		vertex_attrs[vid].rest_position = (edge_cache.v0.rest_position + edge_cache.v1.rest_position) / 2.0;
		vertex_attrs[vid].partition_id = edge_cache.v0.partition_id;
		vertex_attrs[vid].fixed = false;

		// TODO: set the boundary_id of the new edges

		// vertex_attrs[vid].position = (cache[0].position + cache[1].position) / 2.0;
		// vertex_attrs[vid].velocity = (cache[0].velocity + cache[1].velocity) / 2.0;
		// vertex_attrs[vid].acceleration = (cache[0].acceleration + cache[1].acceleration) / 2.0;

		update_positions();

		const double energy_after = compute_global_energy();

		logger().critical("energy_before={} energy_after={} accept={}", energy_before, energy_after, energy_after < energy_before);
		return energy_after < energy_before;
	}

	void WildRemeshing2D::collapse_all_edges()
	{
		write_rest_obj("rest_mesh_before.obj");
		write_deformed_obj("deformed_mesh_before.obj");

		std::vector<std::pair<std::string, Tuple>> collect_all_ops;
		for (const Tuple &loc : get_edges())
		{
			collect_all_ops.emplace_back("edge_collapse", loc);
		}

		logger().debug("Num edges {}", collect_all_ops.size());
		if (NUM_THREADS > 0)
		{
			wmtk::ExecutePass<WildRemeshing2D, wmtk::ExecutionPolicy::kPartition> executor;
			executor.lock_vertices = [](WildRemeshing2D &m,
										const Tuple &e,
										int task_id) -> bool {
				return m.try_set_vertex_mutex_one_ring(e, task_id);
			};
			executor.num_threads = NUM_THREADS;
			executor(*this, collect_all_ops);
		}
		else
		{
			wmtk::ExecutePass<WildRemeshing2D, wmtk::ExecutionPolicy::kSeq> executor;
			executor(*this, collect_all_ops);
		}

		write_rest_obj("rest_mesh_after.obj");
		write_deformed_obj("deformed_mesh_after.obj");
	}

} // namespace polyfem::mesh