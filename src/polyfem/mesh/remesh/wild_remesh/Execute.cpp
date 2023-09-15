#include <polyfem/mesh/remesh/WildRemesher.hpp>

#include <polyfem/utils/Timer.hpp>

#include <wmtk/utils/ExecutorUtils.hpp>
#include <wmtk/utils/TupleUtils.hpp>

// #define SAVE_OPS

namespace polyfem::mesh
{
	template <class WMTKMesh>
	bool WildRemesher<WMTKMesh>::execute()
	{
		const auto &reset_edge_attrs = [&](const Tuple &e) {
			edge_attr(e.eid(*this)).op_attempts = 0;
			edge_attr(e.eid(*this)).op_depth = 0;
		};

		utils::Timer timer(total_time);
		timer.start();

		const bool split = args["split"]["enabled"];
		const bool collapse = args["collapse"]["enabled"];
		const bool swap = args["swap"]["enabled"];
		const bool smooth = args["smooth"]["enabled"];

		wmtk::logger().set_level(logger().level());

		// if (NUM_THREADS > 0)
		// {
		// 	executor.lock_vertices = [&](WildRemesher &m, const Tuple &e, int task_id) -> bool {
		// 		return m.try_set_edge_mutex_n_ring(e, task_id, n_ring_size);
		// 	};
		// 	executor.num_threads = NUM_THREADS;
		// }

		executor.renew_neighbor_tuples = [&](const WildRemesher &m, std::string op, const std::vector<Tuple> &tris) -> Operations {
			return m.renew_neighbor_tuples(op, tris);
		};

		static int aggregate_split_cnt_success = 0;
		static int aggregate_split_cnt_fail = 0;
		static int aggregate_collapse_cnt_success = 0;
		static int aggregate_collapse_cnt_fail = 0;
		static int aggregate_swap_cnt_success = 0;
		static int aggregate_swap_cnt_fail = 0;
		static int aggregate_smooth_cnt_success = 0;
		static int aggregate_smooth_cnt_fail = 0;

#ifdef SAVE_OPS
		static int frame_count = 0;
		if (frame_count == 0)
			write_mesh(state.resolve_output_path(fmt::format("op{:d}.vtu", frame_count++)));
#endif

		int cnt_success = 0;

		cache_before();

		if (split)
		{
			logger().info("Splitting");
			split_edges();
			cnt_success += executor.cnt_success();
			aggregate_split_cnt_success += executor.cnt_success();
			aggregate_split_cnt_fail += executor.cnt_fail();
#ifdef SAVE_OPS
			write_mesh(state.resolve_output_path(fmt::format("op{:d}.vtu", frame_count++)));
#endif
		}

		// Reset operation attempts and depth counters
		WMTKMesh::for_each_edge(reset_edge_attrs);

		bool projection_needed = false;

		if (collapse)
		{
			logger().info("Collapsing");
			executor.m_cnt_success = 0;
			executor.m_cnt_fail = 0;
			collapse_edges();
			cnt_success += executor.cnt_success();
			aggregate_collapse_cnt_success += executor.cnt_success();
			aggregate_collapse_cnt_fail += executor.cnt_fail();
			projection_needed |= executor.cnt_success() > 0;
#ifdef SAVE_OPS
			write_mesh(state.resolve_output_path(fmt::format("op{:d}.vtu", frame_count++)));
#endif
		}

		// Reset operation attempts and depth counters
		WMTKMesh::for_each_edge(reset_edge_attrs);

		if (swap)
		{
			assert(DIM == 2);
			logger().info("Swapping");
			executor.m_cnt_success = 0;
			executor.m_cnt_fail = 0;
			swap_edges();
			cnt_success += executor.cnt_success();
			aggregate_swap_cnt_success += executor.cnt_success();
			aggregate_swap_cnt_fail += executor.cnt_fail();
			projection_needed |= executor.cnt_success() > 0;
#ifdef SAVE_OPS
			write_mesh(state.resolve_output_path(fmt::format("op{:d}.vtu", frame_count++)));
#endif
		}

		if (smooth)
		{
			assert(DIM == 2);
			logger().info("Smoothing");
			executor.m_cnt_success = 0;
			executor.m_cnt_fail = 0;
			smooth_vertices();
			cnt_success += executor.cnt_success();
			aggregate_smooth_cnt_success += executor.cnt_success();
			aggregate_smooth_cnt_fail += executor.cnt_fail();
			projection_needed |= executor.cnt_success() > 0;
#ifdef SAVE_OPS
			write_mesh(state.resolve_output_path(fmt::format("op{:d}.vtu", frame_count++)));
#endif
		}

		logger().info("[split]    aggregate_cnt_success {} aggregate_cnt_fail {}", aggregate_split_cnt_success, aggregate_split_cnt_fail);
		logger().info("[collapse] aggregate_cnt_success {} aggregate_cnt_fail {}", aggregate_collapse_cnt_success, aggregate_collapse_cnt_fail);
		logger().info("[swap]     aggregate_cnt_success {} aggregate_cnt_fail {}", aggregate_swap_cnt_success, aggregate_swap_cnt_fail);
		logger().info("[smooth]   aggregate_cnt_success {} aggregate_cnt_fail {}", aggregate_smooth_cnt_success, aggregate_smooth_cnt_fail);

		if (projection_needed)
			project_quantities();
#ifdef SAVE_OPS
		write_mesh(state.resolve_output_path(fmt::format("op{:d}.vtu", frame_count++)));
#endif

		// Remove unused vertices
		WMTKMesh::consolidate_mesh();

		timer.stop();
		log_timings();

		return cnt_success > 0;
	}

	// ------------------------------------------------------------------------
	// Template specializations
	template class WildRemesher<wmtk::TriMesh>;
	template class WildRemesher<wmtk::TetMesh>;

} // namespace polyfem::mesh