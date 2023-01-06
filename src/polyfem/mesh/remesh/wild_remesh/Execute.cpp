#include <polyfem/mesh/remesh/WildRemesher.hpp>

#include <polyfem/assembler/ElementAssemblyValues.hpp>
#include <polyfem/assembler/NeoHookeanElasticity.hpp>
#include <polyfem/mesh/remesh/wild_remesh/LocalMesh.hpp>
#include <polyfem/solver/NLProblem.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/GeometryUtils.hpp>
#include <polyfem/utils/Timer.hpp>

#include <wmtk/utils/ExecutorUtils.hpp>
#include <wmtk/utils/TupleUtils.hpp>

namespace polyfem::mesh
{
	template <class WMTKMesh>
	double WildRemesher<WMTKMesh>::edge_elastic_energy(const Tuple &e) const
	{
		using namespace polyfem::solver;

		const std::vector<Tuple> elements = get_incident_elements_for_edge(e);

		double volume = 0;
		for (const auto &t : elements)
			volume += element_volume(t);
		assert(volume > 0);

		LocalMesh local_mesh(*this, elements, /*include_global_boundary=*/false);

		const std::vector<polyfem::basis::ElementBases> bases = local_bases(local_mesh);
		const std::vector<int> boundary_nodes; // no boundary nodes
		assembler::AssemblerUtils assembler = create_assembler(local_mesh.body_ids());
		SolveData solve_data;
		assembler::AssemblyValsCache ass_vals_cache;
		Eigen::SparseMatrix<double> mass;
		ipc::CollisionMesh collision_mesh;

		local_solve_data(
			local_mesh, bases, boundary_nodes, assembler,
			solve_data, ass_vals_cache, mass, collision_mesh);

		const Eigen::MatrixXd sol = utils::flatten(local_mesh.displacements());

		return solve_data.nl_problem->value(sol) / volume; // average energy
	}

	template <class WMTKMesh>
	bool WildRemesher<WMTKMesh>::execute(
		const bool split,
		const bool collapse,
		const bool smooth,
		const bool swap,
		const double max_ops_percent)
	{
		using Operations = std::vector<std::pair<std::string, Tuple>>;

		POLYFEM_SCOPED_TIMER(timings.total);

		wmtk::logger().set_level(logger().level());

		Operations collect_all_ops;

		// TODO: implement face swaps for tet meshes

		const std::vector<Tuple> starting_elements = get_elements();
		std::vector<Tuple> included_edges;
		for (const Tuple &t : starting_elements)
		{
			const size_t t_id = element_id(t);

			if (element_attrs[t_id].excluded)
				continue;

			for (int ei = 0; ei < EDGES_IN_ELEMENT; ++ei)
			{
				included_edges.push_back(WMTKMesh::tuple_from_edge(t_id, ei));
			}
		}
		wmtk::unique_edge_tuples(*this, included_edges);

		const std::vector<Tuple> starting_edges = WMTKMesh::get_edges();
		for (const Tuple &e : included_edges)
		{
			if (split)
				collect_all_ops.emplace_back("edge_split", e);
			if (collapse)
				collect_all_ops.emplace_back("edge_collapse", e);
			if (swap)
				collect_all_ops.emplace_back("edge_swap", e);
		}

		const std::vector<Tuple> starting_vertices = WMTKMesh::get_vertices();
		if (smooth)
		{
			for (const Tuple &loc : starting_vertices)
			{
				// TODO: check average energy of surrounding faces
				collect_all_ops.emplace_back("vertex_smooth", loc);
			}
		}

		if (collect_all_ops.empty())
			return false;

		// if (NUM_THREADS > 0)
		// {
		// 	executor.lock_vertices = [&](WildRemesher &m, const Tuple &e, int task_id) -> bool {
		// 		return m.try_set_edge_mutex_n_ring(e, task_id, n_ring_size);
		// 	};
		// 	executor.num_threads = NUM_THREADS;
		// }

		executor.priority = [](const WildRemesher &m, std::string op, const Tuple &t) -> double {
			// NOTE: this code compute the edge length
			// return m.edge_length(t);
			return m.edge_elastic_energy(t);
		};

		executor.renew_neighbor_tuples = [&](const WildRemesher &m, std::string op, const std::vector<Tuple> &tris) -> Operations {
			return m.renew_neighbor_tuples(op, tris, split, collapse, smooth, swap);
		};

		// Split x% of edges
		int num_ops = 0;
		assert(std::isfinite(max_ops_percent * starting_edges.size()));
		const size_t max_ops =
			max_ops_percent >= 0
				? size_t(std::round(max_ops_percent * starting_edges.size()))
				: std::numeric_limits<size_t>::max();
		assert(max_ops > 0);
		executor.stopping_criterion = [&](const WildRemesher &m) -> bool {
			return (++num_ops) > max_ops;
		};
		executor.stopping_criterion_checking_frequency = 1;

		executor(*this, collect_all_ops);

		// Remove unused vertices
		WMTKMesh::consolidate_mesh();

		// if (executor.cnt_success() > 40)
		// {
		// 	logger().critical("exiting now for debugging purposes");
		// 	exit(0);
		// }

		static int aggregate_cnt_success = 0;
		static int aggregate_cnt_fail = 0;
		aggregate_cnt_success += executor.cnt_success();
		aggregate_cnt_fail += executor.cnt_fail();
		logger().info("aggregate_cnt_success {} aggregate_cnt_fail {}", aggregate_cnt_success, aggregate_cnt_fail);

		return executor.cnt_success() > 0;
	}

	// ------------------------------------------------------------------------
	// Template specializations
	template class WildRemesher<wmtk::TriMesh>;
	template class WildRemesher<wmtk::TetMesh>;

} // namespace polyfem::mesh