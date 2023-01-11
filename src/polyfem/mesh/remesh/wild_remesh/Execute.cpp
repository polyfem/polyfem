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
		const bool swap)
	{
		POLYFEM_SCOPED_TIMER(timings.total);

		wmtk::logger().set_level(logger().level());

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
			return m.renew_neighbor_tuples(op, tris);
		};

		static int aggregate_split_cnt_success = 0;
		static int aggregate_split_cnt_fail = 0;
		static int aggregate_collapse_cnt_success = 0;
		static int aggregate_collapse_cnt_fail = 0;

		int cnt_success = 0;

		if (split)
		{
			split_edges();

			cnt_success += executor.cnt_success();

			aggregate_split_cnt_success += executor.cnt_success();
			aggregate_split_cnt_fail += executor.cnt_fail();
		}

		if (collapse)
		{
			collapse_edges();

			cnt_success += executor.cnt_success();

			aggregate_collapse_cnt_success += executor.cnt_success();
			aggregate_collapse_cnt_fail += executor.cnt_fail();
		}

		assert(!swap);
		assert(!smooth);

		logger().info("[split]    aggregate_cnt_success {} aggregate_cnt_fail {}", aggregate_split_cnt_success, aggregate_split_cnt_fail);
		logger().info("[collapse] aggregate_cnt_success {} aggregate_cnt_fail {}", aggregate_collapse_cnt_success, aggregate_collapse_cnt_fail);

		// Remove unused vertices
		WMTKMesh::consolidate_mesh();

		// if (executor.cnt_success() > 40)
		// {
		// 	logger().critical("exiting now for debugging purposes");
		// 	exit(0);
		// }

		return cnt_success > 0;
	}

	// ------------------------------------------------------------------------
	// Template specializations
	template class WildRemesher<wmtk::TriMesh>;
	template class WildRemesher<wmtk::TetMesh>;

} // namespace polyfem::mesh