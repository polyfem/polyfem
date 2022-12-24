#include <polyfem/mesh/remesh/WildTriRemesher.hpp>

#include <polyfem/assembler/ElementAssemblyValues.hpp>
#include <polyfem/assembler/NeoHookeanElasticity.hpp>
#include <polyfem/mesh/remesh/wild_remesh/LocalMesh.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/GeometryUtils.hpp>
#include <polyfem/utils/Timer.hpp>

#include <wmtk/utils/ExecutorUtils.hpp>

namespace polyfem::mesh
{
	double WildTriRemesher::edge_elastic_energy(const Tuple &e) const
	{
		std::vector<Tuple> tris{{e}};
		if (e.switch_face(*this))
			tris.push_back(e.switch_face(*this).value());
		const auto local_mesh =
			LocalMesh<Super>(*this, tris, /*include_global_boundary=*/false);

		std::vector<polyfem::basis::ElementBases> bases;
		Eigen::VectorXi vertex_to_basis;
		WildTriRemesher::build_bases(
			local_mesh.rest_positions(), local_mesh.elements(), state.formulation(),
			bases, vertex_to_basis);

		const Eigen::VectorXd displacements = utils::flatten(utils::reorder_matrix(
			local_mesh.displacements(), vertex_to_basis));

		assembler::AssemblerUtils assembler = create_assembler(local_mesh.body_ids());

		assembler::AssemblyValsCache cache;
		const double energy = assembler.assemble_energy(
			state.formulation(), is_volume(), bases, /*gbases=*/bases, cache,
			/*dt=*/-1, displacements, /*displacement_prev=*/Eigen::MatrixXd());
		assert(std::isfinite(energy));
		return energy / local_mesh.num_elements(); // average energy per face
	}

	bool WildTriRemesher::execute(
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
		const std::vector<Tuple> starting_edges = get_edges();
		for (const Tuple &e : starting_edges)
		{
			// TODO: move this check to the _before function
			if (edge_elastic_energy(e) >= energy_absolute_tolerance)
			{
				// TODO: move this check to the _before function
				if (split && edge_length(e) >= min_edge_length)
					collect_all_ops.emplace_back("edge_split", e);
				if (collapse)
					collect_all_ops.emplace_back("edge_collapse", e);
				if (swap)
					collect_all_ops.emplace_back("edge_swap", e);
			}
		}

		const std::vector<Tuple> starting_vertices = get_vertices();
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

		wmtk::ExecutePass<WildTriRemesher, EXECUTION_POLICY> executor;
		// if (NUM_THREADS > 0)
		// {
		// 	executor.lock_vertices = [&](WildTriRemesher &m, const Tuple &e, int task_id) -> bool {
		// 		return m.try_set_edge_mutex_n_ring(e, task_id, n_ring_size);
		// 	};
		// 	executor.num_threads = NUM_THREADS;
		// }

		executor.priority = [](const WildTriRemesher &m, std::string op, const Tuple &t) -> double {
			// NOTE: this code compute the edge length
			// return m.edge_length(t);
			return m.edge_elastic_energy(t);
		};

		executor.renew_neighbor_tuples = [&](const WildTriRemesher &m, std::string op, const std::vector<Tuple> &tris) -> Operations {
			auto edges = m.new_edges_after(tris);
			Operations new_ops;
			for (auto &e : edges)
			{
				// TODO: move this check to the _before function
				if (m.edge_elastic_energy(e) >= energy_absolute_tolerance)
				{
					if (split && edge_length(e) >= min_edge_length)
						new_ops.emplace_back("edge_split", e);
					if (collapse)
						new_ops.emplace_back("edge_collapse", e);
					if (swap)
						new_ops.emplace_back("edge_swap", e);
				}
			}

			if (smooth)
			{
				assert(false);
			}

			return new_ops;
		};

		// Split x% of edges
		int num_ops = 0;
		assert(std::isfinite(max_ops_percent * starting_edges.size()));
		const size_t max_ops =
			max_ops_percent >= 0
				? size_t(std::round(max_ops_percent * starting_edges.size()))
				: std::numeric_limits<size_t>::max();
		assert(max_ops > 0);
		executor.stopping_criterion = [&](const WildTriRemesher &m) -> bool {
			return (++num_ops) > max_ops;
		};
		executor.stopping_criterion_checking_frequency = 1;

		executor(*this, collect_all_ops);

		// Remove unused vertices
		consolidate_mesh();

		return true;
	}

} // namespace polyfem::mesh