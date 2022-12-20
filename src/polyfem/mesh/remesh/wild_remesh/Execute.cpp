#include <polyfem/mesh/remesh/WildRemeshing2D.hpp>

#include <polyfem/assembler/ElementAssemblyValues.hpp>
#include <polyfem/assembler/NeoHookeanElasticity.hpp>
#include <polyfem/mesh/remesh/wild_remesh/LocalMesh.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/GeometryUtils.hpp>
#include <polyfem/utils/Timer.hpp>

#include <wmtk/utils/ExecutorUtils.hpp>

namespace polyfem::mesh
{
	double WildRemeshing2D::edge_length(const Tuple &e) const
	{
		const Eigen::Vector2d &e0 = vertex_attrs[e.vid(*this)].position;
		const Eigen::Vector2d &e1 = vertex_attrs[e.switch_vertex(*this).vid(*this)].position;
		return (e1 - e0).norm();
	}

	double WildRemeshing2D::edge_elastic_energy(const Tuple &e) const
	{
		std::vector<Tuple> tris{{e}};
		if (e.switch_face(*this))
			tris.push_back(e.switch_face(*this).value());
		const LocalMesh local_mesh(*this, tris, /*include_global_boundary=*/false);

		std::vector<polyfem::basis::ElementBases> bases;
		Eigen::VectorXi vertex_to_basis;
		WildRemeshing2D::build_bases(
			local_mesh.rest_positions(), local_mesh.triangles(), state.formulation(),
			bases, vertex_to_basis);

		const Eigen::VectorXd displacements = utils::flatten(utils::reorder_matrix(
			local_mesh.displacements(), vertex_to_basis));

		assembler::AssemblerUtils assembler = create_assembler(local_mesh.body_ids());

		assembler::AssemblyValsCache cache;
		const double energy = assembler.assemble_energy(
			state.formulation(), is_volume(), bases, /*gbases=*/bases, cache,
			/*dt=*/-1, displacements, /*displacement_prev=*/Eigen::MatrixXd());
		assert(std::isfinite(energy));
		return energy / local_mesh.num_triangles(); // average energy per face
	}

	// double WildRemeshing2D::edge_stress(const Tuple &e) const
	// {
	// 	Eigen::MatrixXd V, U, _, __, ___;
	// 	Eigen::MatrixXi F;
	// 	std::unordered_map<size_t, size_t> vi_map;
	// 	std::vector<int> body_ids;
	// 	std::vector<std::unordered_set<int>> vertex_boundary_ids;
	// 	{
	// 		std::vector<Tuple> tris{{e}};
	// 		if (e.switch_face(*this))
	// 			tris.push_back(e.switch_face(*this).value());
	// 		build_local_matricies(
	// 			tris, V, U, _, __, ___, F, vi_map, body_ids, vertex_boundary_ids);
	// 	}

	// 	std::vector<polyfem::basis::ElementBases> bases;
	// 	Eigen::VectorXi vertex_to_basis;
	// 	WildRemeshing2D::build_bases(V, F, state.formulation(), bases, vertex_to_basis);

	// 	const Eigen::VectorXd displacements = utils::flatten(utils::reorder_matrix(U, vertex_to_basis));

	// 	double max_stress = -std::numeric_limits<double>::infinity();
	// 	for (int el_id = 0; el_id < F.rows(); el_id++)
	// 	{
	// 		Eigen::MatrixXd local_pts(1, DIM);
	// 		local_pts << 1 / 3.0, 1 / 3.0;

	// 		Eigen::MatrixXd stress(1, 1);
	// 		m.state.assembler.compute_scalar_value(
	// 			m.assembler_formulation,
	// 			el_id,
	// 			bases[el_id],
	// 			/*gbases=*/bases[el_id],
	// 			local_pts,
	// 			displacements,
	// 			stress);
	// 		stress *= utils::triangle_area_2D(
	// 			V.row(F(el_id, 0)), V.row(F(el_id, 1)), V.row(F(el_id, 2)));

	// 		max_stress = std::max(max_stress, stress(0));
	// 	}
	// 	assert(std::isfinite(max_stress));
	// 	return max_stress;
	// }

	bool WildRemeshing2D::execute(
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
			if (edge_elastic_energy(e) >= energy_absolute_tolerance)
			{
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

		wmtk::ExecutePass<WildRemeshing2D, EXECUTION_POLICY> executor;
		// if (NUM_THREADS > 0)
		// {
		// 	executor.lock_vertices = [&](WildRemeshing2D &m, const Tuple &e, int task_id) -> bool {
		// 		return m.try_set_edge_mutex_n_ring(e, task_id, n_ring_size);
		// 	};
		// 	executor.num_threads = NUM_THREADS;
		// }

		executor.priority = [](const WildRemeshing2D &m, std::string op, const Tuple &t) -> double {
			// NOTE: this code compute the edge length
			// return m.edge_length(t);
			return m.edge_elastic_energy(t);
		};

		executor.renew_neighbor_tuples = [&](const WildRemeshing2D &m, std::string op, const std::vector<Tuple> &tris) -> Operations {
			auto edges = m.new_edges_after(tris);
			Operations new_ops;
			for (auto &e : edges)
			{
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
		executor.stopping_criterion = [&](const WildRemeshing2D &m) -> bool {
			return (++num_ops) > max_ops;
		};
		executor.stopping_criterion_checking_frequency = 1;

		executor(*this, collect_all_ops);

		return true;
	}

} // namespace polyfem::mesh