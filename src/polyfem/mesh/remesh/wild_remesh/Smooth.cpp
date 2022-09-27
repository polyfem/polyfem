#include <polyfem/mesh/remesh/WildRemesh2D.hpp>
#include <polyfem/mesh/remesh/wild_remesh/AMIPSForm.hpp>
#include <polyfem/solver/NLProblem.hpp>
#include <polyfem/solver/SparseNewtonDescentSolver.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/io/OBJWriter.hpp>

#include <wmtk/ExecutionScheduler.hpp>
#include <wmtk/utils/TriQualityUtils.hpp>
#include <wmtk/utils/TupleUtils.hpp>

namespace polyfem::mesh
{
	using namespace polyfem::solver;

	bool WildRemeshing2D::smooth_before(const Tuple &t)
	{
		if (vertex_attrs[t.vid(*this)].frozen)
			return false;

		rest_positions_before = rest_positions();
		positions_before = positions();
		velocities_before = velocities();
		accelerations_before = accelerations();
		triangles_before = triangles();
		energy_before = compute_global_energy();
		write_rest_obj("rest_mesh_before.obj");
		write_deformed_obj("deformed_mesh_before.obj");

		return true;
	}

	bool WildRemeshing2D::smooth_after(const Tuple &t)
	{
		const size_t vid = t.vid(*this);

		const std::vector<Tuple> locs = get_one_ring_tris_for_vertex(t);
		assert(locs.size() > 0);

		std::vector<std::shared_ptr<Form>> forms;
		for (const Tuple &loc : locs)
		{
			// For each triangle, make a reordered copy of the vertices so that
			// the vertex to optimize is always the first
			assert(!is_inverted(loc));
			const std::array<Tuple, 3> local_tuples = oriented_tri_vertices(loc);
			std::array<size_t, 3> local_verts;
			for (int i = 0; i < 3; i++)
			{
				local_verts[i] = local_tuples[i].vid(*this);
			}
			local_verts = wmtk::orient_preserve_tri_reorder(local_verts, vid);

			Eigen::MatrixXd rest_positions(3, 2), positions(3, 2);
			for (int i = 0; i < 3; ++i)
			{
				rest_positions.row(i) = vertex_attrs[local_verts[i]].rest_position;
				positions.row(i) = vertex_attrs[local_verts[i]].position;
			}

			forms.push_back(std::make_shared<AMIPSForm>(rest_positions, positions));
		}

		FullNLProblem problem(forms);

		// ---------------------------------------------------------------------

		// Make a backup of the current configuration
		const Eigen::Vector2d old_rest_pos = vertex_attrs[vid].rest_position;

		problem.init(old_rest_pos);

		// TODO: expose these parameters
		const json newton_args = R"({
				"f_delta": 1e-7,
				"grad_norm": 1e-7,
				"first_grad_norm_tol": 1e-10,
				"max_iterations": 100,
				"use_grad_norm": true,
				"relative_gradient": false,
				"line_search": {
					"method": "backtracking",
					"use_grad_norm_tol": 0.0001
				}
			})"_json;
		const json linear_solver_args = R"({
			"solver": "Eigen::PardisoLDLT",
			"precond": "Eigen::IdentityPreconditioner"
		})"_json;
		cppoptlib::SparseNewtonDescentSolver<FullNLProblem> solver(newton_args, linear_solver_args);

		Eigen::VectorXd new_rest_pos = old_rest_pos;
		try
		{
			solver.minimize(problem, new_rest_pos);
		}
		catch (const std::exception &e)
		{
			logger().warn("Newton solver failed: {}", e.what());
			return false;
		}

		logger().critical("old pos {} -> new pos {}", old_rest_pos.transpose(), new_rest_pos.transpose());

		vertex_attrs[vid].rest_position = new_rest_pos;

		double energy_before_projection = compute_global_energy();
		logger().critical("energy_before={} energy_before_projection={}", energy_before, energy_before_projection);

		update_positions();

		double energy_after = compute_global_energy();

		logger().critical("energy_before={} energy_after={} accept={}", energy_before, energy_after, energy_after < energy_before);
		return energy_after < energy_before;
	}

	void WildRemeshing2D::smooth_all_vertices()
	{
		std::vector<std::pair<std::string, Tuple>> collect_all_ops;
		for (const Tuple &loc : get_vertices())
		{
			collect_all_ops.emplace_back("vertex_smooth", loc);
		}
		logger().debug("Num verts {}", collect_all_ops.size());
		if (NUM_THREADS > 0)
		{
			// timer.start();
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
	}

} // namespace polyfem::mesh