#include <polyfem/mesh/remesh/WildRemesh2D.hpp>
#include <polyfem/mesh/remesh/wild_remesh/AMIPSForm.hpp>
#include <polyfem/solver/NLProblem.hpp>
#include <polyfem/solver/SparseNewtonDescentSolver.hpp>
#include <polyfem/utils/Logger.hpp>

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
		return true;
	}

	bool WildRemeshing2D::smooth_after(const Tuple &t)
	{
		// Newton iterations are encapsulated here.
		logger().trace("Newton iteration for vertex smoothing.");
		const size_t vid = t.vid(*this);

		const std::vector<Tuple> locs = get_one_ring_tris_for_vertex(t);
		assert(locs.size() > 0);

		// Computes the maximal error around the one ring
		// that is needed to ensure the operation will decrease the error measure
		double max_quality = 0;
		for (const Tuple &tri : locs)
		{
			max_quality = std::max(max_quality, get_quality(tri));
		}

		assert(max_quality > 0); // If max quality is zero it is likely that the triangles are flipped

		// ---------------------------------------------------------------------

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

		// ---------------------------------------------------------------------

		vertex_attrs[vid].rest_position = new_rest_pos;
		// vertex_attrs[vid].position =

		return true; // TODO: check for inversion
	}

	void WildRemeshing2D::smooth_all_vertices()
	{
		// igl::Timer timer;
		// double time;
		// timer.start();
		std::vector<std::pair<std::string, Tuple>> collect_all_ops;
		for (const Tuple &loc : get_vertices())
		{
			collect_all_ops.emplace_back("vertex_smooth", loc);
		}
		// time = timer.getElapsedTime();
		// logger().info("vertex smoothing prepare time: {}s", time);
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
			// time = timer.getElapsedTime();
			// logger().info("vertex smoothing operation time parallel: {}s", time);
		}
		else
		{
			// timer.start();
			wmtk::ExecutePass<WildRemeshing2D, wmtk::ExecutionPolicy::kSeq> executor;
			executor(*this, collect_all_ops);
			// time = timer.getElapsedTime();
			// logger().info("vertex smoothing operation time serial: {}s", time);
		}
	}

} // namespace polyfem::mesh