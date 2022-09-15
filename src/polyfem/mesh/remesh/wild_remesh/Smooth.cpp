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

			Eigen::MatrixXd X_rest(3, 2), X(3, 2);
			for (int i = 0; i < 3; ++i)
			{
				X_rest.row(i) = vertex_attrs[local_verts[i]].rest_position;
				X.row(i) = vertex_attrs[local_verts[i]].position();
			}

			forms.push_back(std::make_shared<AMIPSForm>(X_rest, X));
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
					"use_grad_norm_tol": false
				}
			})"_json;
		cppoptlib::SparseNewtonDescentSolver<FullNLProblem> solver(
			newton_args,
			"Eigen::PardisoLDLT",
			"Eigen::IdentityPreconditioner");
		solver.set_line_search(newton_args["line_search"]["method"]);
		Eigen::VectorXd new_rest_pos = old_rest_pos;
		solver.minimize(problem, new_rest_pos);

		// ---------------------------------------------------------------------

		vertex_attrs[vid].rest_position = new_rest_pos;
		// vertex_attrs[vid].displacement =
		// vertex_attrs[vid].velocity =
		// vertex_attrs[vid].acceleration =

		// Logging
		logger().info("old pos {} -> new pos {}", old_rest_pos.transpose(), new_rest_pos.transpose());

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

	Eigen::Vector2d newton_method_from_stack_2d(
		std::vector<std::array<double, 6>> &rest_positions,
		std::vector<std::array<double, 6>> &displacements,
		std::function<double(const std::array<double, 6> &)> compute_energy,
		std::function<void(const std::array<double, 6> &, Eigen::Vector2d &)> compute_jacobian,
		std::function<void(const std::array<double, 6> &, Eigen::Matrix2d &)> compute_hessian)
	{
		assert(!rest_positions.empty());
		assert(rest_positions.size() == displacements.size());
		auto &T0 = rest_positions.front();
		Eigen::Vector2d old_pos(T0[0], T0[1]);

		auto energy_from_point = [&](const Eigen::Vector2d &pos) -> double {
			auto total_energy = 0.;
			for (int i = 0; i < rest_positions.size(); i++)
			{
				auto &T = rest_positions[i];
				std::array<double, 6> T_deformed;

				for (auto j = 0; j < 2; j++)
				{
					T[j] = pos[j]; // only filling the front point x,y,z.
				}
				for (int j = 0; j < 6; j++)
				{
					T_deformed[j] = T[j] + displacements[i][j];
				}
				total_energy += compute_energy(T_deformed);
			}
			return total_energy;
		};

		// TODO: These three functions should not be in global namespace
		auto newton_direction_2d = [&](auto &compute_energy,
									   auto &compute_jacobian,
									   auto &compute_hessian,
									   auto &assembles,
									   const Eigen::Vector2d &pos) -> Eigen::Vector2d {
			auto total_energy = 0.;
			Eigen::Vector2d total_jac = Eigen::Vector2d::Zero();
			Eigen::Matrix2d total_hess = Eigen::Matrix2d::Zero();

			// E = \sum_i E_i(x)
			// J = \sum_i J_i(x)
			// H = \sum_i H_i(x)
			auto local_id = 0;
			for (auto &T : assembles)
			{
				for (auto j = 0; j < 2; j++)
				{
					T[j] = pos[j]; // only filling the front point.
				}
				auto jac = decltype(total_jac)();
				auto hess = decltype(total_hess)();
				total_energy += compute_energy(T);
				compute_jacobian(T, jac);
				compute_hessian(T, hess);
				total_jac += jac;
				total_hess += hess;
				assert(!std::isnan(total_energy));
			}
			Eigen::Vector2d x = total_hess.ldlt().solve(total_jac);
			wmtk::logger().info("energy {}", total_energy);
			if (total_jac.isApprox(total_hess * x)) // a hacky PSD trick. TODO: change this.
				return -x;
			else
			{
				wmtk::logger().info("gradient descent instead.");
				return -total_jac;
			}
		};

		auto linesearch_2d = [](auto &&energy_from_point,
								const Eigen::Vector2d &pos,
								const Eigen::Vector2d &dir,
								const int &max_iter) {
			auto lr = 0.5;
			auto old_energy = energy_from_point(pos);
			wmtk::logger().info("old energy {} dir {}", old_energy, dir.transpose());
			for (auto iter = 1; iter <= max_iter; iter++)
			{
				Eigen::Vector2d newpos = pos + std::pow(lr, iter) * dir;
				wmtk::logger().info("pos {}, dir {}, [{}]", pos.transpose(), dir.transpose(), std::pow(lr, iter));
				auto new_energy = energy_from_point(newpos);
				wmtk::logger().info("iter {}, E= {}, [{}]", iter, new_energy, newpos.transpose());
				if (new_energy < old_energy)
					return newpos; // TODO: armijo conditions.
			}
			return pos;
		};

		auto compute_new_valid_pos = [&](const Eigen::Vector2d &pos) {
			auto current_pos = pos;
			auto line_search_iters = 12;
			auto newton_iters = 10;
			for (auto iter = 0; iter < newton_iters; iter++)
			{
				std::vector<std::array<double, 6>> assembles(rest_positions.size());
				for (int i = 0; i < assembles.size(); i++)
				{
					for (int j = 0; j < 6; j++)
					{
						assembles[i][j] = rest_positions[i][j] + displacements[i][j];
					}
				}
				auto dir = newton_direction_2d(
					compute_energy,
					compute_jacobian,
					compute_hessian,
					assembles,
					current_pos + Eigen::Vector2d(displacements[0][0], displacements[0][1]));
				auto newpos = linesearch_2d(energy_from_point, current_pos, dir, line_search_iters);
				if ((newpos - current_pos).norm() < 1e-9) // barely moves
				{
					break;
				}
				current_pos = newpos;
			}
			return current_pos;
		};
		return compute_new_valid_pos(old_pos);
	}

} // namespace polyfem::mesh