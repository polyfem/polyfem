#include <polyfem/mesh/remesh/WildTriRemesher.hpp>
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

	bool WildTriRemesher::smooth_before(const Tuple &t)
	{
		if (vertex_attrs[t.vid(*this)].fixed)
			return false;

		cache_before(); // Cache global quantities for projection

		return true;
	}

	bool WildTriRemesher::smooth_after(const Tuple &t)
	{
		const size_t vid = t.vid(*this);

		const std::vector<Tuple> one_ring = get_one_ring_tris_for_vertex(t);
		assert(one_ring.size() > 0);

		// ---------------------------------------------------------------------
		// 1. update rest position of new vertex

		// TODO: use same energy as we simulate
		std::vector<std::shared_ptr<Form>> forms;
		for (const Tuple &loc : one_ring)
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
		// TODO: This should be dense
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

		vertex_attrs[vid].rest_position = new_rest_pos;

		if (!invariants(one_ring))
		{
			assert(false); // this should be satisfied because of the line-search
			return false;
		}

		// ---------------------------------------------------------------------
		// 2. project quantities so to minimize the L2 error
		project_quantities();

		// There is no non-inversion check in project_quantities, so check it here.
		if (!invariants(one_ring))
			return false;

		// ---------------------------------------------------------------------
		// 3. perform a local relaxation of the n-ring to get an estimate of the
		//    energy decrease.
		return local_relaxation(t);
	}

} // namespace polyfem::mesh