#include <polyfem/mesh/remesh/PhysicsRemesher.hpp>
#include <polyfem/mesh/remesh/wild_remesh/LocalRelaxationData.hpp>
#include <polyfem/solver/forms/BodyForm.hpp>
#include <polyfem/solver/NLProblem.hpp>
#include <polyfem/time_integrator/ImplicitTimeIntegrator.hpp>

#include <polysolve/nonlinear/Solver.hpp>

namespace polyfem::mesh
{
	void add_solver_timings(
		decltype(Remesher::timings) &timings,
		const polyfem::json &solver_info)
	{
		// Copy over timing data
		const int solver_iters = solver_info["iterations"];
		for (const auto &[key, value] : solver_info.items())
		{
			if (utils::StringUtils::startswith(key, "time_")
				&& key != "total_time"
				&& key != "time_line_search")
			{
				// The solver reports the time per iteration, so we need to
				// multiply by the number of iterations.
				const std::string new_key = "NonlinearSolver::" + key.substr(5, key.size() - 5);
				timings[new_key] += solver_iters * value.get<double>();
			}
		}
	}

	template <class WMTKMesh>
	bool PhysicsRemesher<WMTKMesh>::local_relaxation(
		const std::vector<Tuple> &local_mesh_tuples,
		const double acceptance_tolerance)
	{
		// --------------------------------------------------------------------
		// 1. Get the n-ring of elements around the vertex.

		const bool include_global_boundary =
			state.is_contact_enabled() && std::any_of(local_mesh_tuples.begin(), local_mesh_tuples.end(), [&](const Tuple &t) {
				const size_t tid = this->element_id(t);
				for (int i = 0; i < Super::FACETS_PER_ELEMENT; ++i)
					if (this->is_boundary_facet(this->tuple_from_facet(tid, i)))
						return true;
				return false;
			});

		LocalMesh<Super> local_mesh(*this, local_mesh_tuples, include_global_boundary);

		// --------------------------------------------------------------------
		// 2. Perform "relaxation" by minimizing the elastic energy of the
		// n-ring with the internal boundary edges fixed.

		LocalRelaxationData data(this->state, local_mesh, this->current_time, include_global_boundary);
		solver::SolveData &solve_data = data.solve_data;

		const int n_free_dof = data.n_free_dof();
		if (n_free_dof <= 0)
		{
			assert(n_free_dof == 0);
			return false;
		}

		this->total_ndofs += n_free_dof;
		this->num_solves++;

		// Nonlinear solver
		auto nl_solver = state.make_nl_solver(/*for_al=*/false); // TODO: Use Eigen::LLT
		nl_solver->stop_criteria().iterations = args["local_relaxation"]["max_nl_iterations"];
		if (this->is_boundary_op())
			nl_solver->stop_criteria().iterations = std::max(nl_solver->stop_criteria().iterations, size_t(5));
		nl_solver->allow_out_of_iterations = true;

		Eigen::VectorXd reduced_sol = solve_data.nl_problem->full_to_reduced(data.sol());

		const auto level_before = logger().level();
		logger().set_level(spdlog::level::warn);
		try
		{
			POLYFEM_REMESHER_SCOPED_TIMER("Local relaxation solve");
			nl_solver->minimize(*(solve_data.nl_problem), reduced_sol);
		}
		catch (const std::runtime_error &e)
		{
			logger().set_level(level_before);
			assert(false);
			return false;
		}
		logger().set_level(level_before);

		// Copy over timing data
		add_solver_timings(this->timings, nl_solver->info());

		Eigen::VectorXd sol = solve_data.nl_problem->reduced_to_full(reduced_sol);

		// --------------------------------------------------------------------
		// 3. Determine if we should accept the operation based on a decrease in
		// energy.

		const double local_energy_after = solve_data.nl_problem->value(sol);
		assert(std::isfinite(local_energy_before()));
		assert(std::isfinite(local_energy_after));
		const double abs_diff = local_energy_before() - local_energy_after; // > 0 if energy decreased
		// TODO: compute global_energy_before
		// Right now using: starting_energy = state.solve_data.nl_problem->value(sol)
		// const double global_energy_before = abs(starting_energy);
		// const double rel_diff = abs_diff / global_energy_before;

		// NOTE: only use abs_diff
		// accept = rel_diff >= energy_relative_tolerance && abs_diff >= energy_absolute_tolerance;
		// NOTE: account for Δt² in energy by multiplying acceptance tol by Δt²
		const double dt_sqr = solve_data.time_integrator ? solve_data.time_integrator->acceleration_scaling() : 1.0;
		const bool accept = abs_diff >= dt_sqr * acceptance_tolerance;

		// Update positions only on acceptance
		if (accept)
		{
			static int save_i = 0;
			// local_mesh.write_mesh(state.resolve_output_path(fmt::format("local_mesh_{:04d}.vtu", save_i)), target_x);
			// write_mesh(state.resolve_output_path(fmt::format("relaxation_{:04d}.vtu", save_i++)));

			// Re-solve with more iterations
			if (!is_converged_status(nl_solver->status()))
			{
				nl_solver->stop_criteria().iterations = 100;

				const auto level_before = logger().level();
				logger().set_level(spdlog::level::warn);
				try
				{
					POLYFEM_REMESHER_SCOPED_TIMER("Local relaxation resolve");
					nl_solver->minimize(*(solve_data.nl_problem), reduced_sol);
				}
				catch (const std::runtime_error &e)
				{
					logger().set_level(level_before);
					assert(false);
					return false;
				}
				logger().set_level(level_before);

				// Copy over timing data
				add_solver_timings(this->timings, nl_solver->info());

				sol = solve_data.nl_problem->reduced_to_full(reduced_sol);
			}

			for (const auto &[glob_vi, loc_vi] : local_mesh.global_to_local())
			{
				const auto u = sol.middleRows(this->dim() * loc_vi, this->dim());
				const auto u_old = vertex_attrs[glob_vi].displacement();
				vertex_attrs[glob_vi].position = vertex_attrs[glob_vi].rest_position + u;
			}

			// local_mesh.write_mesh(state.resolve_output_path(fmt::format("local_mesh_{:04d}.vtu", save_i)), sol);
			// write_mesh(state.resolve_output_path(fmt::format("relaxation_{:04d}.vtu", save_i++)));
		}

		static const std::string accept_str =
			fmt::format(fmt::fg(fmt::terminal_color::green), "accept");
		static const std::string reject_str =
			fmt::format(fmt::fg(fmt::terminal_color::yellow), "reject");
		logger().debug(
			"[{:s}] E0={:<10g} E1={:<10g} (E0-E1)={:<11g} tol={:g} local_ndof={:d} n_iters={:d}",
			accept ? accept_str : reject_str, local_energy_before(),
			local_energy_after, abs_diff, dt_sqr * acceptance_tolerance,
			n_free_dof, nl_solver->current_criteria().iterations);

		return accept;
	}

	// ----------------------------------------------------------------------------------------------
	// Template specializations
	template class PhysicsRemesher<wmtk::TriMesh>;
	template class PhysicsRemesher<wmtk::TetMesh>;
} // namespace polyfem::mesh