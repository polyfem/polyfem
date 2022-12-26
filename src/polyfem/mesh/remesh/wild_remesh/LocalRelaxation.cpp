#include <polyfem/mesh/remesh/WildRemesher.hpp>

#include <polyfem/mesh/remesh/wild_remesh/LocalMesh.hpp>
#include <polyfem/mesh/remesh/L2Projection.hpp>
#include <polyfem/solver/ALSolver.hpp>
#include <polyfem/solver/forms/ContactForm.hpp>
#include <polyfem/solver/forms/FrictionForm.hpp>
#include <polyfem/solver/forms/LinearForm.hpp>
#include <polyfem/time_integrator/ImplicitTimeIntegrator.hpp>

#define POLYFEM_REMESH_USE_FRICTION_FORM

namespace polyfem::mesh
{
	template <class WMTKMesh>
	bool WildRemesher<WMTKMesh>::local_relaxation(const Tuple &t, const int n_ring)
	{
		using namespace polyfem::solver;
		using namespace polyfem::time_integrator;

		constexpr bool free_boundary = true;

		// 1. Get the n-ring of elements around the vertex.
		using This = typename std::remove_pointer<decltype(this)>::type;
		// LocalMesh local_mesh = LocalMesh::n_ring(
		// 	*this, t, n_ring, /*include_global_boundary=*/free_boundary);
		LocalMesh<This> local_mesh = LocalMesh<This>::flood_fill_n_ring(
			*this, t, flood_fill_rel_area * total_volume, /*include_global_boundary=*/free_boundary);
		// LocalMesh<This> local_mesh = LocalMesh<This>::ball_selection(
		// 	*this, vertex_attrs[t.vid(*this)].rest_position,
		// 	flood_fill_rel_area, /*include_global_boundary=*/free_boundary);
		// LocalMesh local_mesh(*this, get_faces(), /*include_global_boundary=*/free_boundary);

		std::vector<polyfem::basis::ElementBases> bases;
		int n_bases;
		{
			POLYFEM_SCOPED_TIMER(timings.build_bases);
			Eigen::VectorXi vertex_to_basis;
			n_bases = Remesher::build_bases(
				local_mesh.rest_positions(), local_mesh.elements(),
				state.formulation(), bases, vertex_to_basis);

			assert(n_bases == local_mesh.num_local_vertices());
			n_bases = local_mesh.num_vertices();
			assert(vertex_to_basis.size() == n_bases);

			const int start_i = local_mesh.num_local_vertices();
			// set tail to range [start_i, n_bases)
			std::iota(vertex_to_basis.begin() + start_i, vertex_to_basis.end(), start_i);

#ifndef NDEBUG
			for (const int basis_id : vertex_to_basis)
				assert(basis_id >= 0);
#endif

			local_mesh.reorder_vertices(vertex_to_basis);
		}
		const int ndof = n_bases * dim();

		// --------------------------------------------------------------------

		std::vector<int> boundary_nodes;
		{
			POLYFEM_SCOPED_TIMER(timings.create_boundary_nodes);

			const auto add_vertex_to_boundary_nodes = [&](const int vi) {
				for (int d = 0; d < dim(); ++d)
					boundary_nodes.push_back(dim() * vi + d);
			};

			for (const int vi : local_mesh.fixed_vertices())
				add_vertex_to_boundary_nodes(vi);

			if (free_boundary)
			{
				const Eigen::MatrixXi &BF = local_mesh.boundary_facets();
				for (int i = 0; i < BF.rows(); i++)
				{
					const int boundary_id = local_mesh.boundary_ids()[i];

					// TODO: handle the correct DBC
					if (boundary_id != 2 && boundary_id != 4)
						continue;

					for (int j = 0; j < BF.cols(); ++j)
						add_vertex_to_boundary_nodes(BF(i, j));
				}
			}

			// Sort and remove the duplicate boundary_nodes.
			std::sort(boundary_nodes.begin(), boundary_nodes.end());
			auto new_end = std::unique(boundary_nodes.begin(), boundary_nodes.end());
			boundary_nodes.erase(new_end, boundary_nodes.end());
		}

		timings.total_ndofs += ndof - boundary_nodes.size();
		timings.n_solves++;

		const Eigen::MatrixXd target_x = utils::flatten(local_mesh.displacements());

		// --------------------------------------------------------------------

		// 2. Perform "relaxation" by minimizing the elastic energy of the n-ring
		// with the internal boundary edges fixed.

		assembler::AssemblerUtils assembler = create_assembler(local_mesh.body_ids());
		assembler::AssemblyValsCache ass_vals_cache;

		Eigen::SparseMatrix<double> mass;
		{
			POLYFEM_SCOPED_TIMER(timings.assemble_mass_matrix);
			ass_vals_cache.init(is_volume(), bases, /*gbases=*/bases, /*is_mass=*/true);
			assembler.assemble_mass_matrix(
				/*assembler_formulation=*/"", is_volume(), n_bases,
				/*use_density=*/true, bases, /*gbases=*/bases, ass_vals_cache, mass);
			// Set the mass of the codimensional fixed vertices to the average mass.
			const int local_ndof = dim() * local_mesh.num_local_vertices();
			for (int i = local_ndof; i < ndof; ++i)
				mass.coeffRef(i, i) = state.avg_mass;
		}
		ass_vals_cache.init(is_volume(), bases, /*gbases=*/bases, /*is_mass=*/false);

		ipc::CollisionMesh collision_mesh; // This has to stay alive
		const bool contact_enabled = state.args["contact"]["enabled"] && free_boundary;
		if (contact_enabled)
		{
			POLYFEM_SCOPED_TIMER(timings.create_collision_mesh);
			collision_mesh = ipc::CollisionMesh::build_from_full_mesh(
				local_mesh.rest_positions(), local_mesh.boundary_edges(),
				local_mesh.boundary_faces());
		}

		SolveData solve_data;

		// Initialize time integrator
		if (state.problem->is_time_dependent())
		{
			solve_data.time_integrator =
				ImplicitTimeIntegrator::construct_time_integrator(state.args["time"]["integrator"]);
			std::vector<Eigen::VectorXd> x_prevs;
			std::vector<Eigen::VectorXd> v_prevs;
			std::vector<Eigen::VectorXd> a_prevs;
			const auto &project_quantities = local_mesh.projection_quantities();
			split_time_integrator_quantities(
				// Drop the last column of projection_quantities, which is the friction gradient.
				project_quantities.leftCols(project_quantities.cols() - 1),
				dim(), x_prevs, v_prevs, a_prevs);
			solve_data.time_integrator->init(
				x_prevs, v_prevs, a_prevs, state.args["time"]["dt"]);
		}

		// TODO: initialize solve_data.rhs_assembler

		const std::vector<mesh::LocalBoundary> local_boundary;
		const std::vector<mesh::LocalBoundary> local_neumann_boundary;
		const auto rhs = Eigen::MatrixXd::Zero(target_x.rows(), target_x.cols());

		std::vector<std::shared_ptr<Form>> forms;
		{
			POLYFEM_SCOPED_TIMER(timings.init_forms);
			forms = solve_data.init_forms(
				// General
				dim(), current_time,
				// Elastic form
				n_bases, bases, /*geom_bases=*/bases, assembler, ass_vals_cache, state.formulation(),
				// Body form
				/*n_pressure_bases=*/0, boundary_nodes, local_boundary, local_neumann_boundary,
				state.n_boundary_samples(), rhs, /*sol=*/target_x,
				// Inertia form
				state.args["solver"]["ignore_inertia"], mass,
				// Lagged regularization form
				state.args["solver"]["advanced"]["lagged_regularization_weight"],
				state.args["solver"]["advanced"]["lagged_regularization_iterations"],
				// Augmented lagrangian form
				Obstacle(),
				// Contact form
				contact_enabled,
				collision_mesh,
				state.args["contact"]["dhat"],
				state.avg_mass,
				state.args["contact"]["use_convergent_formulation"],
				state.args["solver"]["contact"]["barrier_stiffness"],
				state.args["solver"]["contact"]["CCD"]["broad_phase"],
				state.args["solver"]["contact"]["CCD"]["tolerance"],
				state.args["solver"]["contact"]["CCD"]["max_iterations"],
				// Friction form
				state.args["contact"]["friction_coefficient"],
				state.args["contact"]["epsv"],
				state.args["solver"]["contact"]["friction_iterations"],
				// Rayleigh damping form
				state.args["solver"]["rayleigh_damping"]);

			// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

			assert(solve_data.al_form == nullptr);
			solve_data.al_form = std::make_shared<ALForm>(ndof, boundary_nodes, mass, Obstacle(), target_x);
			forms.push_back(solve_data.al_form);
			assert(state.solve_data.al_form != nullptr);
			solve_data.al_form->set_weight(state.solve_data.al_form->weight());

			if (solve_data.contact_form)
			{
				assert(state.solve_data.contact_form != nullptr);
				solve_data.contact_form->set_weight(state.solve_data.contact_form->weight());
			}

#ifndef POLYFEM_REMESH_USE_FRICTION_FORM
			if (solve_data.friction_form)
			{
				// add linear form ∇D(x₀)ᵀ x
				solve_data.friction_form->disable();
				forms.push_back(std::make_shared<LinearForm>(
					utils::flatten(utils::reorder_matrix(local_mesh.friction_gradient(), vertex_to_basis))));
			}
#endif
		}

		// --------------------------------------------------------------------

		Eigen::MatrixXd sol = target_x;
		{
			POLYFEM_SCOPED_TIMER(timings.local_relaxation_solve);
			solve_data.nl_problem = std::make_shared<StaticBoundaryNLProblem>(
				ndof, boundary_nodes, target_x, forms);

			// Create augmented Lagrangian solver
			ALSolver al_solver(
				state.make_nl_solver<NLProblem>(), solve_data.al_form,
				state.args["solver"]["augmented_lagrangian"]["initial_weight"],
				state.args["solver"]["augmented_lagrangian"]["scaling"],
				state.args["solver"]["augmented_lagrangian"]["max_steps"],
				/*update_barrier_stiffness=*/[](const Eigen::VectorXd &) {});

			assert(solve_data.time_integrator != nullptr);
			solve_data.nl_problem->init(sol);
			solve_data.nl_problem->init_lagging(solve_data.time_integrator->x_prev());
			solve_data.nl_problem->update_lagging(sol, /*iter_num=*/0);

			const auto level_before = logger().level();
			logger().set_level(spdlog::level::warn);
			al_solver.solve(*(solve_data.nl_problem), sol, state.args["solver"]["augmented_lagrangian"]["force"]);
			logger().set_level(level_before);
		}
		// --------------------------------------------------------------------

		// 3. Determine if we should accept the operation based on a decrease in energy.
		bool accept;
		{
			POLYFEM_SCOPED_TIMER(timings.acceptance_check);

			const double local_energy_before = solve_data.nl_problem->value(target_x);
			const double local_energy_after = solve_data.nl_problem->value(sol);

			assert(std::isfinite(local_energy_before));
			assert(std::isfinite(local_energy_after));

			const double abs_diff = local_energy_before - local_energy_after; // > 0 if energy decreased
			// TODO: compute global_energy_before
			// Right now using: starting_energy = state.solve_data.nl_problem->value(sol)
			const double global_energy_before = abs(starting_energy);
			const double rel_diff = abs_diff / global_energy_before;

			accept = rel_diff >= energy_relative_tolerance && abs_diff >= energy_absolute_tolerance;

			static int log_i = 0;
			logger().log(
				accept ? spdlog::level::critical : spdlog::level::debug,
				"{} {:g} | rel_diff={:g} rel_tol={:g} | abs_diff={:g} abs_tol={:g}",
				log_i++, starting_energy, rel_diff, energy_relative_tolerance, abs_diff, energy_absolute_tolerance);
		}

		// Update positions only on acceptance
		if (accept)
		{
			static int save_i = 0;
			local_mesh.write_mesh(state.resolve_output_path(fmt::format("local_mesh_{:04d}.vtu", save_i)), target_x);
			write_deformed_mesh(state.resolve_output_path(fmt::format("split_{:04d}.vtu", save_i++)));

#ifndef POLYFEM_REMESH_USE_FRICTION_FORM
			Eigen::VectorXd friction_gradient = Eigen::VectorXd::Zero(target_x.rows());
			if (solve_data.friction_form)
			{
				POLYFEM_SCOPED_TIMER(timings.local_relaxation_solve);
				forms.back()->disable(); // disable linear form ∇D(x₀)ᵀ x
				solve_data.friction_form->enable();

				// discard the solution from the solve with the linear form
				sol = target_x;

				solve_data.nl_problem = std::make_shared<StaticBoundaryNLProblem>(
					ndof, boundary_nodes, target_x, forms);

				assert(solve_data.nl_problem->uses_lagging());
				assert(solve_data.time_integrator != nullptr);
				solve_data.nl_problem->init_lagging(solve_data.time_integrator->x_prev());
				solve_data.nl_problem->update_lagging(sol, /*iter_num=*/0);

				// Create augmented Lagrangian solver
				ALSolver al_solver(
					state.make_nl_solver<NLProblem>(), solve_data.al_form,
					state.args["solver"]["augmented_lagrangian"]["initial_weight"],
					state.args["solver"]["augmented_lagrangian"]["scaling"],
					state.args["solver"]["augmented_lagrangian"]["max_steps"],
					/*update_barrier_stiffness=*/[](const Eigen::VectorXd &) {});

				const auto level_before = logger().level();
				logger().set_level(spdlog::level::warn);
				al_solver.solve(*(solve_data.nl_problem), sol, state.args["solver"]["augmented_lagrangian"]["force"]);
				logger().set_level(level_before);

				// Update the lagging to get a more accurate friction gradient
				solve_data.nl_problem->update_lagging(sol, /*iter_num=*/0);
				solve_data.friction_form->first_derivative(sol, friction_gradient);
			}
#endif

			for (const auto &[glob_vi, loc_vi] : local_mesh.global_to_local())
			{
				const auto u = sol.middleRows(dim() * loc_vi, dim());
				const auto u_old = vertex_attrs[glob_vi].displacement();
				vertex_attrs[glob_vi].position = vertex_attrs[glob_vi].rest_position + u;

#ifndef POLYFEM_REMESH_USE_FRICTION_FORM
				const auto f = friction_gradient.segment(dim() * loc_vi, dim());
				const auto f_old = vertex_attrs[glob_vi].projection_quantities.rightCols(1);
				if (f_old.norm() != 0)
					logger().critical(
						"(f_old⋅f)/(‖f_old‖‖f‖)={:g} ‖f_old‖/‖f‖={:g} ‖u_old - u‖={:g}",
						(f_old / f_old.norm()).dot(f / f.norm()), f.norm() / f_old.norm(), (u_old - u).norm());
				vertex_attrs[glob_vi].projection_quantities.rightCols(1) = f;
#endif
			}

#ifndef POLYFEM_REMESH_USE_FRICTION_FORM
			m_obstacle_vals.rightCols(1) = friction_gradient.tail(m_obstacle_vals.rows());
#endif

			local_mesh.write_mesh(state.resolve_output_path(fmt::format("local_mesh_{:04d}.vtu", save_i)), sol);
			write_deformed_mesh(state.resolve_output_path(fmt::format("split_{:04d}.vtu", save_i++)));
		}

		return accept;
	}

	// ----------------------------------------------------------------------------------------------
	// Template specializations
	template class WildRemesher<wmtk::TriMesh>;
	template class WildRemesher<wmtk::TetMesh>;
} // namespace polyfem::mesh