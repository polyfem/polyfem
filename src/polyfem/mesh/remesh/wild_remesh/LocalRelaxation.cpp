#include <polyfem/mesh/remesh/PhysicsRemesher.hpp>

#include <polyfem/assembler/Mass.hpp>
#include <polyfem/solver/ALSolver.hpp>
#include <polyfem/solver/problems/StaticBoundaryNLProblem.hpp>
#include <polyfem/solver/forms/ContactForm.hpp>
#include <polyfem/solver/forms/LinearForm.hpp>
#include <polyfem/time_integrator/ImplicitTimeIntegrator.hpp>

namespace polyfem::mesh
{
	template <class WMTKMesh>
	bool PhysicsRemesher<WMTKMesh>::local_relaxation(
		const Tuple &t,
		const double acceptance_tolerance)
	{
		using namespace polyfem::solver;
		using namespace polyfem::basis;
		using namespace polyfem::time_integrator;

		// --------------------------------------------------------------------
		// 1. Get the n-ring of elements around the vertex.

		std::vector<Tuple> local_mesh_tuples = this->local_mesh_tuples(t);

		const bool include_global_boundary =
			state.is_contact_enabled() && std::any_of(local_mesh_tuples.begin(), local_mesh_tuples.end(), [&](const Tuple &t) {
				const size_t tid = this->element_id(t);
				for (int i = 0; i < Super::FACETS_PER_ELEMENT; ++i)
					if (this->is_boundary_facet(this->tuple_from_facet(tid, i)))
						return true;
				return false;
			});

		LocalMesh<PhysicsRemesher<WMTKMesh>> local_mesh(
			*this, local_mesh_tuples, include_global_boundary);

		const int n_bases = local_mesh.num_vertices();
		const int ndof = n_bases * this->dim();

		// --------------------------------------------------------------------
		// 2. Perform "relaxation" by minimizing the elastic energy of the
		// n-ring with the internal boundary edges fixed.

		const std::vector<ElementBases> bases = local_mesh.build_bases(state.formulation());
		const std::vector<int> boundary_nodes = local_boundary_nodes(local_mesh);

		assert(ndof >= boundary_nodes.size());
		if (ndof - boundary_nodes.size() == 0)
			return false;

		this->total_ndofs += ndof - boundary_nodes.size();
		this->num_solves++;

		// These have to stay alive
		this->init_assembler(local_mesh.body_ids());
		SolveData solve_data;
		assembler::AssemblyValsCache ass_vals_cache;
		Eigen::SparseMatrix<double> mass;
		ipc::CollisionMesh collision_mesh;

		local_solve_data(
			local_mesh, bases, boundary_nodes, *this->assembler, *this->mass_matrix_assembler,
			include_global_boundary, solve_data, ass_vals_cache, mass, collision_mesh);

		const Eigen::MatrixXd target_x = utils::flatten(local_mesh.displacements());
		Eigen::MatrixXd sol = target_x;

		// Nonlinear solver
		auto nl_solver = state.template make_nl_solver<NLProblem>("Eigen::LLT");
		auto criteria = nl_solver->getStopCriteria();
		criteria.iterations = args["local_relaxation"]["max_nl_iterations"];
		if (this->is_boundary_op())
			criteria.iterations = std::max(criteria.iterations, 5ul);
		nl_solver->setStopCriteria(criteria);

		// Create augmented Lagrangian solver
		ALSolver al_solver(
			nl_solver, solve_data.al_form,
			state.args["solver"]["augmented_lagrangian"]["initial_weight"],
			state.args["solver"]["augmented_lagrangian"]["scaling"],
			state.args["solver"]["augmented_lagrangian"]["max_steps"],
			/*update_barrier_stiffness=*/[](const Eigen::VectorXd &) {});

		const auto level_before = logger().level();
		logger().set_level(spdlog::level::warn);
		try
		{
			POLYFEM_REMESHER_SCOPED_TIMER("Local relaxation solve");
			al_solver.solve(*(solve_data.nl_problem), sol, state.args["solver"]["augmented_lagrangian"]["force"]);
		}
		catch (const std::runtime_error &e)
		{
			assert(false);
			return false;
		}
		logger().set_level(level_before);

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

		// TODO: only use abs_diff
		// accept = rel_diff >= energy_relative_tolerance && abs_diff >= energy_absolute_tolerance;
		const bool accept = abs_diff >= acceptance_tolerance;

		// Update positions only on acceptance
		if (accept)
		{
			static int save_i = 0;
			// local_mesh.write_mesh(state.resolve_output_path(fmt::format("local_mesh_{:04d}.vtu", save_i)), target_x);
			// write_mesh(state.resolve_output_path(fmt::format("relaxation_{:04d}.vtu", save_i++)));

			// Re-solve with more iterations
			if (!nl_solver->converged())
			{
				auto criteria = nl_solver->getStopCriteria();
				criteria.iterations = 100;
				nl_solver->setStopCriteria(criteria);

				const auto level_before = logger().level();
				logger().set_level(spdlog::level::warn);
				try
				{
					POLYFEM_REMESHER_SCOPED_TIMER("Local relaxation solve");
					al_solver.solve(
						*(solve_data.nl_problem), sol,
						state.args["solver"]["augmented_lagrangian"]["force"]);
				}
				catch (const std::runtime_error &e)
				{
					assert(false);
					return false;
				}
				logger().set_level(level_before);
			}

			for (const auto &[glob_vi, loc_vi] : local_mesh.global_to_local())
			{
				const auto u = sol.middleRows(this->dim() * loc_vi, this->dim());
				const auto u_old = vertex_attrs[glob_vi].displacement();
				vertex_attrs[glob_vi].position = vertex_attrs[glob_vi].rest_position + u;
			}

			// local_mesh.write_mesh(state.resolve_output_path(fmt::format("local_mesh_{:04d}.vtu", save_i)), sol);
			// write_mesh(state.resolve_output_path(fmt::format("relaxation_{:04d}.vtu", save_i++)));

			// Increase the hash of the triangles that have been modified
			// to invalidate all tuples that point to them.
			this->extend_local_patch(local_mesh_tuples);
			for (Tuple &t : local_mesh_tuples)
			{
				assert(t.is_valid(*this));
				if constexpr (std::is_same_v<wmtk::TriMesh, WMTKMesh>)
					this->m_tri_connectivity[t.fid(*this)].hash++;
				else
					this->m_tet_connectivity[t.tid(*this)].hash++;
				assert(!t.is_valid(*this));
				t.update_hash(*this);
				assert(t.is_valid(*this));
			}
		}

		static const std::string accept_str =
			fmt::format(fmt::fg(fmt::terminal_color::green), "accept");
		static const std::string reject_str =
			fmt::format(fmt::fg(fmt::terminal_color::yellow), "reject");
		logger().debug(
			"[{:s}] E0={:<10g} E1={:<10g} (E1-E0)={:<10g} tol={:g} local_ndof={:d} n_iters={:d}",
			accept ? accept_str : reject_str, local_energy_before(),
			local_energy_after, abs_diff, acceptance_tolerance,
			ndof - boundary_nodes.size(), nl_solver->criteria().iterations);

		return accept;
	}

	template <class WMTKMesh>
	std::vector<int> PhysicsRemesher<WMTKMesh>::local_boundary_nodes(
		const LocalMesh<This> &local_mesh) const
	{
		POLYFEM_REMESHER_SCOPED_TIMER("Create boundary nodes");

		std::vector<int> boundary_nodes;

		const auto add_vertex_to_boundary_nodes = [&](const int vi) {
			for (int d = 0; d < this->dim(); ++d)
				boundary_nodes.push_back(this->dim() * vi + d);
		};

		for (const int vi : local_mesh.fixed_vertices())
			add_vertex_to_boundary_nodes(vi);

		// TODO: get this from state rather than building it
		assert(state.args["boundary_conditions"]["dirichlet_boundary"].is_array());
		const std::vector<json> bcs = state.args["boundary_conditions"]["dirichlet_boundary"];
		std::unordered_set<int> bc_ids;
		for (const json &bc : bcs)
		{
			if (!bc.is_object())
				log_and_throw_error("Boundary condition is not an object");
			assert(bc.contains("id") && bc["id"].is_number_integer());
			bc_ids.insert(bc["id"].get<int>());

#ifndef NDEBUG
			// Only all dimensions constrained are supported right now.
			const std::vector<bool> bc_dim = bc["dimension"];
			assert(std::all_of(bc_dim.begin(), bc_dim.end(), [](const bool b) { return b; }));
#endif
		}

		const Eigen::MatrixXi &BF = local_mesh.boundary_facets();
		for (int i = 0; i < BF.rows(); i++)
		{
			const int boundary_id = local_mesh.boundary_ids()[i];

			if (bc_ids.find(boundary_id) == bc_ids.end())
				continue;

			for (int j = 0; j < BF.cols(); ++j)
				add_vertex_to_boundary_nodes(BF(i, j));
		}

		// Sort and remove the duplicate boundary_nodes.
		std::sort(boundary_nodes.begin(), boundary_nodes.end());
		auto new_end = std::unique(boundary_nodes.begin(), boundary_nodes.end());
		boundary_nodes.erase(new_end, boundary_nodes.end());

		return boundary_nodes;
	}

	template <class WMTKMesh>
	void PhysicsRemesher<WMTKMesh>::local_solve_data(
		const LocalMesh<This> &local_mesh,
		const std::vector<polyfem::basis::ElementBases> &bases,
		const std::vector<int> &boundary_nodes,
		const assembler::Assembler &assembler,
		const assembler::Mass &mass_matrix_assembler,
		const bool contact_enabled,
		solver::SolveData &solve_data,
		assembler::AssemblyValsCache &ass_vals_cache,
		Eigen::SparseMatrix<double> &mass,
		ipc::CollisionMesh &collision_mesh) const
	{
		using namespace polyfem::solver;
		using namespace polyfem::time_integrator;

		const int n_bases = local_mesh.num_vertices();
		const int ndof = n_bases * this->dim();

		// Current solution.
		const Eigen::MatrixXd target_x = utils::flatten(local_mesh.displacements());

		// Assemble the mass matrix.
		{
			POLYFEM_REMESHER_SCOPED_TIMER("Assemble mass matrix");
			ass_vals_cache.init(this->is_volume(), bases, /*gbases=*/bases, /*is_mass=*/true);
			mass_matrix_assembler.assemble(
				this->is_volume(), n_bases, bases, /*gbases=*/bases,
				ass_vals_cache, mass, /*is_mass=*/true);
			// Set the mass of the codimensional fixed vertices to the average mass.
			const int local_ndof = this->dim() * local_mesh.num_local_vertices();
			for (int i = local_ndof; i < ndof; ++i)
				mass.coeffRef(i, i) = state.avg_mass;
		}
		// Assemble the stiffness matrix.
		ass_vals_cache.init(this->is_volume(), bases, /*gbases=*/bases, /*is_mass=*/false);

		// Create collision mesh.
		if (contact_enabled)
		{
			POLYFEM_REMESHER_SCOPED_TIMER("Create collision mesh");
			collision_mesh = ipc::CollisionMesh::build_from_full_mesh(
				local_mesh.rest_positions(), local_mesh.boundary_edges(),
				local_mesh.boundary_faces());

			// Ignore all collisions between fixed elements.
			std::vector<bool> is_vertex_fixed(local_mesh.num_vertices(), false);
			for (const int vi : local_mesh.fixed_vertices())
				is_vertex_fixed[vi] = true;
			collision_mesh.can_collide = [is_vertex_fixed, &collision_mesh](size_t vi, size_t vj) {
				return !is_vertex_fixed[collision_mesh.to_full_vertex_id(vi)]
					   || !is_vertex_fixed[collision_mesh.to_full_vertex_id(vj)];
			};
		}

		// Initialize time integrator
		if (state.problem->is_time_dependent())
		{
			solve_data.time_integrator =
				ImplicitTimeIntegrator::construct_time_integrator(state.args["time"]["integrator"]);
			std::vector<Eigen::VectorXd> x_prevs;
			std::vector<Eigen::VectorXd> v_prevs;
			std::vector<Eigen::VectorXd> a_prevs;
			this->split_time_integrator_quantities(
				local_mesh.projection_quantities(), this->dim(), x_prevs, v_prevs,
				a_prevs);
			solve_data.time_integrator->init(
				x_prevs, v_prevs, a_prevs, state.args["time"]["dt"]);
		}

		// TODO: Initialize solve_data.rhs_assembler
		assert(solve_data.rhs_assembler == nullptr);
		// NOTE: These need to stay alive if we use the rhs_assembler.
		const std::vector<mesh::LocalBoundary> local_boundary;
		const std::vector<mesh::LocalBoundary> local_neumann_boundary;
		const auto rhs = Eigen::MatrixXd::Zero(target_x.rows(), target_x.cols());

		std::vector<std::shared_ptr<Form>> forms;
		{
			POLYFEM_REMESHER_SCOPED_TIMER("Init forms");
			forms = solve_data.init_forms(
				// General
				this->dim(), this->current_time,
				// Elastic form
				n_bases, bases, /*geom_bases=*/bases, assembler, ass_vals_cache,
				// Body form
				/*n_pressure_bases=*/0, boundary_nodes, local_boundary,
				local_neumann_boundary, state.n_boundary_samples(), rhs,
				/*sol=*/target_x, mass_matrix_assembler.density(),
				// Inertia form
				state.args["solver"]["ignore_inertia"], mass, /*damping_assembler=*/nullptr,
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
				contact_enabled
					? state.solve_data.contact_form->barrier_stiffness()
					: 0,
				state.args["solver"]["contact"]["CCD"]["broad_phase"],
				state.args["solver"]["contact"]["CCD"]["tolerance"],
				state.args["solver"]["contact"]["CCD"]["max_iterations"],
				// Friction form
				state.args["contact"]["friction_coefficient"],
				state.args["contact"]["epsv"],
				state.args["solver"]["contact"]["friction_iterations"],
				// Rayleigh damping form
				state.args["solver"]["rayleigh_damping"]);

			assert(solve_data.body_form == nullptr);

			// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

			// Augmented Lagrangian form
			assert(solve_data.al_form == nullptr);
			solve_data.al_form = std::make_shared<ALForm>(
				ndof, boundary_nodes, mass, Obstacle(), target_x);
			forms.push_back(solve_data.al_form);
			assert(state.solve_data.al_form != nullptr);
			solve_data.al_form->set_weight(state.solve_data.al_form->weight());
		}

		solve_data.nl_problem = std::make_shared<polyfem::solver::StaticBoundaryNLProblem>(
			ndof, boundary_nodes, target_x, forms);

		assert(solve_data.time_integrator != nullptr);
		solve_data.nl_problem->update_quantities(this->current_time, solve_data.time_integrator->x_prev());
		solve_data.nl_problem->init(target_x);
		solve_data.nl_problem->init_lagging(solve_data.time_integrator->x_prev());
		solve_data.nl_problem->update_lagging(target_x, /*iter_num=*/0);
	}

	// ----------------------------------------------------------------------------------------------
	// Template specializations
	template class PhysicsRemesher<wmtk::TriMesh>;
	template class PhysicsRemesher<wmtk::TetMesh>;
} // namespace polyfem::mesh