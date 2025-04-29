#include "LocalRelaxationData.hpp"

#include <polyfem/mesh/remesh/Remesher.hpp>
#include <polyfem/mesh/remesh/WildRemesher.hpp>
#include <polyfem/assembler/Mass.hpp>
#include <polyfem/solver/forms/ContactForm.hpp>
#include <polyfem/solver/forms/lagrangian/BCLagrangianForm.hpp>
#include <polyfem/solver/forms/ContactForm.hpp>
#include <polyfem/solver/problems/StaticBoundaryNLProblem.hpp>
#include <polyfem/time_integrator/ImplicitTimeIntegrator.hpp>
#include <polyfem/solver/forms/lagrangian/AugmentedLagrangianForm.hpp>

namespace polyfem::mesh
{
	template <typename M>
	LocalRelaxationData<M>::LocalRelaxationData(
		const State &state,
		LocalMesh<M> &local_mesh,
		const double current_time,
		const bool contact_enabled)
		: local_mesh(local_mesh)
	{
		problem = std::make_shared<assembler::GenericTensorProblem>("GenericTensor");

		init_mesh(state);
		init_bases(state);
		init_boundary_conditions(state);
		init_assembler(state);
		init_mass_matrix(state);
		init_solve_data(state, current_time, contact_enabled);
	}

	template <typename M>
	void LocalRelaxationData<M>::init_mesh(const State &)
	{
		POLYFEM_REMESHER_SCOPED_TIMER("LocalRelaxationData::init_mesh");

		mesh = Mesh::create(local_mesh.rest_positions(), local_mesh.elements());
		assert(mesh->n_vertices() == local_mesh.num_vertices());

		std::vector<int> boundary_ids(mesh->n_boundary_elements(), -1);
		if constexpr (std::is_same_v<M, WildTriRemesher>)
		{
			const auto local_boundary_ids = std::get<Remesher::EdgeMap<int>>(local_mesh.boundary_ids());
			for (int i = 0; i < mesh->n_edges(); i++)
			{
				const size_t e0 = mesh->edge_vertex(i, 0);
				const size_t e1 = mesh->edge_vertex(i, 1);
				boundary_ids[i] = local_boundary_ids.at({{e0, e1}});
			}
		}
		else
		{
			const auto local_boundary_ids = std::get<Remesher::FaceMap<int>>(local_mesh.boundary_ids());
			for (int i = 0; i < mesh->n_faces(); i++)
			{
				std::array<size_t, 3> f = {{
					(size_t)mesh->face_vertex(i, 0),
					(size_t)mesh->face_vertex(i, 1),
					(size_t)mesh->face_vertex(i, 2),
				}};
				boundary_ids[i] = local_boundary_ids.at(f);
			}
		}

		mesh->set_boundary_ids(boundary_ids);
		mesh->set_body_ids(local_mesh.body_ids());
	}

	template <typename M>
	void LocalRelaxationData<M>::init_bases(const State &state)
	{
		POLYFEM_REMESHER_SCOPED_TIMER("LocalRelaxationData::init_bases");

		Eigen::VectorXi vertex_to_basis;
		m_n_bases = Remesher::build_bases(
			*mesh, state.formulation(), bases, local_boundary, vertex_to_basis);

		assert(m_n_bases == local_mesh.num_local_vertices());
		m_n_bases = local_mesh.num_vertices(); // the real n_bases includes the global boundary nodes
		assert(vertex_to_basis.size() == m_n_bases);

		const int start_i = local_mesh.num_local_vertices();
		if (start_i < m_n_bases)
		{
			// set tail to range [start_i, n_bases)
			std::iota(vertex_to_basis.begin() + start_i, vertex_to_basis.end(), start_i);
		}

		assert(std::all_of(vertex_to_basis.begin(), vertex_to_basis.end(), [](const int basis_id) {
			return basis_id >= 0;
		}));

		// State::build_node_mapping();
		local_mesh.reorder_vertices(vertex_to_basis);
		problem->update_nodes(vertex_to_basis);
		mesh->update_nodes(vertex_to_basis);
	}

	template <typename M>
	void LocalRelaxationData<M>::init_boundary_conditions(const State &state)
	{
		POLYFEM_REMESHER_SCOPED_TIMER("LocalRelaxationData::init_boundary_conditions");

		assert(mesh != nullptr);
		state.problem->init(*mesh);

		std::vector<int> pressure_boundary_nodes;
		state.problem->setup_bc(
			*mesh, n_bases() - state.obstacle.n_vertices(), bases, /*geom_bases=*/bases,
			/*pressure_bases=*/std::vector<basis::ElementBases>(), local_boundary,
			boundary_nodes, local_neumann_boundary, local_pressure_boundary,
			local_pressure_cavity, pressure_boundary_nodes, dirichlet_nodes, neumann_nodes);

		auto find_node_position = [&](const int n_id) {
			for (const auto &bs : bases)
				for (const auto &b : bs.bases)
					for (const auto &lg : b.global())
						if (lg.index == n_id)
							return lg.node;
			log_and_throw_error("Node not found");
		};

		// setup nodal values
		dirichlet_nodes_position.resize(dirichlet_nodes.size());
		for (int n = 0; n < dirichlet_nodes.size(); ++n)
			dirichlet_nodes_position[n] = find_node_position(dirichlet_nodes[n]);

		neumann_nodes_position.resize(neumann_nodes.size());
		for (int n = 0; n < neumann_nodes.size(); ++n)
			neumann_nodes_position[n] = find_node_position(neumann_nodes[n]);

		// Add fixed boundary DOF
		for (const int vi : local_mesh.fixed_vertices())
		{
			for (int d = 0; d < dim(); ++d)
			{
				boundary_nodes.push_back(vi * dim() + d);
			}
		}

		std::sort(boundary_nodes.begin(), boundary_nodes.end());
		auto it = std::unique(boundary_nodes.begin(), boundary_nodes.end());
		boundary_nodes.erase(it, boundary_nodes.end());
	}

	template <typename M>
	void LocalRelaxationData<M>::init_assembler(const State &state)
	{
		POLYFEM_REMESHER_SCOPED_TIMER("LocalRelaxationData::init_assembler");
		assert(utils::is_param_valid(state.args, "materials"));

		assembler = assembler::AssemblerUtils::make_assembler(state.formulation());
		assert(assembler->name() == state.formulation());
		assembler->set_size(dim());
		assembler->set_materials(local_mesh.body_ids(), state.args["materials"], state.units);

		mass_matrix_assembler = std::make_shared<assembler::Mass>();
		mass_matrix_assembler->set_size(dim());
		mass_matrix_assembler->set_materials(local_mesh.body_ids(), state.args["materials"], state.units);

		pressure_assembler = nullptr; // TODO: implement this
	}

	template <typename M>
	void LocalRelaxationData<M>::init_mass_matrix(const State &state)
	{
		POLYFEM_REMESHER_SCOPED_TIMER("LocalRelaxationData::init_mass_matrix");

		// Assemble the mass matrix.
		mass_assembly_vals_cache.init(
			is_volume(), bases, /*gbases=*/bases, /*is_mass=*/true);
		assert(mass_matrix_assembler != nullptr);
		mass_matrix_assembler->assemble(
			is_volume(), n_bases(), bases, /*gbases=*/bases,
			mass_assembly_vals_cache,
			/*t=*/0, // TODO: time-dependent mass matrix
			mass, /*is_mass=*/true);

		// Set the mass of the codimensional fixed vertices to the average mass.
		const int local_ndof = dim() * local_mesh.num_local_vertices();
		for (int i = local_ndof; i < ndof(); ++i)
			mass.coeffRef(i, i) = state.avg_mass;
	}

	template <typename M>
	void LocalRelaxationData<M>::init_solve_data(
		const State &state,
		const double current_time,
		const bool contact_enabled)
	{
		// Current solution.
		const Eigen::MatrixXd target_x = this->sol();

		// Assemble the stiffness matrix.
		assembly_vals_cache.init(is_volume(), bases, /*gbases=*/bases, /*is_mass=*/false);

		// Create collision mesh.
		if (contact_enabled)
		{
			POLYFEM_REMESHER_SCOPED_TIMER("LocalRelaxationData::init_solve_data -> create collision mesh");

			collision_mesh = ipc::CollisionMesh::build_from_full_mesh(
				local_mesh.rest_positions(), local_mesh.boundary_edges(),
				local_mesh.boundary_faces());

			// Ignore all collisions between fixed elements.
			std::vector<bool> is_vertex_fixed(local_mesh.num_vertices(), false);
			for (const int vi : local_mesh.fixed_vertices())
				is_vertex_fixed[vi] = true;
			collision_mesh.can_collide = [is_vertex_fixed, this](size_t vi, size_t vj) {
				return !is_vertex_fixed[this->collision_mesh.to_full_vertex_id(vi)]
					   || !is_vertex_fixed[this->collision_mesh.to_full_vertex_id(vj)];
			};
		}

		// Initialize time integrator
		if (state.problem->is_time_dependent())
		{
			POLYFEM_REMESHER_SCOPED_TIMER("LocalRelaxationData::init_solve_data -> create time integrator");
			solve_data.time_integrator =
				time_integrator::ImplicitTimeIntegrator::construct_time_integrator(
					state.args["time"]["integrator"]);
			Eigen::MatrixXd x_prevs, v_prevs, a_prevs;
			Remesher::split_time_integrator_quantities(
				local_mesh.projection_quantities(), dim(), x_prevs, v_prevs, a_prevs);
			solve_data.time_integrator->init(
				x_prevs, v_prevs, a_prevs, state.args["time"]["dt"]);
		}

		// Initialize solve_data.rhs_assembler
		{
			POLYFEM_REMESHER_SCOPED_TIMER("LocalRelaxationData::init_solve_data -> create RHS assembler");

			json rhs_solver_params = state.args["solver"]["linear"];
			if (!rhs_solver_params.contains("Pardiso"))
				rhs_solver_params["Pardiso"] = {};
			rhs_solver_params["Pardiso"]["mtype"] = -2; // matrix type for Pardiso (2 = SPD)

			const int size = state.problem->is_scalar() ? 1 : dim();
			solve_data.rhs_assembler = std::make_shared<assembler::RhsAssembler>(
				*assembler, *mesh, Obstacle(), dirichlet_nodes, neumann_nodes,
				dirichlet_nodes_position, neumann_nodes_position, n_bases(),
				dim(), bases, /*geom_bases=*/bases, mass_assembly_vals_cache,
				*state.problem, state.args["space"]["advanced"]["bc_method"],
				rhs_solver_params);

			solve_data.rhs_assembler->assemble(mass_matrix_assembler->density(), rhs);
			rhs *= -1;
		}

		std::vector<std::shared_ptr<solver::Form>> forms;
		{
			POLYFEM_REMESHER_SCOPED_TIMER("LocalRelaxationData::init_solve_data -> init forms");
			forms = solve_data.init_forms(
				// General
				state.units, dim(), current_time,
				// Elastic form
				n_bases(), bases, /*geom_bases=*/bases, *assembler,
				assembly_vals_cache, assembly_vals_cache, state.args["solver"]["advanced"]["jacobian_threshold"], state.args["solver"]["advanced"]["check_inversion"],
				// Body form
				/*n_pressure_bases=*/0, boundary_nodes, local_boundary,
				local_neumann_boundary, state.n_boundary_samples(), rhs,
				target_x, mass_matrix_assembler->density(),
				// Pressure form
				local_pressure_boundary, local_pressure_cavity, pressure_assembler,
				// Inertia form
				state.args.value("/time/quasistatic"_json_pointer, true), mass,
				/*damping_assembler=*/nullptr,
				// Lagged regularization form
				state.args["solver"]["advanced"]["lagged_regularization_weight"],
				state.args["solver"]["advanced"]["lagged_regularization_iterations"],
				// Augmented lagrangian form
				/*obstacle_ndof=*/0,
				// Contact form
				contact_enabled, collision_mesh, state.args["contact"]["dhat"],
				state.avg_mass, state.args["contact"]["use_convergent_formulation"],
				contact_enabled ? state.solve_data.contact_form->barrier_stiffness() : 0,
				state.args["solver"]["contact"]["CCD"]["broad_phase"],
				state.args["solver"]["contact"]["CCD"]["tolerance"],
				state.args["solver"]["contact"]["CCD"]["max_iterations"],
				/*enable_shape_derivatives=*/false,
				state.args["contact"],
				// Homogenization
				assembler::MacroStrainValue(),
				// Periodic contact
				/*periodic_contact=*/false, /*tiled_to_single=*/Eigen::VectorXi(), /*periodicbc=*/nullptr,
				// Friction form
				state.args["contact"]["friction_coefficient"],
				state.args["contact"]["epsv"],
				state.args["solver"]["contact"]["friction_iterations"],
				// Rayleigh damping form
				state.args["solver"]["rayleigh_damping"]);

			// Remove all AL forms because we do not need them in the remeshing process.
			for (auto &lf : solve_data.al_form)
				forms.erase(std::remove(forms.begin(), forms.end(), lf), forms.end());
			solve_data.al_form.clear();
		}
		const std::vector<std::shared_ptr<polyfem::solver::AugmentedLagrangianForm>> penalty_forms;
		solve_data.nl_problem = std::make_shared<polyfem::solver::StaticBoundaryNLProblem>(
			ndof(), target_x, forms, penalty_forms);

		assert(solve_data.time_integrator != nullptr);
		solve_data.nl_problem->update_quantities(current_time, solve_data.time_integrator->x_prev());
		solve_data.nl_problem->init(target_x);
		solve_data.nl_problem->init_lagging(solve_data.time_integrator->x_prev());
		solve_data.nl_problem->update_lagging(target_x, /*iter_num=*/0);
	}

	// -------------------------------------------------------------------------
	// Template instantiations
	template class LocalRelaxationData<WildTriRemesher>;
	template class LocalRelaxationData<WildTetRemesher>;
} // namespace polyfem::mesh