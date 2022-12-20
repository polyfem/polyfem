#include <polyfem/mesh/remesh/WildRemesh2D.hpp>

#include <polyfem/mesh/mesh2D/CMesh2D.hpp>
#include <polyfem/mesh/remesh/wild_remesh/LocalMesh.hpp>
#include <polyfem/mesh/remesh/L2Projection.hpp>
#include <polyfem/basis/LagrangeBasis2d.hpp>
#include <polyfem/solver/ALSolver.hpp>
#include <polyfem/solver/forms/ContactForm.hpp>
#include <polyfem/solver/forms/FrictionForm.hpp>
#include <polyfem/solver/forms/LinearForm.hpp>
#include <polyfem/time_integrator/ImplicitTimeIntegrator.hpp>
#include <polyfem/io/OBJWriter.hpp>
#include <polyfem/utils/Timer.hpp>

#include <igl/boundary_facets.h>

#define POLYFEM_REMESH_USE_FRICTION_FORM

namespace polyfem::mesh
{
	int WildRemeshing2D::build_bases(
		const Eigen::MatrixXd &V,
		const Eigen::MatrixXi &F,
		const std::string &assembler_formulation,
		std::vector<polyfem::basis::ElementBases> &bases,
		Eigen::VectorXi &vertex_to_basis)
	{
		using namespace polyfem::basis;

		CMesh2D mesh;
		mesh.build_from_matrices(V, F);
		std::vector<LocalBoundary> local_boundary;
		std::map<int, basis::InterfaceData> poly_edge_to_data;
		std::shared_ptr<mesh::MeshNodes> mesh_nodes;
		const int n_bases = LagrangeBasis2d::build_bases(
			mesh,
			assembler_formulation,
			/*quadrature_order=*/1,
			/*mass_quadrature_order=*/2,
			/*discr_order=*/1,
			/*serendipity=*/false,
			/*has_polys=*/false,
			/*is_geom_bases=*/false,
			bases,
			local_boundary,
			poly_edge_to_data,
			mesh_nodes);

		// TODO: use mesh_nodes to build vertex_to_basis
		vertex_to_basis.setConstant(V.rows(), -1);
		for (const ElementBases &elm : bases)
		{
			for (const Basis &basis : elm.bases)
			{
				assert(basis.global().size() == 1);
				const int basis_id = basis.global()[0].index;
				const RowVectorNd v = basis.global()[0].node;

				for (int i = 0; i < V.rows(); i++)
				{
					// if ((V.row(i) - v).norm() < 1e-14)
					if ((V.row(i).array() == v.array()).all())
					{
						if (vertex_to_basis[i] == -1)
							vertex_to_basis[i] = basis_id;
						assert(vertex_to_basis[i] == basis_id);
						break;
					}
				}
			}
		}

		return n_bases;
	}

	assembler::AssemblerUtils WildRemeshing2D::create_assembler(
		const std::vector<int> &body_ids) const
	{
		POLYFEM_SCOPED_TIMER(timings.create_assembler);
		assembler::AssemblerUtils new_assembler = state.assembler;
		assert(utils::is_param_valid(state.args, "materials"));
		new_assembler.set_materials(body_ids, state.args["materials"]);
		return new_assembler;
	}

	bool WildRemeshing2D::local_relaxation(const Tuple &t, const int n_ring)
	{
		using namespace polyfem::solver;
		using namespace polyfem::time_integrator;

		constexpr bool free_boundary = true;

		// 1. Get the n-ring of triangles around the vertex.
		// LocalMesh local_mesh = LocalMesh::n_ring(
		// 	*this, t, n_ring, /*include_global_boundary=*/free_boundary);
		LocalMesh local_mesh = LocalMesh::flood_fill_n_ring(
			*this, t, flood_fill_rel_area * total_area, /*include_global_boundary=*/free_boundary);
		// LocalMesh local_mesh(*this, get_faces(), /*include_global_boundary=*/free_boundary);

		std::vector<polyfem::basis::ElementBases> bases;
		int n_bases;
		{
			POLYFEM_SCOPED_TIMER(timings.build_bases);
			Eigen::VectorXi vertex_to_basis;
			n_bases = WildRemeshing2D::build_bases(
				local_mesh.rest_positions(), local_mesh.triangles(),
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
		const int ndof = n_bases * DIM;

		// io::OBJWriter::write(
		// 	state.resolve_output_path("local_rest_mesh_before_local_relaxation.obj"),
		// 	local_mesh.rest_positions(), local_mesh.triangles());
		// io::OBJWriter::write(
		// 	state.resolve_output_path("local_deformed_mesh_before_local_relaxation.obj"),
		// 	local_mesh.positions(), local_mesh.triangles());

		// write_rest_obj(state.resolve_output_path("rest_mesh_before_local_relaxation.obj"));
		// write_deformed_obj(state.resolve_output_path("deformed_mesh_before_local_relaxation.obj"));

		// io::OBJWriter::write(
		// 	state.resolve_output_path("fixed_vertices.obj"),
		// 	local_mesh.positions()(local_mesh.fixed_vertices(), Eigen::all), Eigen::MatrixXi());

		// --------------------------------------------------------------------

		std::vector<int> boundary_nodes;
		{
			POLYFEM_SCOPED_TIMER(timings.create_boundary_nodes);

			const auto add_vertex_to_boundary_nodes = [&](const int vi) {
				for (int d = 0; d < DIM; ++d)
					boundary_nodes.push_back(DIM * vi + d);
			};

			for (const int vi : local_mesh.fixed_vertices())
				add_vertex_to_boundary_nodes(vi);

			if (free_boundary)
			{
				// TODO: handle the correct DBC
				for (int ei = 0; ei < local_mesh.boundary_edges().rows(); ei++)
				{
					const int boundary_id = local_mesh.boundary_ids()[ei];
					if (boundary_id == 2 || boundary_id == 4)
					{
						add_vertex_to_boundary_nodes(local_mesh.boundary_edges()(ei, 0));
						add_vertex_to_boundary_nodes(local_mesh.boundary_edges()(ei, 1));
					}
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
			ass_vals_cache.init(/*is_volume=*/DIM == 3, bases, /*gbases=*/bases, /*is_mass=*/true);
			assembler.assemble_mass_matrix(
				/*assembler_formulation=*/"", /*is_volume=*/DIM == 3, n_bases,
				/*use_density=*/true, bases, /*gbases=*/bases, ass_vals_cache, mass);
			// Set the mass of the codimensional fixed vertices to the average mass.
			for (int i = DIM * local_mesh.num_local_vertices(); i < DIM * local_mesh.num_vertices(); ++i)
				mass.coeffRef(i, i) = state.avg_mass;
		}
		ass_vals_cache.init(/*is_volume=*/DIM == 3, bases, /*gbases=*/bases, /*is_mass=*/false);

		ipc::CollisionMesh collision_mesh; // This has to stay alive
		const bool contact_enabled = state.args["contact"]["enabled"] && free_boundary;
		if (contact_enabled)
		{
			POLYFEM_SCOPED_TIMER(timings.create_collision_mesh);
			collision_mesh = ipc::CollisionMesh::build_from_full_mesh(
				local_mesh.rest_positions(), local_mesh.boundary_edges(),
				/*boundary_faces=*/Eigen::MatrixXi());
		}

		SolveData solve_data;

		// Initialize time integrator
		if (state.problem->is_time_dependent())
		{
			solve_data.time_integrator =
				ImplicitTimeIntegrator::construct_time_integrator(state.args["time"]["integrator"]);
			solve_data.time_integrator->init(
				utils::flatten(local_mesh.prev_displacements()),
				utils::flatten(local_mesh.prev_velocities()),
				utils::flatten(local_mesh.prev_accelerations()),
				state.args["time"]["dt"]);
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
				DIM, current_time,
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
			const double global_energy_before = starting_energy;
			const double rel_diff = abs_diff / global_energy_before;

			accept = rel_diff >= energy_relative_tolerance && abs_diff >= energy_absolute_tolerance;

			static int log_i = 0;
			logger().log(
				accept ? spdlog::level::critical : spdlog::level::trace,
				"{} | rel_diff={:g} rel_tol={:g} | abs_diff={:g} abs_tol={:g}",
				log_i++, rel_diff, energy_relative_tolerance, abs_diff, energy_absolute_tolerance);
		}

		// Update positions only on acceptance
		if (accept)
		{
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

			static int save_i = 0;
			write_deformed_mesh(state.resolve_output_path(fmt::format("split_{:04d}.vtu", save_i++)));

			for (const auto &[glob_vi, loc_vi] : local_mesh.global_to_local())
			{
				const Eigen::Vector2d u = sol.middleRows<DIM>(DIM * loc_vi);
				const Eigen::Vector2d u_old = vertex_attrs[glob_vi].displacement();
				vertex_attrs[glob_vi].position = vertex_attrs[glob_vi].rest_position + u;

#ifndef POLYFEM_REMESH_USE_FRICTION_FORM
				const Eigen::Vector2d f = friction_gradient.segment<DIM>(DIM * loc_vi);
				const Eigen::Vector2d f_old = vertex_attrs[glob_vi].projection_quantities.rightCols(1);
				if (f_old.norm() != 0)
					logger().critical("(f_old⋅f)/(‖f_old‖‖f‖)={:g} ‖f_old‖/‖f‖={:g} ‖u_old - u‖={:g}", (f_old / f_old.norm()).dot(f / f.norm()), f.norm() / f_old.norm(), (u_old - u).norm());
				vertex_attrs[glob_vi].projection_quantities.rightCols(1) = f;
#endif
			}

#ifndef POLYFEM_REMESH_USE_FRICTION_FORM
			m_obstacle_vals.rightCols(1) = friction_gradient.tail(m_obstacle_vals.rows());
#endif

			write_deformed_mesh(state.resolve_output_path(fmt::format("split_{:04d}.vtu", save_i++)));
		}

		return accept;
	}

} // namespace polyfem::mesh