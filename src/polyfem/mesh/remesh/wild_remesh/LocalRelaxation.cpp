#include <polyfem/mesh/remesh/WildRemesh2D.hpp>

#include <polyfem/mesh/mesh2D/CMesh2D.hpp>
#include <polyfem/mesh/remesh/wild_remesh/LocalMesh.hpp>
#include <polyfem/mesh/remesh/L2Projection.hpp>
#include <polyfem/basis/FEBasis2d.hpp>
#include <polyfem/solver/ALSolver.hpp>
#include <polyfem/solver/SparseNewtonDescentSolver.hpp>
#include <polyfem/solver/forms/ContactForm.hpp>
#include <polyfem/solver/forms/ElasticForm.hpp>
#include <polyfem/solver/forms/InertiaForm.hpp>
#include <polyfem/time_integrator/ImplicitTimeIntegrator.hpp>
#include <polyfem/io/OBJWriter.hpp>
#include <polyfem/io/MatrixIO.hpp>

#include <igl/boundary_facets.h>

namespace polyfem::mesh
{
	namespace
	{
		inline bool contains(const std::unordered_set<int> &set, int val)
		{
			return set.find(val) != set.end();
		}
	} // namespace

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
		const int n_bases = FEBasis2d::build_bases(
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
		// const LocalMesh local_mesh = LocalMesh::n_ring(
		// 	*this, t, n_ring, /*include_global_boundary=*/free_boundary);
		const LocalMesh local_mesh = LocalMesh::flood_fill_n_ring(
			*this, t, flood_fill_rel_area * total_area, /*include_global_boundary=*/free_boundary);

		std::vector<polyfem::basis::ElementBases> bases;
		Eigen::VectorXi vertex_to_basis;
		int n_bases = WildRemeshing2D::build_bases(
			local_mesh.rest_positions(), local_mesh.triangles(), state.formulation(),
			bases, vertex_to_basis);

		assert(n_bases == local_mesh.num_local_vertices());
		n_bases = local_mesh.num_vertices();
		assert(vertex_to_basis.size() == n_bases);
		for (int i = local_mesh.num_local_vertices(); i < n_bases; i++)
		{
			vertex_to_basis[i] = i;
		}

#ifndef NDEBUG
		for (const int basis_id : vertex_to_basis)
		{
			assert(basis_id >= 0);
		}
#endif

		// io::OBJWriter::write(
		// 	state.resolve_output_path("local_mesh0_before.obj"),
		// 	local_mesh.rest_positions(), local_mesh.triangles());
		// io::OBJWriter::write(
		// 	state.resolve_output_path("local_mesh1_before.obj"),
		// 	local_mesh.positions(), local_mesh.triangles());

		// write_rest_obj(state.resolve_output_path("global_mesh0_before.obj"));
		// write_deformed_obj(state.resolve_output_path("global_mesh1_before.obj"));

		// io::OBJWriter::write(
		// 	state.resolve_output_path("fixed_vertices.obj"),
		// 	local_mesh.positions()(local_mesh.fixed_vertices(), Eigen::all), Eigen::MatrixXi());

		// --------------------------------------------------------------------

		std::vector<int> boundary_nodes;
		const auto add_vertex_to_boundary_nodes = [&](const int v) {
			const int basis_id = vertex_to_basis[v];
			assert(basis_id >= 0);
			for (int d = 0; d < DIM; ++d)
			{
				boundary_nodes.push_back(DIM * basis_id + d);
			}
		};
		for (const int vi : local_mesh.fixed_vertices())
		{
			add_vertex_to_boundary_nodes(vi);
		}
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

		const Eigen::MatrixXd target_x = utils::flatten(utils::reorder_matrix(
			local_mesh.displacements(), vertex_to_basis));

		// --------------------------------------------------------------------

		// 2. Perform "relaxation" by minimizing the elastic energy of the n-ring
		// with the boundary fixed.

		std::vector<std::shared_ptr<Form>> forms;

		// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

		assembler::AssemblerUtils assembler = create_assembler(local_mesh.body_ids());

		assembler::AssemblyValsCache ass_vals_cache;
		ass_vals_cache.init(/*is_volume=*/DIM == 3, bases, /*gbases=*/bases, /*is_mass=*/false);
		std::shared_ptr<ElasticForm> elastic_form = std::make_shared<ElasticForm>(
			n_bases, bases, bases, assembler, ass_vals_cache, state.formulation(),
			/*dt=*/0, /*is_volume=*/DIM == 3);
		elastic_form->set_weight(state.solve_data.elastic_form->weight());
		forms.push_back(elastic_form);

		// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

		ass_vals_cache.init(/*is_volume=*/DIM == 3, bases, /*gbases=*/bases, /*is_mass=*/true);
		Eigen::SparseMatrix<double> M;
		assembler.assemble_mass_matrix(
			/*assembler_formulation=*/"", /*is_volume=*/DIM == 3, n_bases,
			/*use_density=*/true, bases, /*gbases=*/bases, ass_vals_cache, M);
		for (int i = DIM * local_mesh.num_local_vertices(); i < DIM * local_mesh.num_vertices(); ++i)
		{
			M.coeffRef(i, i) = state.avg_mass;
		}

		// TODO: set the diagonal of M to be 1 for the collision boundary nodes.

		std::shared_ptr<ALForm> al_form = std::make_shared<ALForm>(
			n_bases * DIM, boundary_nodes, M, Obstacle(), target_x);
		al_form->set_weight(state.solve_data.al_form->weight());
		forms.push_back(al_form);

		// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

		ipc::CollisionMesh collision_mesh; // This has to stay alive
		if (free_boundary)
		{
			collision_mesh = ipc::CollisionMesh::build_from_full_mesh(
				utils::reorder_matrix(local_mesh.rest_positions(), vertex_to_basis),
				utils::map_index_matrix(local_mesh.boundary_edges(), vertex_to_basis),
				/*boundary_faces=*/Eigen::MatrixXi());
			// io::OBJWriter::write(
			// 	state.resolve_output_path("collision_mesh_before.obj"),
			// 	collision_mesh.displace_vertices(utils::unflatten(target_x, DIM)), collision_mesh.edges());

			std::shared_ptr<ContactForm> contact_form = std::make_shared<ContactForm>(
				collision_mesh,
				state.args["contact"]["dhat"],
				state.avg_mass,
				/*use_adaptive_barrier_stiffness=*/false,
				/*is_time_dependent=*/false,
				state.args["solver"]["contact"]["CCD"]["broad_phase"],
				state.args["solver"]["contact"]["CCD"]["tolerance"],
				state.args["solver"]["contact"]["CCD"]["max_iterations"]);
			contact_form->set_weight(state.solve_data.contact_form->weight());
			forms.push_back(contact_form);
		}

		// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

		std::shared_ptr<ImplicitTimeIntegrator> time_integrator;
		if (!state.args["solver"]["ignore_inertia"])
		{
			time_integrator =
				ImplicitTimeIntegrator::construct_time_integrator(state.args["time"]["integrator"]);
			time_integrator->init(
				utils::flatten(utils::reorder_matrix(local_mesh.prev_positions(), vertex_to_basis)),
				utils::flatten(utils::reorder_matrix(local_mesh.prev_velocities(), vertex_to_basis)),
				utils::flatten(utils::reorder_matrix(local_mesh.prev_accelerations(), vertex_to_basis)),
				state.args["time"]["dt"]);

			std::shared_ptr<InertiaForm> inertia_form = std::make_shared<InertiaForm>(M, *time_integrator);
			inertia_form->set_weight(state.solve_data.inertia_form->weight());
			forms.push_back(inertia_form);
		}

		// --------------------------------------------------------------------

		StaticBoundaryNLProblem nl_problem(
			n_bases * DIM, boundary_nodes, target_x, forms);

		// --------------------------------------------------------------------

		// Create Newton solver
		std::shared_ptr<cppoptlib::NonlinearSolver<decltype(nl_problem)>> nl_solver;
		{
			// TODO: expose these parameters
			const json newton_args = R"({
				"f_delta": 1e-10,
				"grad_norm": 1e-5,
				"use_grad_norm": true,
				"first_grad_norm_tol": 1e-10,
				"max_iterations": 1000,
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
			using NewtonSolver = cppoptlib::SparseNewtonDescentSolver<decltype(nl_problem)>;
			nl_solver = std::make_shared<NewtonSolver>(newton_args, linear_solver_args);
		}

		// --------------------------------------------------------------------

		// TODO: Make these parameters
		const double al_initial_weight = 1e6;
		const double al_max_weight = 1e11;
		const bool force_al = false;

		// Create augmented Lagrangian solver
		ALSolver al_solver(
			nl_solver, al_form, al_initial_weight, al_max_weight,
			[](const Eigen::MatrixXd &) {});

		Eigen::MatrixXd sol = target_x;
		const auto level_before = logger().level();
		logger().set_level(spdlog::level::warn);
		al_solver.solve(nl_problem, sol, force_al);
		logger().set_level(level_before);

		// --------------------------------------------------------------------

		for (const auto &[glob_vi, loc_vi] : local_mesh.global_to_local())
		{
			const long basis_vi = vertex_to_basis[loc_vi];

			if (basis_vi < 0)
				continue;

			const Eigen::Vector2d u = sol.middleRows<DIM>(DIM * basis_vi);
			vertex_attrs[glob_vi].position = vertex_attrs[glob_vi].rest_position + u;
		}

		// 3. Return the energy of the relaxed mesh.
		const double local_energy_before = nl_problem.value(target_x);
		const double local_energy_after = nl_problem.value(sol);

		assert(std::isfinite(local_energy_before));
		assert(std::isfinite(local_energy_after));

		const double abs_diff = local_energy_before - local_energy_after; // > 0 if energy decreased
		// TODO: compute global_energy_before
		const double global_energy_before = local_energy_before;
		const double rel_diff = abs_diff / (global_energy_before);

		// const LocalMesh local_mesh_after = LocalMesh::n_ring(*this, t, n_ring);

		// io::OBJWriter::write(
		// 	state.resolve_output_path("local_mesh0.obj"),
		// 	local_mesh_after.rest_positions(), local_mesh_after.triangles());
		// io::OBJWriter::write(
		// 	state.resolve_output_path("local_mesh1.obj"),
		// 	local_mesh_after.positions(), local_mesh_after.triangles());

		// write_rest_obj(state.resolve_output_path("global_mesh0.obj"));
		// write_deformed_obj(state.resolve_output_path("global_mesh1.obj"));

		// io::OBJWriter::write(
		// 	state.resolve_output_path("collision_mesh.obj"),
		// 	collision_mesh.displace_vertices(utils::unflatten(sol, DIM)), collision_mesh.edges());

		static int i = 0;
		if (rel_diff >= energy_relative_tolerance && abs_diff >= energy_absolute_tolerance)
		{
			logger().critical(
				"{} energy_before={:g} energy_after={:g} rel_diff={:g} abs_diff={:g}",
				i, local_energy_before, local_energy_after, rel_diff, abs_diff);
		}
		i++;

		return rel_diff >= energy_relative_tolerance && abs_diff >= energy_absolute_tolerance;
	}

} // namespace polyfem::mesh