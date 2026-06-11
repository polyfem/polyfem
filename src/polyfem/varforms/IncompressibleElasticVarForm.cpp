#include "IncompressibleElasticVarForm.hpp"

#include <cmath>
#include <numeric>
#include <algorithm>

#include <polyfem/assembler/AssemblerUtils.hpp>
#include <polyfem/basis/LagrangeBasis2d.hpp>
#include <polyfem/basis/LagrangeBasis3d.hpp>
#include <polyfem/mesh/mesh2D/Mesh2D.hpp>
#include <polyfem/mesh/mesh3D/Mesh3D.hpp>
#include <polyfem/time_integrator/BDF.hpp>
#include <polyfem/varforms/ResolveDiscrOrder.hpp>
#include <polyfem/varforms/ShouldUseIsoparametric.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/MatrixUtils.hpp>
#include <polyfem/utils/Timer.hpp>
#include <polyfem/varforms/VarFormUtils.hpp>

#include <polysolve/linear/FEMSolver.hpp>

namespace polyfem::varform
{
	using namespace varform::internal;

	IncompressibleElasticVarForm::IncompressibleElasticVarForm(const std::string &formulation, const Units &units, const json &args, const std::string &out_path)
		: ElasticVarForm(formulation, units, args, out_path)
	{
		mixed_assembler = assembler::AssemblerUtils::make_mixed_assembler(formulation);
		pressure_assembler = assembler::AssemblerUtils::make_assembler(assembler::AssemblerUtils::other_assembler_name(formulation));
		assert(assembler->is_linear());
		assert(assembler->is_tensor());
	}

	void IncompressibleElasticVarForm::save_json(const Eigen::MatrixXd &solution, std::ostream &out) const
	{
		save_json_stats(solution, incompressible_spaces.pressure.n_bases, out);
	}

	void IncompressibleElasticVarForm::load_mesh(const mesh::Mesh &mesh, const json &args)
	{
		ElasticVarForm::load_mesh(mesh, args);
		if (mixed_assembler)
			mixed_assembler->set_size(mesh.dimension());
		if (pressure_assembler)
			set_materials(*pressure_assembler);
	}

	int IncompressibleElasticVarForm::primary_ndof() const
	{
		return mesh_ ? displacement_space.n_bases * mesh_->dimension() : 0;
	}

	int IncompressibleElasticVarForm::stacked_ndof() const
	{
		return incompressible_spaces.layout.total_dof();
	}

	void IncompressibleElasticVarForm::build_rhs_assembler()
	{
		json rhs_solver_params = args["solver"]["linear"];
		if (!rhs_solver_params.contains("Pardiso"))
			rhs_solver_params["Pardiso"] = {};
		rhs_solver_params["Pardiso"]["mtype"] = -2;

		rhs_assembler = std::make_shared<assembler::RhsAssembler>(
			*assembler, *mesh_, nullptr,
			boundary.dirichlet_nodes, boundary.neumann_nodes,
			boundary.dirichlet_nodes_position, boundary.neumann_nodes_position,
			displacement_space.n_bases, mesh_->dimension(), *displacement_space.bases, geom_bases(), displacement_caches.mass, *problem,
			args["space"]["advanced"]["bc_method"],
			rhs_solver_params);
	}

	void IncompressibleElasticVarForm::build_basis(mesh::Mesh &mesh, const json &args)
	{
		const bool iso_parametric = should_use_isoparametric(mesh, args);
		displacement_space.value_dim = mesh.dimension();
		displacement_space.geometry = geometry_mapping;

		displacement_space.bases = std::make_shared<std::vector<basis::ElementBases>>();
		geometry_mapping->bases = iso_parametric ? displacement_space.bases : std::make_shared<std::vector<basis::ElementBases>>();

		auto disc = resolve_discr_orders(args, root_path, mesh, stats);
		displacement_space.disc_orders = disc.orders;
		displacement_space.disc_ordersq = disc.ordersq;
		geometry_mapping->disc_orders = resolve_geom_orders(mesh, displacement_space.disc_orders, iso_parametric);

		VarForm::build_basis(mesh, args);

		if (displacement_space.disc_orders.maxCoeff() != displacement_space.disc_orders.minCoeff())
			log_and_throw_error("p refinement not supported in mixed formulation!");

		igl::Timer timer;
		timer.start();

		const auto &all_boundary = boundary.total_local_boundary;
		const int prev_bases = displacement_space.n_bases;
		const int prev_b_size = int(all_boundary.size());
		const bool has_polys = mesh.has_poly();
		const bool use_corner_quadrature = args["space"]["advanced"]["use_corner_quadrature"];
		const int quadrature_order = args["space"]["advanced"]["quadrature_order"].get<int>();
		const int mass_quadrature_order = args["space"]["advanced"]["mass_quadrature_order"].get<int>();
		const int order = args["space"]["pressure_discr_order"];
		std::vector<mesh::LocalBoundary> pressure_local_boundary;
		std::map<int, basis::InterfaceData> pressure_poly_edge_to_data;

		incompressible_spaces.pressure.bases = std::make_shared<std::vector<basis::ElementBases>>();
		incompressible_spaces.pressure.n_bases = 0;
		if (mesh.is_volume())
		{
			const mesh::Mesh3D &tmp_mesh = dynamic_cast<const mesh::Mesh3D &>(mesh);
			incompressible_spaces.pressure.n_bases = basis::LagrangeBasis3d::build_bases(
				tmp_mesh, assembler->name(), quadrature_order, mass_quadrature_order,
				order, order,
				args["space"]["basis_type"] == "Bernstein", false,
				has_polys, false, use_corner_quadrature,
				*incompressible_spaces.pressure.bases, pressure_local_boundary, pressure_poly_edge_to_data, incompressible_spaces.pressure.mesh_nodes);
		}
		else
		{
			const mesh::Mesh2D &tmp_mesh = dynamic_cast<const mesh::Mesh2D &>(mesh);
			incompressible_spaces.pressure.n_bases = basis::LagrangeBasis2d::build_bases(
				tmp_mesh, assembler->name(), quadrature_order, mass_quadrature_order,
				order,
				args["space"]["basis_type"] == "Bernstein", false,
				has_polys, false, use_corner_quadrature,
				*incompressible_spaces.pressure.bases, pressure_local_boundary, pressure_poly_edge_to_data, incompressible_spaces.pressure.mesh_nodes);
		}

		assert(displacement_space.bases->size() == incompressible_spaces.pressure.bases->size());
		for (int i = 0; i < int(incompressible_spaces.pressure.bases->size()); ++i)
		{
			quadrature::Quadrature b_quad;
			displacement_space.bases->at(i).compute_quadrature(b_quad);
			incompressible_spaces.pressure.bases->at(i).set_quadrature([b_quad](quadrature::Quadrature &quad) { quad = b_quad; });
		}

		boundary.local_boundary.clear();
		for (const auto &lb : all_boundary)
			boundary.local_boundary.emplace_back(lb);

		boundary.local_neumann_boundary.clear();
		boundary.local_pressure_boundary.clear();
		boundary.local_pressure_cavity.clear();
		boundary.boundary_nodes.clear();
		boundary.pressure_boundary_nodes.clear();
		boundary.dirichlet_nodes.clear();
		boundary.neumann_nodes.clear();

		problem->setup_bc(
			mesh, displacement_space.n_bases,
			*displacement_space.bases, geom_bases(), *incompressible_spaces.pressure.bases,
			boundary.local_boundary,
			boundary.boundary_nodes,
			boundary.local_neumann_boundary,
			boundary.local_pressure_boundary,
			boundary.local_pressure_cavity,
			boundary.pressure_boundary_nodes,
			boundary.dirichlet_nodes, boundary.neumann_nodes);

		rebuild_node_positions(*displacement_space.bases, boundary.dirichlet_nodes, boundary.dirichlet_nodes_position);
		rebuild_node_positions(*displacement_space.bases, boundary.neumann_nodes, boundary.neumann_nodes_position);

		const bool has_neumann = !boundary.local_neumann_boundary.empty() || int(boundary.local_boundary.size()) < prev_b_size;
		use_avg_pressure = !has_neumann;
		incompressible_spaces.pressure.geometry = geometry_mapping;
		incompressible_spaces.pressure.value_dim = 1;
		incompressible_spaces.layout = SolutionLayout();
		incompressible_spaces.displacement_block = incompressible_spaces.layout.add_block(primary_ndof(), problem->is_time_dependent());
		incompressible_spaces.pressure_block = incompressible_spaces.layout.add_block(incompressible_spaces.pressure.n_bases, false);

		for (int i = prev_bases; i < displacement_space.n_bases; ++i)
			for (int d = 0; d < mesh.dimension(); ++d)
				boundary.boundary_nodes.push_back(i * mesh.dimension() + d);

		std::sort(boundary.boundary_nodes.begin(), boundary.boundary_nodes.end());
		boundary.boundary_nodes.erase(std::unique(boundary.boundary_nodes.begin(), boundary.boundary_nodes.end()), boundary.boundary_nodes.end());

		if (displacement_space.n_bases <= args["solver"]["advanced"]["cache_size"])
			pressure_ass_vals_cache.init(mesh.is_volume(), *incompressible_spaces.pressure.bases, geom_bases());
		else
			pressure_ass_vals_cache.init_empty();

		build_rhs_assembler();

		timer.stop();
		timings.building_basis_time += timer.getElapsedTime();
		logger().info("n pressure bases: {}", incompressible_spaces.pressure.n_bases);
	}

	void IncompressibleElasticVarForm::assemble_rhs(const mesh::Mesh &mesh, const json &args)
	{
		build_rhs_assembler();
		VarForm::assemble_rhs(mesh, args);
		const int prev_size = rhs.rows();
		rhs.conservativeResize(prev_size + incompressible_spaces.pressure.n_bases, rhs.cols());
		rhs.bottomRows(incompressible_spaces.pressure.n_bases).setZero();
	}

	void IncompressibleElasticVarForm::assemble_mass_mat(const mesh::Mesh &mesh, const json &args)
	{
		if (!problem->is_time_dependent())
		{
			avg_mass = 1;
			timings.assembling_mass_mat_time = 0;
			return;
		}

		mass.resize(0, 0);
		igl::Timer timer;
		timer.start();
		logger().info("Assembling mass mat...");
		mass_matrix_assembler->assemble(mesh.is_volume(), displacement_space.n_bases, *displacement_space.bases, geom_bases(), displacement_caches.mass, 0, mass, true);
		avg_mass = 0;
		for (int k = 0; k < mass.outerSize(); ++k)
			for (StiffnessMatrix::InnerIterator it(mass, k); it; ++it)
			{
				assert(it.col() == k);
				avg_mass += it.value();
			}
		avg_mass /= std::max(1, int(mass.rows()));
		if (args["solver"]["advanced"]["lump_mass_matrix"])
			mass = utils::lump_matrix(mass);
		timer.stop();
		timings.assembling_mass_mat_time = timer.getElapsedTime();
		logger().info(" took {}s", timings.assembling_mass_mat_time);
		stats.nn_zero = mass.nonZeros();
		stats.num_dofs = mass.rows();
		stats.mat_size = (long long)mass.rows() * (long long)mass.cols();
	}

	void IncompressibleElasticVarForm::prepare_initial_solution(Eigen::MatrixXd &sol) const
	{
		if (sol.size() <= 0)
			initial_solution(sol);
		if (sol.cols() > 1)
			sol.conservativeResize(Eigen::NoChange, 1);
		sol.conservativeResize(stacked_ndof(), sol.cols());
		sol.bottomRows(incompressible_spaces.pressure.n_bases).setZero();
	}

	void IncompressibleElasticVarForm::split_solution(const Eigen::MatrixXd &stacked, Eigen::MatrixXd &primary, Eigen::MatrixXd &pressure) const
	{
		const int cols = std::max(1, int(stacked.cols()));
		primary.setZero(primary_ndof(), cols);
		pressure.setZero(incompressible_spaces.pressure.n_bases, cols);
		const int primary_rows = std::min(primary_ndof(), int(stacked.rows()));
		if (primary_rows > 0)
			primary.topRows(primary_rows) = stacked.topRows(primary_rows);
		if (stacked.rows() > primary_ndof())
		{
			const int pressure_rows = std::min(incompressible_spaces.pressure.n_bases, int(stacked.rows()) - primary_ndof());
			if (pressure_rows > 0)
				pressure.topRows(pressure_rows) = stacked.middleRows(primary_ndof(), pressure_rows);
		}
	}

	void IncompressibleElasticVarForm::build_stiffness_mat(StiffnessMatrix &stiffness)
	{
		igl::Timer timer;
		timer.start();
		logger().info("Assembling stiffness mat...");

		StiffnessMatrix elastic_stiffness, mixed_stiffness, pressure_stiffness;
		assembler->assemble(mesh_->is_volume(), displacement_space.n_bases, *displacement_space.bases, geom_bases(), displacement_caches.values, 0, elastic_stiffness);
		mixed_assembler->assemble(mesh_->is_volume(), incompressible_spaces.pressure.n_bases, displacement_space.n_bases, *incompressible_spaces.pressure.bases, *displacement_space.bases, geom_bases(), pressure_ass_vals_cache, displacement_caches.values, 0, mixed_stiffness);
		pressure_assembler->assemble(mesh_->is_volume(), incompressible_spaces.pressure.n_bases, *incompressible_spaces.pressure.bases, geom_bases(), pressure_ass_vals_cache, 0, pressure_stiffness);

		assembler::AssemblerUtils::merge_mixed_matrices(
			displacement_space.n_bases, incompressible_spaces.pressure.n_bases, mesh_->dimension(), /*add_average=*/false,
			elastic_stiffness, mixed_stiffness, pressure_stiffness, stiffness);

		timer.stop();
		timings.assembling_stiffness_mat_time = timer.getElapsedTime();
		logger().info(" took {}s", timings.assembling_stiffness_mat_time);
		stats.nn_zero = stiffness.nonZeros();
		stats.num_dofs = stiffness.rows();
		stats.mat_size = (long long)stiffness.rows() * (long long)stiffness.cols();
		write_matrix_market(args, stiffness);
	}

	void IncompressibleElasticVarForm::solve_linear_system(
		const std::unique_ptr<polysolve::linear::Solver> &solver,
		StiffnessMatrix &A,
		Eigen::VectorXd &b,
		const bool compute_spectrum,
		Eigen::MatrixXd &sol)
	{
		Eigen::VectorXd x;
		stats.spectrum = dirichlet_solve(
			*solver,
			A,
			b,
			boundary.boundary_nodes,
			x,
			primary_ndof(),
			args["output"]["data"]["stiffness_mat"],
			compute_spectrum,
			/*is_fluid=*/false,
			/*use_avg_pressure=*/false);
		sol = x;
		solver->get_info(stats.solver_info);
	}

	void IncompressibleElasticVarForm::solve_static_linear(Eigen::MatrixXd &sol)
	{
		auto solver = polysolve::linear::Solver::create(args["solver"]["linear"], logger());
		logger().info("{}...", solver->name());
		rhs_assembler->set_bc(boundary.local_boundary, boundary.boundary_nodes, n_boundary_samples(), boundary.local_neumann_boundary, rhs);
		StiffnessMatrix A;
		build_stiffness_mat(A);
		Eigen::VectorXd b = rhs;
		solve_linear_system(solver, A, b, args["output"]["advanced"]["spectrum"], sol);
	}

	void IncompressibleElasticVarForm::solve_transient_linear(Eigen::MatrixXd &sol)
	{
		auto solver = polysolve::linear::Solver::create(args["solver"]["linear"], logger());
		logger().info("{}...", solver->name());

		Eigen::MatrixXd displacement, pressure;
		split_solution(sol, displacement, pressure);
		auto bdf = std::make_shared<time_integrator::BDF>();
		bdf->set_parameters(args["time"]["integrator"]);
		bdf->init(
			displacement,
			Eigen::MatrixXd::Zero(displacement.rows(), displacement.cols()),
			Eigen::MatrixXd::Zero(displacement.rows(), displacement.cols()),
			dt);
		time_integrator = bdf;

		save_timestep(t0, 0, t0, dt, sol);

		Eigen::MatrixXd current_rhs = rhs;
		StiffnessMatrix stiffness, expanded_mass;
		build_stiffness_mat(stiffness);
		expand_primary_matrix(stacked_ndof(), mass, expanded_mass);

		for (int t = 1; t <= time_steps; ++t)
		{
			const double time = t0 + t * dt;
			rhs_assembler->compute_energy_grad(
				boundary.local_boundary, boundary.boundary_nodes, mass_matrix_assembler->density(), n_boundary_samples(), boundary.local_neumann_boundary, rhs, time,
				current_rhs);
			rhs_assembler->set_bc(
				boundary.local_boundary, boundary.boundary_nodes, n_boundary_samples(), boundary.local_neumann_boundary, current_rhs, displacement, time);

			if (current_rhs.rows() != stacked_ndof())
			{
				const int old_rows = current_rhs.rows();
				current_rhs.conservativeResize(stacked_ndof(), current_rhs.cols());
				if (stacked_ndof() > old_rows)
					current_rhs.bottomRows(stacked_ndof() - old_rows).setZero();
			}
			current_rhs.bottomRows(incompressible_spaces.pressure.n_bases).setZero();

			StiffnessMatrix A = expanded_mass / bdf->beta_dt() + stiffness;
			Eigen::VectorXd b = Eigen::VectorXd::Zero(stacked_ndof());
			b.head(primary_ndof()) = (mass * bdf->weighted_sum_x_prevs()) / bdf->beta_dt();
			for (int i : boundary.boundary_nodes)
				b[i] = 0;
			b += current_rhs;

			solve_linear_system(solver, A, b, args["output"]["advanced"]["spectrum"].get<bool>() && t == time_steps, sol);
			split_solution(sol, displacement, pressure);
			bdf->update_quantities(displacement.col(0));

			save_timestep(time, t, t0, dt, sol);
			save_elastic_step_state(t0, dt, t, time_integrator.get());
			logger().info("{}/{}  t={}", t, time_steps, time);
			notify_time_step(t);
		}
	}

	void IncompressibleElasticVarForm::solve_problem(Eigen::MatrixXd &sol)
	{
		stats.spectrum.setZero();
		igl::Timer timer;
		timer.start();
		logger().info("Solving {}", assembler->name());
		prepare_initial_solution(sol);
		if (problem->is_time_dependent())
			solve_transient_linear(sol);
		else
		{
			time_integrator = nullptr;
			solve_static_linear(sol);
		}
		timer.stop();
		timings.solving_time = timer.getElapsedTime();
		logger().info(" took {}s", timings.solving_time);
	}

	std::vector<io::OutputField> IncompressibleElasticVarForm::output_fields(
		const io::OutputSample &sample,
		const Eigen::MatrixXd &solution,
		const io::OutputFieldOptions &options) const
	{
		Eigen::MatrixXd displacement, pressure;
		split_solution(solution, displacement, pressure);
		const std::vector<std::pair<std::string, std::shared_ptr<solver::Form>>> named_forms;
		auto fields = elastic_output_fields(
			sample, displacement, options, nullptr, time_integrator.get(), named_forms, nullptr);
		const bool export_pressure_gradient =
			!options.fields.empty() && options.export_field("pressure_gradient");
		if (mesh_ && (options.export_field("pressure") || export_pressure_gradient))
		{
			Eigen::MatrixXd values, gradients;
			if (sample_scalar_field(
					*mesh_, *incompressible_spaces.pressure.bases, geom_bases(), sample, pressure, values,
					export_pressure_gradient ? &gradients : nullptr))
			{
				if (options.export_field("pressure"))
					fields.push_back({"pressure", values, io::OutputField::Association::Point});
				if (export_pressure_gradient)
					fields.push_back({"pressure_gradient", gradients, io::OutputField::Association::Point});
			}
		}
		return fields;
	}
} // namespace polyfem::varform
