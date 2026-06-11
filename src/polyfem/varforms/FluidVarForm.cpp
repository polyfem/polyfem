#include "FluidVarForm.hpp"

#include <polyfem/assembler/AssemblerUtils.hpp>
#include <polyfem/assembler/GenericProblem.hpp>
#include <polyfem/assembler/Stokes.hpp>
#include <polyfem/basis/LagrangeBasis2d.hpp>
#include <polyfem/basis/LagrangeBasis3d.hpp>
#include <polyfem/io/Evaluator.hpp>
#include <polyfem/mesh/mesh2D/Mesh2D.hpp>
#include <polyfem/mesh/mesh3D/Mesh3D.hpp>
#include <polyfem/problem/KernelProblem.hpp>
#include <polyfem/problem/ProblemFactory.hpp>
#include <polyfem/solver/NavierStokesSolver.hpp>
#include <polyfem/solver/TransientNavierStokesSolver.hpp>
#include <polyfem/time_integrator/BDF.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/MatrixUtils.hpp>
#include <polyfem/utils/Timer.hpp>
#include <polyfem/varforms/VarFormUtils.hpp>

#include <polysolve/linear/FEMSolver.hpp>

namespace polyfem::varform
{
	using namespace varform::internal;

	FluidVarForm::FluidVarForm(const std::string &formulation, const Units &units, const json &args, const std::string &out_path)
		: VarForm(units, args, out_path)
	{
		const bool is_time_dependent = args.contains("time") && !args["time"].is_null();

		assembler = assembler::AssemblerUtils::make_assembler(formulation);
		mass_matrix_assembler = std::make_shared<assembler::Mass>();
		pure_mass_matrix_assembler = std::make_shared<assembler::HRZMass>();
		mixed_assembler = assembler::AssemblerUtils::make_mixed_assembler(formulation);
		pressure_assembler = assembler::AssemblerUtils::make_assembler(assembler::AssemblerUtils::other_assembler_name(formulation));

		if (!args.contains("preset_problem"))
		{
			problem = std::make_shared<assembler::GenericTensorProblem>("GenericTensor");
			problem->clear();

			json tmp;
			tmp["is_time_dependent"] = is_time_dependent;
			problem->set_parameters(tmp, root_path);

			auto bc = args["boundary_conditions"];
			bc["root_path"] = root_path;
			problem->set_parameters(bc, root_path);
			problem->set_parameters(args["initial_conditions"], root_path);
			problem->set_parameters(args["output"], root_path);
		}
		else
		{
			if (args["preset_problem"]["type"] == "Kernel")
			{
				problem = std::make_shared<problem::KernelProblem>("Kernel", *assembler);
				problem->clear();
			}
			else
			{
				problem = problem::ProblemFactory::factory().get_problem(args["preset_problem"]["type"]);
				problem->clear();
			}
			problem->set_parameters(args["preset_problem"], root_path);
		}

		problem->set_units(*assembler, units);

		t0 = is_time_dependent ? args["time"]["t0"].get<double>() : 0.0;
		time_steps = is_time_dependent ? args["time"]["time_steps"].get<int>() : 0;
		dt = is_time_dependent ? args["time"]["dt"].get<double>() : 0.0;

		assert(assembler->is_fluid());
	}

	void FluidVarForm::save_json(const Eigen::MatrixXd &solution, std::ostream &out) const
	{
		save_json_stats(solution, fluid_spaces.pressure.n_bases, out);
	}

	void FluidVarForm::load_mesh(const mesh::Mesh &mesh, const json &args)
	{
		VarForm::load_mesh(mesh, args);
		if (mixed_assembler)
			mixed_assembler->set_size(mesh.dimension());
		if (pressure_assembler)
			set_materials(*pressure_assembler);
	}

	int FluidVarForm::primary_ndof() const
	{
		return mesh_ ? velocity_space.n_bases * mesh_->dimension() : 0;
	}

	int FluidVarForm::pressure_block_size() const
	{
		return fluid_spaces.pressure.n_bases + ((use_avg_pressure && assembler && assembler->is_fluid()) ? 1 : 0);
	}

	int FluidVarForm::stacked_ndof() const
	{
		return fluid_spaces.layout.total_dof();
	}

	void FluidVarForm::build_rhs_assembler()
	{
		json rhs_solver_params = args["solver"]["linear"];
		if (!rhs_solver_params.contains("Pardiso"))
			rhs_solver_params["Pardiso"] = {};
		rhs_solver_params["Pardiso"]["mtype"] = -2;

		rhs_assembler = std::make_shared<assembler::RhsAssembler>(
			*assembler, *mesh_, nullptr, // no obtacle for the rhs assembler
			boundary.dirichlet_nodes, boundary.neumann_nodes,
			boundary.dirichlet_nodes_position, boundary.neumann_nodes_position,
			velocity_space.n_bases, mesh_->dimension(), velocity_space.bases, geom_bases(), velocity_caches.mass, *problem,
			args["space"]["advanced"]["bc_method"],
			rhs_solver_params);
	}

	void FluidVarForm::build_basis(mesh::Mesh &mesh, const bool iso_parametric, const json &args)
	{
		VarForm::build_basis(mesh, iso_parametric, args);

		if (velocity_space.disc_orders.maxCoeff() != velocity_space.disc_orders.minCoeff())
			log_and_throw_error("p refinement not supported in mixed formulation!");

		igl::Timer timer;
		timer.start();

		const auto &all_boundary = boundary.total_local_boundary;
		const int prev_bases = velocity_space.n_bases;
		const int prev_b_size = int(all_boundary.size());
		const bool has_polys = mesh.has_poly();
		const bool use_corner_quadrature = args["space"]["advanced"]["use_corner_quadrature"];
		const int quadrature_order = args["space"]["advanced"]["quadrature_order"].get<int>();
		const int mass_quadrature_order = args["space"]["advanced"]["mass_quadrature_order"].get<int>();
		const int order = args["space"]["pressure_discr_order"];
		std::vector<mesh::LocalBoundary> pressure_local_boundary;
		std::map<int, basis::InterfaceData> pressure_poly_edge_to_data;

		fluid_spaces.pressure.bases.clear();
		fluid_spaces.pressure.n_bases = 0;
		if (mesh.is_volume())
		{
			const mesh::Mesh3D &tmp_mesh = dynamic_cast<const mesh::Mesh3D &>(mesh);
			fluid_spaces.pressure.n_bases = basis::LagrangeBasis3d::build_bases(
				tmp_mesh, assembler->name(), quadrature_order, mass_quadrature_order,
				order, order,
				args["space"]["basis_type"] == "Bernstein", false,
				has_polys, false, use_corner_quadrature,
				fluid_spaces.pressure.bases, pressure_local_boundary, pressure_poly_edge_to_data, fluid_spaces.pressure.mesh_nodes);
		}
		else
		{
			const mesh::Mesh2D &tmp_mesh = dynamic_cast<const mesh::Mesh2D &>(mesh);
			fluid_spaces.pressure.n_bases = basis::LagrangeBasis2d::build_bases(
				tmp_mesh, assembler->name(), quadrature_order, mass_quadrature_order,
				order,
				args["space"]["basis_type"] == "Bernstein", false,
				has_polys, false, use_corner_quadrature,
				fluid_spaces.pressure.bases, pressure_local_boundary, pressure_poly_edge_to_data, fluid_spaces.pressure.mesh_nodes);
		}

		assert(velocity_space.bases.size() == fluid_spaces.pressure.bases.size());
		for (int i = 0; i < int(fluid_spaces.pressure.bases.size()); ++i)
		{
			quadrature::Quadrature b_quad;
			velocity_space.bases[i].compute_quadrature(b_quad);
			fluid_spaces.pressure.bases[i].set_quadrature([b_quad](quadrature::Quadrature &quad) { quad = b_quad; });
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
			mesh, velocity_space.n_bases,
			velocity_space.bases, geom_bases(), fluid_spaces.pressure.bases,
			boundary.local_boundary,
			boundary.boundary_nodes,
			boundary.local_neumann_boundary,
			boundary.local_pressure_boundary,
			boundary.local_pressure_cavity,
			boundary.pressure_boundary_nodes,
			boundary.dirichlet_nodes, boundary.neumann_nodes);

		rebuild_node_positions(velocity_space.bases, boundary.dirichlet_nodes, boundary.dirichlet_nodes_position);
		rebuild_node_positions(velocity_space.bases, boundary.neumann_nodes, boundary.neumann_nodes_position);

		const bool has_neumann = !boundary.local_neumann_boundary.empty() || int(boundary.local_boundary.size()) < prev_b_size;
		use_avg_pressure = !has_neumann;
		fluid_spaces.pressure.geometry = geometry_mapping;
		fluid_spaces.pressure.value_dim = 1;
		fluid_spaces.layout = SolutionLayout();
		fluid_spaces.velocity_block = fluid_spaces.layout.add_block(primary_ndof(), problem->is_time_dependent());
		fluid_spaces.pressure_block = fluid_spaces.layout.add_block(fluid_spaces.pressure.n_bases, false);
		fluid_spaces.pressure_mean_constraint_block = -1;
		if (use_avg_pressure && assembler && assembler->is_fluid())
			fluid_spaces.pressure_mean_constraint_block = fluid_spaces.layout.add_block(1, false, true);

		for (int i = prev_bases; i < velocity_space.n_bases; ++i)
			for (int d = 0; d < mesh.dimension(); ++d)
				boundary.boundary_nodes.push_back(i * mesh.dimension() + d);

		std::sort(boundary.boundary_nodes.begin(), boundary.boundary_nodes.end());
		boundary.boundary_nodes.erase(std::unique(boundary.boundary_nodes.begin(), boundary.boundary_nodes.end()), boundary.boundary_nodes.end());

		if (velocity_space.n_bases <= args["solver"]["advanced"]["cache_size"])
			pressure_ass_vals_cache.init(mesh.is_volume(), fluid_spaces.pressure.bases, geom_bases());
		else
			pressure_ass_vals_cache.init_empty();

		build_rhs_assembler();

		timer.stop();
		timings.building_basis_time += timer.getElapsedTime();
		logger().info("n pressure bases: {}", fluid_spaces.pressure.n_bases);
	}

	void FluidVarForm::assemble_rhs(const mesh::Mesh &mesh, const json &args)
	{
		build_rhs_assembler();
		VarForm::assemble_rhs(mesh, args);

		const int prev_size = rhs.rows();
		rhs.conservativeResize(prev_size + pressure_block_size(), rhs.cols());
		rhs.bottomRows(pressure_block_size()).setZero();
	}

	void FluidVarForm::assemble_mass_mat(const mesh::Mesh &mesh, const json &args)
	{
		if (!problem->is_time_dependent())
		{
			avg_mass = 1;
			timings.assembling_mass_mat_time = 0;
			if (!assembler->is_linear())
				pure_mass_matrix_assembler->assemble(mesh.is_volume(), velocity_space.n_bases, velocity_space.bases, geom_bases(), velocity_caches.pure_mass, 0, pure_mass, true);
			return;
		}

		mass.resize(0, 0);
		igl::Timer timer;
		timer.start();
		logger().info("Assembling mass mat...");

		StiffnessMatrix velocity_mass;
		mass_matrix_assembler->assemble(mesh.is_volume(), velocity_space.n_bases, velocity_space.bases, geom_bases(), velocity_caches.mass, 0, velocity_mass, true);
		if (!assembler->is_linear())
			pure_mass_matrix_assembler->assemble(mesh.is_volume(), velocity_space.n_bases, velocity_space.bases, geom_bases(), velocity_caches.pure_mass, 0, pure_mass, true);

		std::vector<Eigen::Triplet<double>> blocks;
		blocks.reserve(velocity_mass.nonZeros());
		for (int k = 0; k < velocity_mass.outerSize(); ++k)
			for (StiffnessMatrix::InnerIterator it(velocity_mass, k); it; ++it)
				blocks.emplace_back(it.row(), it.col(), it.value());

		mass.resize(primary_ndof(), primary_ndof());
		mass.setFromTriplets(blocks.begin(), blocks.end());
		mass.makeCompressed();

		avg_mass = 0;
		for (int k = 0; k < velocity_mass.outerSize(); ++k)
			for (StiffnessMatrix::InnerIterator it(velocity_mass, k); it; ++it)
			{
				assert(it.col() == k);
				avg_mass += it.value();
			}
		avg_mass /= std::max(1, int(velocity_mass.rows()));
		logger().info("average mass {}", avg_mass);

		if (args["solver"]["advanced"]["lump_mass_matrix"])
			mass = utils::lump_matrix(mass);

		timer.stop();
		timings.assembling_mass_mat_time = timer.getElapsedTime();
		logger().info(" took {}s", timings.assembling_mass_mat_time);

		stats.nn_zero = mass.nonZeros();
		stats.num_dofs = mass.rows();
		stats.mat_size = (long long)mass.rows() * (long long)mass.cols();
		logger().info("sparsity: {}/{}", stats.nn_zero, stats.mat_size);
	}

	void FluidVarForm::prepare_initial_solution(Eigen::MatrixXd &sol) const
	{
		if (sol.size() <= 0)
			initial_solution(sol);
		if (sol.cols() > 1)
			sol.conservativeResize(Eigen::NoChange, 1);
		sol.conservativeResize(stacked_ndof(), sol.cols());
		sol.bottomRows(pressure_block_size()).setZero();
	}

	void FluidVarForm::split_solution(const Eigen::MatrixXd &stacked, Eigen::MatrixXd &primary, Eigen::MatrixXd &pressure) const
	{
		const int cols = std::max(1, int(stacked.cols()));
		primary.setZero(primary_ndof(), cols);
		pressure.setZero(fluid_spaces.pressure.n_bases, cols);

		const int primary_rows = std::min(primary_ndof(), int(stacked.rows()));
		if (primary_rows > 0)
			primary.topRows(primary_rows) = stacked.topRows(primary_rows);

		if (stacked.rows() > primary_ndof())
		{
			const int pressure_rows = std::min(fluid_spaces.pressure.n_bases, int(stacked.rows()) - primary_ndof());
			if (pressure_rows > 0)
				pressure.topRows(pressure_rows) = stacked.middleRows(primary_ndof(), pressure_rows);
		}
	}

	void FluidVarForm::build_stiffness_mat(StiffnessMatrix &stiffness)
	{
		igl::Timer timer;
		timer.start();
		logger().info("Assembling stiffness mat...");

		StiffnessMatrix velocity_stiffness, mixed_stiffness, pressure_stiffness;
		assembler->assemble(mesh_->is_volume(), velocity_space.n_bases, velocity_space.bases, geom_bases(), velocity_caches.values, 0, velocity_stiffness);
		mixed_assembler->assemble(mesh_->is_volume(), fluid_spaces.pressure.n_bases, velocity_space.n_bases, fluid_spaces.pressure.bases, velocity_space.bases, geom_bases(), pressure_ass_vals_cache, velocity_caches.values, 0, mixed_stiffness);
		pressure_assembler->assemble(mesh_->is_volume(), fluid_spaces.pressure.n_bases, fluid_spaces.pressure.bases, geom_bases(), pressure_ass_vals_cache, 0, pressure_stiffness);

		assembler::AssemblerUtils::merge_mixed_matrices(
			velocity_space.n_bases, fluid_spaces.pressure.n_bases, mesh_->dimension(), use_avg_pressure,
			velocity_stiffness, mixed_stiffness, pressure_stiffness, stiffness);

		timer.stop();
		timings.assembling_stiffness_mat_time = timer.getElapsedTime();
		logger().info(" took {}s", timings.assembling_stiffness_mat_time);

		stats.nn_zero = stiffness.nonZeros();
		stats.num_dofs = stiffness.rows();
		stats.mat_size = (long long)stiffness.rows() * (long long)stiffness.cols();
		logger().info("sparsity: {}/{}", stats.nn_zero, stats.mat_size);

		write_matrix_market(args, stiffness);
	}

	void FluidVarForm::solve_linear_system(
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
			/*is_fluid=*/true,
			use_avg_pressure);

		sol = x;
		solver->get_info(stats.solver_info);

		const double error = (A * x - b).norm();
		if (error > 1e-4)
			logger().error("Solver error: {}", error);
		else
			logger().debug("Solver error: {}", error);
	}

	std::vector<io::OutputField> FluidVarForm::output_fields(
		const io::OutputSample &sample,
		const Eigen::MatrixXd &solution,
		const io::OutputFieldOptions &options) const
	{
		std::vector<io::OutputField> fields = common_output_fields(sample, solution, options);
		if (!mesh_ || !problem || solution.size() <= 0)
			return fields;

		Eigen::MatrixXd velocity, pressure;
		split_solution(solution, velocity, pressure);

		const int field_dim = mesh_->dimension();
		const bool has_element_samples = sample.local_points.rows() > 0 && sample.local_points.rows() == sample.element_ids.size();
		const int output_rows = sample.points.rows() > 0 ? sample.points.rows() : std::max<int>(sample.local_points.rows(), sample.node_ids.size());
		const bool export_solution_gradient =
			!options.fields.empty() && options.export_field("solution_gradient");
		const bool export_pressure_gradient =
			!options.fields.empty() && options.export_field("pressure_gradient");

		const auto resize_to_output_rows = [&](Eigen::MatrixXd &values) {
			if (output_rows <= values.rows())
				return;

			const int previous_rows = values.rows();
			values.conservativeResize(output_rows, values.cols());
			values.bottomRows(output_rows - previous_rows).setZero();
		};

		const auto sample_vector_field = [&](const Eigen::MatrixXd &dof_values, Eigen::MatrixXd &values, Eigen::MatrixXd *gradients = nullptr) -> bool {
			if (dof_values.size() <= 0 || field_dim <= 0)
				return false;

			if (has_element_samples)
			{
				values.resize(sample.local_points.rows(), field_dim);
				if (gradients)
					gradients->resize(sample.local_points.rows(), field_dim * mesh_->dimension());
				for (int i = 0; i < sample.local_points.rows(); ++i)
				{
					const int element_id = sample.element_ids(i);
					if (element_id < 0)
					{
						values.row(i).setZero();
						if (gradients)
							gradients->row(i).setZero();
						continue;
					}

					Eigen::MatrixXd local_sol, local_grad;
					io::Evaluator::interpolate_at_local_vals(
						*mesh_, field_dim, velocity_space.bases, geom_bases(),
						element_id, sample.local_points.row(i), dof_values, local_sol, local_grad);

					for (int d = 0; d < field_dim; ++d)
						values(i, d) = local_sol(d);
					if (gradients)
						gradients->row(i) = local_grad;
				}

				resize_to_output_rows(values);
				if (gradients)
					resize_to_output_rows(*gradients);
				return true;
			}

			if (sample.node_ids.size() > 0)
			{
				values.resize(sample.node_ids.size(), field_dim);
				for (int i = 0; i < sample.node_ids.size(); ++i)
				{
					const int node_id = sample.node_ids(i);
					for (int d = 0; d < field_dim; ++d)
					{
						const int dof = node_id * field_dim + d;
						if (dof < 0 || dof >= dof_values.rows())
							return false;
						values(i, d) = dof_values(dof);
					}
				}
				return sample.points.rows() == 0 || sample.points.rows() == values.rows();
			}

			return false;
		};

		Eigen::MatrixXd velocity_values, velocity_gradients;
		const bool sampled_velocity = sample_vector_field(
			velocity, velocity_values,
			export_solution_gradient ? &velocity_gradients : nullptr);
		if (sampled_velocity && options.export_field("velocity"))
			fields.push_back({"velocity", velocity_values, io::OutputField::Association::Point});
		if (sampled_velocity && options.export_field("solution"))
			fields.push_back({"solution", velocity_values, io::OutputField::Association::Point});
		if (sampled_velocity && export_solution_gradient)
			fields.push_back({"solution_gradient", velocity_gradients, io::OutputField::Association::Point});

		if (mesh_ && (options.export_field("pressure") || export_pressure_gradient))
		{
			Eigen::MatrixXd values, gradients;
			if (sample_scalar_field(
					*mesh_, fluid_spaces.pressure.bases, geom_bases(), sample, pressure, values,
					export_pressure_gradient ? &gradients : nullptr))
			{
				if (options.export_field("pressure"))
					fields.push_back({"pressure", values, io::OutputField::Association::Point});
				if (export_pressure_gradient)
					fields.push_back({"pressure_gradient", gradients, io::OutputField::Association::Point});
			}
		}

		const auto &paraview_options = args["output"]["paraview"]["options"];
		if (paraview_options["material"] && has_element_samples)
		{
			const auto &params = assembler->parameters();
			std::map<std::string, Eigen::MatrixXd> param_values;
			for (const auto &[p, _] : params)
				param_values[p].setZero(output_rows, 1);

			Eigen::MatrixXd rhos = Eigen::MatrixXd::Zero(output_rows, 1);
			const auto &density = mass_matrix_assembler->density();
			for (int i = 0; i < sample.local_points.rows(); ++i)
			{
				const int element_id = sample.element_ids(i);
				if (element_id < 0)
					continue;

				for (const auto &[p, func] : params)
					param_values.at(p)(i) = func(sample.local_points.row(i), sample.points.row(i), sample.time, element_id);
				rhos(i) = density(sample.local_points.row(i), sample.points.row(i), sample.time, element_id);
			}

			for (const auto &[name, values] : param_values)
				if (options.export_field(name))
					fields.push_back({name, values, io::OutputField::Association::Point});
			if (options.export_field("rho"))
				fields.push_back({"rho", rhos, io::OutputField::Association::Point});
		}

		if (paraview_options["body_ids"] && options.export_field("body_ids") && has_element_samples)
		{
			Eigen::MatrixXd ids = Eigen::MatrixXd::Zero(output_rows, 1);
			for (int i = 0; i < sample.element_ids.size(); ++i)
			{
				const int element_id = sample.element_ids(i);
				if (element_id >= 0)
					ids(i) = mesh_->get_body_id(element_id);
			}
			fields.push_back({"body_ids", ids, io::OutputField::Association::Point});
		}

		return fields;
	}

	void StokesVarForm::solve_static_linear(Eigen::MatrixXd &sol)
	{
		auto solver = polysolve::linear::Solver::create(args["solver"]["linear"], logger());
		logger().info("{}...", solver->name());

		rhs_assembler->set_bc(
			boundary.local_boundary, boundary.boundary_nodes, n_boundary_samples(),
			boundary.local_neumann_boundary, rhs);

		StiffnessMatrix A;
		build_stiffness_mat(A);
		Eigen::VectorXd b = rhs;
		solve_linear_system(solver, A, b, args["output"]["advanced"]["spectrum"], sol);
	}

	void StokesVarForm::solve_transient_linear(Eigen::MatrixXd &sol)
	{
		auto solver = polysolve::linear::Solver::create(args["solver"]["linear"], logger());
		logger().info("{}...", solver->name());

		Eigen::MatrixXd velocity, pressure;
		split_solution(sol, velocity, pressure);

		auto bdf = std::make_shared<time_integrator::BDF>();
		bdf->set_parameters(args["time"]["integrator"]);
		bdf->init(
			velocity,
			Eigen::MatrixXd::Zero(velocity.rows(), velocity.cols()),
			Eigen::MatrixXd::Zero(velocity.rows(), velocity.cols()),
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
				boundary.local_boundary, boundary.boundary_nodes, n_boundary_samples(), boundary.local_neumann_boundary, current_rhs, velocity, time);

			if (current_rhs.rows() != stacked_ndof())
			{
				const int old_rows = current_rhs.rows();
				current_rhs.conservativeResize(stacked_ndof(), current_rhs.cols());
				if (stacked_ndof() > old_rows)
					current_rhs.bottomRows(stacked_ndof() - old_rows).setZero();
			}
			current_rhs.bottomRows(pressure_block_size()).setZero();

			StiffnessMatrix A = expanded_mass / bdf->beta_dt() + stiffness;
			Eigen::VectorXd b = Eigen::VectorXd::Zero(stacked_ndof());
			b.head(primary_ndof()) = (mass * bdf->weighted_sum_x_prevs()) / bdf->beta_dt();
			for (int i : boundary.boundary_nodes)
				b[i] = 0;
			b += current_rhs;

			solve_linear_system(solver, A, b, args["output"]["advanced"]["spectrum"].get<bool>() && t == time_steps, sol);
			split_solution(sol, velocity, pressure);
			bdf->update_quantities(velocity.col(0));

			save_timestep(time, t, t0, dt, sol);
			save_step_state(t0, dt, t, time_integrator.get());
			logger().info("{}/{}  t={}", t, time_steps, time);
			notify_time_step(t);
		}
	}

	void StokesVarForm::solve_problem(Eigen::MatrixXd &sol)
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

	void NavierStokesVarForm::solve_static(Eigen::MatrixXd &sol)
	{
		assert(rhs_assembler != nullptr);
		rhs_assembler->set_bc(
			boundary.local_boundary, boundary.boundary_nodes, n_boundary_samples(), boundary.local_neumann_boundary, rhs);

		auto velocity_stokes_assembler = std::make_shared<assembler::StokesVelocity>();
		set_materials(*velocity_stokes_assembler);

		Eigen::VectorXd x;
		solver::NavierStokesSolver ns_solver(args["solver"]);
		ns_solver.minimize(
			velocity_space.n_bases, fluid_spaces.pressure.n_bases,
			velocity_space.bases, fluid_spaces.pressure.bases,
			geom_bases(),
			*velocity_stokes_assembler,
			*dynamic_cast<assembler::NavierStokesVelocity *>(assembler.get()),
			*mixed_assembler,
			*pressure_assembler,
			velocity_caches.values,
			pressure_ass_vals_cache,
			boundary.boundary_nodes,
			use_avg_pressure,
			mesh_->dimension(),
			mesh_->is_volume(),
			rhs,
			x);
		sol = x;
		ns_solver.get_info(stats.solver_info);
	}

	void NavierStokesVarForm::solve_transient(Eigen::MatrixXd &sol)
	{

		Eigen::MatrixXd velocity, pressure;
		split_solution(sol, velocity, pressure);

		auto bdf = std::make_shared<time_integrator::BDF>();
		if (args["time"]["integrator"].is_object() && args["time"]["integrator"]["type"] == "BDF")
			bdf->set_parameters(args["time"]["integrator"]);
		bdf->init(
			velocity,
			Eigen::MatrixXd::Zero(velocity.rows(), velocity.cols()),
			Eigen::MatrixXd::Zero(velocity.rows(), velocity.cols()),
			dt);
		time_integrator = bdf;

		save_timestep(t0, 0, t0, dt, sol);

		Eigen::MatrixXd current_rhs = rhs;
		StiffnessMatrix velocity_mass;
		mass_matrix_assembler->assemble(mesh_->is_volume(), velocity_space.n_bases, velocity_space.bases, geom_bases(), velocity_caches.mass, 0, velocity_mass, true);

		StiffnessMatrix velocity_stiffness, mixed_stiffness, pressure_stiffness;
		auto velocity_stokes_assembler = std::make_shared<assembler::StokesVelocity>();
		set_materials(*velocity_stokes_assembler);

		mixed_assembler->assemble(mesh_->is_volume(), fluid_spaces.pressure.n_bases, velocity_space.n_bases, fluid_spaces.pressure.bases, velocity_space.bases, geom_bases(), pressure_ass_vals_cache, velocity_caches.values, 0, mixed_stiffness);
		pressure_assembler->assemble(mesh_->is_volume(), fluid_spaces.pressure.n_bases, fluid_spaces.pressure.bases, geom_bases(), pressure_ass_vals_cache, 0, pressure_stiffness);

		solver::TransientNavierStokesSolver ns_solver(args["solver"]);
		for (int t = 1; t <= time_steps; ++t)
		{
			const double time = t0 + t * dt;
			velocity_stokes_assembler->assemble(mesh_->is_volume(), velocity_space.n_bases, velocity_space.bases, geom_bases(), velocity_caches.values, time, velocity_stiffness);

			logger().info("{}/{} steps, dt={}s t={}s", t, time_steps, dt, time);

			const Eigen::VectorXd prev_sol = bdf->weighted_sum_x_prevs();
			rhs_assembler->compute_energy_grad(
				boundary.local_boundary, boundary.boundary_nodes, mass_matrix_assembler->density(), n_boundary_samples(), boundary.local_neumann_boundary, rhs, time, current_rhs);
			rhs_assembler->set_bc(
				boundary.local_boundary, boundary.boundary_nodes, n_boundary_samples(), boundary.local_neumann_boundary, current_rhs, velocity, time);

			if (current_rhs.rows() != stacked_ndof())
			{
				const int old_rows = current_rhs.rows();
				current_rhs.conservativeResize(stacked_ndof(), current_rhs.cols());
				if (stacked_ndof() > old_rows)
					current_rhs.bottomRows(stacked_ndof() - old_rows).setZero();
			}
			current_rhs.bottomRows(pressure_block_size()).setZero();

			Eigen::VectorXd tmp_sol;
			ns_solver.minimize(
				velocity_space.n_bases, fluid_spaces.pressure.n_bases,
				time,
				velocity_space.bases, geom_bases(),
				*dynamic_cast<assembler::NavierStokesVelocity *>(assembler.get()),
				velocity_caches.values,
				boundary.boundary_nodes,
				use_avg_pressure,
				mesh_->dimension(),
				mesh_->is_volume(),
				std::sqrt(bdf->acceleration_scaling()), prev_sol, velocity_stiffness, mixed_stiffness,
				pressure_stiffness, velocity_mass, current_rhs, tmp_sol);
			sol = tmp_sol;
			split_solution(sol, velocity, pressure);
			bdf->update_quantities(velocity.col(0));
			save_timestep(time, t, t0, dt, sol);
			save_step_state(t0, dt, t, time_integrator.get());
			notify_time_step(t);
		}

		ns_solver.get_info(stats.solver_info);
	}

	void NavierStokesVarForm::solve_problem(Eigen::MatrixXd &sol)
	{
		stats.spectrum.setZero();
		igl::Timer timer;
		timer.start();
		logger().info("Solving {}", assembler->name());

		prepare_initial_solution(sol);
		if (problem->is_time_dependent())
			solve_transient(sol);
		else
		{
			time_integrator = nullptr;
			solve_static(sol);
		}

		timer.stop();
		timings.solving_time = timer.getElapsedTime();
		logger().info(" took {}s", timings.solving_time);
	}
} // namespace polyfem::varform
