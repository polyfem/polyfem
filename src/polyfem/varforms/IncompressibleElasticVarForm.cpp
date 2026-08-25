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
#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/MatrixUtils.hpp>
#include <polyfem/utils/Timer.hpp>
#include <polyfem/varforms/VarFormUtils.hpp>

#include <polysolve/linear/FEMSolver.hpp>

namespace polyfem::varform
{
	using namespace varform::internal;

	void IncompressibleElasticVarForm::reset()
	{
		ElasticVarForm::reset();
		pressure_space_.reset();
		displacement_space_id_ = -1;
		pressure_space_id_ = -1;
		pressure_boundary_.reset();
		pressure_ass_vals_cache_.init_empty();
		mixed_assembler_ = nullptr;
		pressure_assembler_ = nullptr;
		time_integrator = nullptr;
	}

	void IncompressibleElasticVarForm::init(const std::string &formulation, const Units &units, const json &args, const std::string &out_path)
	{
		ElasticVarForm::init(formulation, units, args, out_path);
		const json &discr_orders = args.at("space").at("discr_order");

		const json &materials = args.at("materials");
		if (materials.is_array() && materials.empty())
			log_and_throw_error("Incompressible elasticity requires at least one material.");
		const json &first_material = materials.is_array() ? materials.at(0) : materials;
		displacement_space_id_ = first_material.at("displacement_space_id").get<int>();
		pressure_space_id_ = first_material.at("pressure_space_id").get<int>();
		if (displacement_space_id_ == pressure_space_id_)
			log_and_throw_error("Incompressible displacement and pressure must use different FE space IDs.");

		if (discr_orders.is_array())
		{
			bool has_displacement_space = false;
			bool has_pressure_space = false;
			for (const json &entry : discr_orders)
			{
				const int fe_space_id = entry.at("fe_space").get<int>();
				has_displacement_space |= fe_space_id == displacement_space_id_;
				has_pressure_space |= fe_space_id == pressure_space_id_;
			}
			if (!has_displacement_space || !has_pressure_space)
				log_and_throw_error("Incompressible discretization-order lists must explicitly name the displacement and pressure FE spaces.");
		}

		if (materials.is_array())
		{
			for (const json &material : materials)
			{
				if (material.at("displacement_space_id").get<int>() != displacement_space_id_
					|| material.at("pressure_space_id").get<int>() != pressure_space_id_)
					log_and_throw_error("All incompressible materials must use the same displacement and pressure FE space IDs.");
			}
		}

		mixed_assembler_ = assembler::AssemblerUtils::make_mixed_assembler(formulation);
		pressure_assembler_ = assembler::AssemblerUtils::make_assembler(assembler::AssemblerUtils::other_assembler_name(formulation));
		assert(primary_assembler_->is_linear());
		assert(primary_assembler_->is_tensor());
	}

	void IncompressibleElasticVarForm::save_json(const Eigen::MatrixXd &solution, std::ostream &out) const
	{
		if (!mesh_)
		{
			logger().error("Load the mesh first!");
			return;
		}
		if (solution.size() <= 0)
		{
			logger().error("Solve the problem first!");
			return;
		}

		logger().info("Saving json...");
		const int primary_size = primary_ndof();
		const Eigen::MatrixXd stats_solution =
			solution.rows() >= primary_size
				? solution.topRows(primary_size).eval()
				: solution;

		nlohmann::json j;
		stats.save_json(
			args, space_.n_bases, pressure_space_.n_bases,
			stats_solution, *mesh_, space_.disc_orders, space_.disc_ordersq, *problem,
			timings, primary_assembler_ ? primary_assembler_->name() : name(), space_.is_iso_parametric(),
			args["output"]["advanced"]["sol_at_node"], j);
		out << j.dump(4) << std::endl;
	}

	io::OutStatsData IncompressibleElasticVarForm::compute_errors(const Eigen::MatrixXd &solution)
	{
		if (!args["output"]["advanced"]["compute_error"])
			return stats;

		double tend = 0;
		if (!args["time"].is_null())
			tend = args["time"]["tend"];

		Eigen::MatrixXd displacement, pressure;
		split_solution(solution, displacement, pressure);
		stats.compute_errors(space_.n_bases, space_.basis_list(), space_.geometry_basis_list(), *mesh_, *problem, tend, displacement);
		return stats;
	}

	void IncompressibleElasticVarForm::load_mesh(const mesh::Mesh &mesh, const json &args)
	{
		ElasticVarForm::load_mesh(mesh, args);
		if (mixed_assembler_)
			mixed_assembler_->set_size(mesh.dimension());
		if (pressure_assembler_)
			set_materials(*pressure_assembler_, mesh.dimension());
	}

	int IncompressibleElasticVarForm::primary_ndof() const
	{
		return mesh_ ? space_.n_bases * mesh_->dimension() : 0;
	}

	int IncompressibleElasticVarForm::stacked_ndof() const
	{
		return primary_ndof() + pressure_space_.n_bases;
	}

	void IncompressibleElasticVarForm::build_rhs_assembler()
	{
		json rhs_solver_params = args["solver"]["linear"];
		if (!rhs_solver_params.contains("Pardiso"))
			rhs_solver_params["Pardiso"] = {};
		rhs_solver_params["Pardiso"]["mtype"] = -2;

		rhs_assembler_ = std::make_shared<assembler::RhsAssembler>(
			*primary_assembler_, *mesh_, nullptr,
			boundary_.dirichlet_nodes, boundary_.neumann_nodes,
			boundary_.dirichlet_nodes_position, boundary_.neumann_nodes_position,
			space_.n_bases, mesh_->dimension(), space_.basis_list(), space_.geometry_basis_list(), mass_ass_vals_cache_, *problem,
			args["space"]["advanced"]["bc_method"],
			rhs_solver_params,
			displacement_space_id_);
	}

	void IncompressibleElasticVarForm::build_basis(mesh::Mesh &mesh, const bool iso_parametric, const json &args)
	{
		build_elastic_basis(mesh, iso_parametric, args, displacement_space_id_);

		if (space_.disc_orders.maxCoeff() != space_.disc_orders.minCoeff())
			log_and_throw_error("p refinement not supported in mixed formulation!");
		if (!space_.poly_edge_to_data.empty())
			log_and_throw_error("Polygonal bases are not supported in mixed formulations!");

		const auto &all_boundary = boundary_.total_local_boundary;
		const int prev_bases = space_.n_bases;
		const bool use_corner_quadrature = args["space"]["advanced"]["use_corner_quadrature"];
		const int quadrature_order = args["space"]["advanced"]["quadrature_order"].get<int>();
		const int mass_quadrature_order = args["space"]["advanced"]["mass_quadrature_order"].get<int>();
		Eigen::VectorXi pressure_disc_orders;
		assign_discr_orders(args["space"]["discr_order"], pressure_space_id_, mesh, pressure_disc_orders);
		// to avoid serendipity
		const std::string pressure_basis_type = args["space"]["basis_type"].get<std::string>() == "Bernstein" ? "Bernstein" : "Lagrange";
		build_fe_space(
			mesh,
			/*iso_parametric=*/true,
			pressure_disc_orders,
			pressure_basis_type,
			args["space"]["poly_basis_type"],
			*primary_assembler_,
			/*value_dim=*/1,
			quadrature_order,
			mass_quadrature_order,
			use_corner_quadrature,
			args["space"]["advanced"]["n_harmonic_samples"],
			args["space"]["advanced"]["integral_constraints"],
			pressure_space_,
			pressure_boundary_,
			space_.geometry);

		assert(space_.basis_list().size() == pressure_space_.basis_list().size());
		for (int i = 0; i < int(pressure_space_.basis_list().size()); ++i)
		{
			quadrature::Quadrature b_quad;
			space_.basis_list()[i].compute_quadrature(b_quad);
			(*pressure_space_.bases)[i].set_quadrature([b_quad](quadrature::Quadrature &quad) { quad = b_quad; });
		}

		boundary_.clear_boundary_conditions();
		for (const auto &lb : all_boundary)
			boundary_.local_boundary.emplace_back(lb);
		pressure_boundary_.clear_boundary_conditions();

		problem->setup_bc(
			mesh, space_.n_bases,
			space_.basis_list(), space_.geometry_basis_list(), pressure_space_.basis_list(),
			boundary_.local_boundary,
			boundary_.boundary_nodes,
			boundary_.local_neumann_boundary,
			boundary_.local_pressure_boundary,
			boundary_.local_pressure_cavity,
			pressure_boundary_.boundary_nodes,
			boundary_.dirichlet_nodes, boundary_.neumann_nodes);
		pressure_boundary_.normalize_boundary_nodes();

		rebuild_node_positions(space_.basis_list(), boundary_.dirichlet_nodes, boundary_.dirichlet_nodes_position);
		rebuild_node_positions(space_.basis_list(), boundary_.neumann_nodes, boundary_.neumann_nodes_position);

		for (int i = prev_bases; i < space_.n_bases; ++i)
			for (int d = 0; d < mesh.dimension(); ++d)
				boundary_.boundary_nodes.push_back(i * mesh.dimension() + d);

		boundary_.normalize_boundary_nodes();

		if (space_.n_bases <= args["solver"]["advanced"]["cache_size"])
			pressure_ass_vals_cache_.init(mesh.is_volume(), pressure_space_.basis_list(), space_.geometry_basis_list());
		else
			pressure_ass_vals_cache_.init_empty();

		build_rhs_assembler();

		logger().info("n pressure bases: {}", pressure_space_.n_bases);
	}

	void IncompressibleElasticVarForm::assemble_rhs(const mesh::Mesh &mesh)
	{
		ElasticVarForm::assemble_rhs(mesh);
		const int prev_size = rhs_.rows();
		rhs_.conservativeResize(prev_size + pressure_space_.n_bases, rhs_.cols());
		rhs_.bottomRows(pressure_space_.n_bases).setZero();
	}

	void IncompressibleElasticVarForm::assemble_mass_mat(const mesh::Mesh &mesh, const json &args)
	{
		if (!problem->is_time_dependent())
		{
			avg_mass_ = 1;
			timings.assembling_mass_mat_time = 0;
			return;
		}

		mass_.resize(0, 0);
		igl::Timer timer;
		timer.start();
		logger().info("Assembling mass mat...");
		mass_assembler_->assemble(mesh.is_volume(), space_.n_bases, space_.basis_list(), space_.geometry_basis_list(), mass_ass_vals_cache_, 0, mass_, true);
		avg_mass_ = 0;
		for (int k = 0; k < mass_.outerSize(); ++k)
			for (StiffnessMatrix::InnerIterator it(mass_, k); it; ++it)
			{
				assert(it.col() == k);
				avg_mass_ += it.value();
			}
		avg_mass_ /= std::max(1, int(mass_.rows()));
		if (args["solver"]["advanced"]["lump_mass_matrix"])
			mass_ = utils::lump_matrix(mass_);
		timer.stop();
		timings.assembling_mass_mat_time = timer.getElapsedTime();
		logger().info(" took {}s", timings.assembling_mass_mat_time);
		stats.nn_zero = mass_.nonZeros();
		stats.num_dofs = mass_.rows();
		stats.mat_size = (long long)mass_.rows() * (long long)mass_.cols();
	}

	void IncompressibleElasticVarForm::prepare_initial_solution(Eigen::MatrixXd &sol) const
	{
		if (sol.size() <= 0)
			initial_elastic_solution(sol);
		if (sol.cols() > 1)
			sol.conservativeResize(Eigen::NoChange, 1);
		sol.conservativeResize(stacked_ndof(), sol.cols());
		sol.bottomRows(pressure_space_.n_bases).setZero();
	}

	void IncompressibleElasticVarForm::split_solution(const Eigen::MatrixXd &stacked, Eigen::MatrixXd &primary, Eigen::MatrixXd &pressure) const
	{
		const int cols = std::max(1, int(stacked.cols()));
		primary.setZero(primary_ndof(), cols);
		pressure.setZero(pressure_space_.n_bases, cols);
		const int primary_rows = std::min(primary_ndof(), int(stacked.rows()));
		if (primary_rows > 0)
			primary.topRows(primary_rows) = stacked.topRows(primary_rows);
		if (stacked.rows() > primary_ndof())
		{
			const int pressure_rows = std::min(pressure_space_.n_bases, int(stacked.rows()) - primary_ndof());
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
		primary_assembler_->assemble(mesh_->is_volume(), space_.n_bases, space_.basis_list(), space_.geometry_basis_list(), ass_vals_cache_, 0, elastic_stiffness);
		mixed_assembler_->assemble(mesh_->is_volume(), pressure_space_.n_bases, space_.n_bases, pressure_space_.basis_list(), space_.basis_list(), space_.geometry_basis_list(), pressure_ass_vals_cache_, ass_vals_cache_, 0, mixed_stiffness);
		pressure_assembler_->assemble(mesh_->is_volume(), pressure_space_.n_bases, pressure_space_.basis_list(), space_.geometry_basis_list(), pressure_ass_vals_cache_, 0, pressure_stiffness);

		assembler::AssemblerUtils::merge_mixed_matrices(
			space_.n_bases, pressure_space_.n_bases, mesh_->dimension(), /*add_average=*/false,
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
			boundary_.boundary_nodes,
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
		rhs_assembler_->set_bc(boundary_.local_boundary, boundary_.boundary_nodes, elastic_boundary_samples(), boundary_.local_neumann_boundary, rhs_);
		StiffnessMatrix A;
		build_stiffness_mat(A);
		Eigen::VectorXd b = rhs_;
		solve_linear_system(solver, A, b, args["output"]["advanced"]["spectrum"], sol);
	}

	void IncompressibleElasticVarForm::solve_transient_linear(Eigen::MatrixXd &sol)
	{
		auto solver = polysolve::linear::Solver::create(args["solver"]["linear"], logger());
		logger().info("{}...", solver->name());

		Eigen::MatrixXd displacement, pressure;
		split_solution(sol, displacement, pressure);
		auto bdf = time_integrator::ImplicitTimeIntegrator::construct_bdf_integrator(
			args["time"]["integrator"]);
		bdf->init(
			displacement,
			Eigen::MatrixXd::Zero(displacement.rows(), displacement.cols()),
			Eigen::MatrixXd::Zero(displacement.rows(), displacement.cols()),
			dt);
		time_integrator = bdf;

		save_timestep(t0, 0, t0, dt, sol);

		Eigen::MatrixXd current_rhs = rhs_;
		StiffnessMatrix stiffness, expanded_mass;
		build_stiffness_mat(stiffness);
		expand_primary_matrix(stacked_ndof(), mass_, expanded_mass);

		for (int t = 1; t <= time_steps; ++t)
		{
			const double time = t0 + t * dt;
			rhs_assembler_->compute_energy_grad(
				boundary_.local_boundary, boundary_.boundary_nodes, mass_assembler_->density(), elastic_boundary_samples(), boundary_.local_neumann_boundary, rhs_, time,
				current_rhs);
			rhs_assembler_->set_bc(
				boundary_.local_boundary, boundary_.boundary_nodes, elastic_boundary_samples(), boundary_.local_neumann_boundary, current_rhs, displacement, time);

			if (current_rhs.rows() != stacked_ndof())
			{
				const int old_rows = current_rhs.rows();
				current_rhs.conservativeResize(stacked_ndof(), current_rhs.cols());
				if (stacked_ndof() > old_rows)
					current_rhs.bottomRows(stacked_ndof() - old_rows).setZero();
			}
			current_rhs.bottomRows(pressure_space_.n_bases).setZero();

			StiffnessMatrix A = expanded_mass / bdf->beta_dt() + stiffness;
			Eigen::VectorXd b = Eigen::VectorXd::Zero(stacked_ndof());
			b.head(primary_ndof()) = (mass_ * bdf->weighted_sum_x_prevs()) / bdf->beta_dt();
			for (int i : boundary_.boundary_nodes)
				b[i] = 0;
			b += current_rhs;

			solve_linear_system(solver, A, b, args["output"]["advanced"]["spectrum"].get<bool>() && t == time_steps, sol);
			split_solution(sol, displacement, pressure);
			bdf->update_quantities(displacement.col(0));

			save_timestep(time, t, t0, dt, sol);
			save_elastic_step_state(t0, dt, t, time_integrator.get());
			logger().info("{}/{}  t={}", t, time_steps, time);
			notify_time_step(t, time_steps, t0, dt);
		}
	}

	void IncompressibleElasticVarForm::solve_problem(Eigen::MatrixXd &sol)
	{
		stats.spectrum.setZero();
		igl::Timer timer;
		timer.start();
		logger().info("Solving {}", primary_assembler_->name());
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
					*mesh_, pressure_space_.basis_list(), space_.geometry_basis_list(), sample, pressure, values,
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
