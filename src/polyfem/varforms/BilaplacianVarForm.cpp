#include "BilaplacianVarForm.hpp"

#include <polyfem/assembler/AssemblerUtils.hpp>
#include <polyfem/assembler/GenericProblem.hpp>
#include <polyfem/basis/LagrangeBasis2d.hpp>
#include <polyfem/basis/LagrangeBasis3d.hpp>
#include <polyfem/mesh/mesh2D/Mesh2D.hpp>
#include <polyfem/mesh/mesh3D/Mesh3D.hpp>
#include <polyfem/problem/ProblemFactory.hpp>
#include <polyfem/time_integrator/BDF.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/MatrixUtils.hpp>
#include <polyfem/utils/Timer.hpp>
#include <polyfem/varforms/VarFormUtils.hpp>

#include <polysolve/linear/FEMSolver.hpp>

namespace polyfem::varform
{
	using namespace varform::internal;

	void BilaplacianVarForm::reset()
	{
		ScalarVarForm::reset();
		pressure_bases.clear();
		n_pressure_bases = 0;
		pressure_mesh_nodes = nullptr;
		pressure_ass_vals_cache.init_empty();
		pressure_boundary_nodes.clear();
		mixed_assembler = nullptr;
		pressure_assembler = nullptr;
		use_avg_pressure = true;
		time_integrator = nullptr;
	}

	void BilaplacianVarForm::init(const std::string &formulation, const Units &units, const json &args, const std::string &out_path)
	{
		VarForm::init(formulation, units, args, out_path);
		const bool is_time_dependent = args.contains("time") && !args["time"].is_null();

		assembler = assembler::AssemblerUtils::make_assembler(formulation);
		mass_matrix_assembler = std::make_shared<assembler::Mass>();
		pure_mass_matrix_assembler = std::make_shared<assembler::HRZMass>();
		mixed_assembler = assembler::AssemblerUtils::make_mixed_assembler(formulation);
		pressure_assembler = assembler::AssemblerUtils::make_assembler(assembler::AssemblerUtils::other_assembler_name(formulation));

		if (!args.contains("preset_problem"))
		{
			problem = std::make_shared<assembler::GenericScalarProblem>("GenericScalar");
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
			problem = problem::ProblemFactory::factory().get_problem(args["preset_problem"]["type"]);
			problem->clear();
			problem->set_parameters(args["preset_problem"], root_path);
		}

		problem->set_units(*assembler, units);
		t0 = is_time_dependent ? args["time"]["t0"].get<double>() : 0.0;
		time_steps = is_time_dependent ? args["time"]["time_steps"].get<int>() : 0;
		dt = is_time_dependent ? args["time"]["dt"].get<double>() : 0.0;
	}

	void BilaplacianVarForm::save_json(const Eigen::MatrixXd &solution, std::ostream &out) const
	{
		save_json_stats(solution, n_pressure_bases, out);
	}

	void BilaplacianVarForm::load_mesh(const mesh::Mesh &mesh, const json &args)
	{
		VarForm::load_mesh(mesh, args);
		if (mixed_assembler)
			mixed_assembler->set_size(1);
		if (pressure_assembler)
			set_materials(*pressure_assembler);
	}

	std::shared_ptr<assembler::RhsAssembler> BilaplacianVarForm::build_rhs_assembler(
		const int n_bases,
		const std::vector<basis::ElementBases> &bases,
		const assembler::AssemblyValsCache &ass_vals_cache)
	{
		json rhs_solver_params = args["solver"]["linear"];
		if (!rhs_solver_params.contains("Pardiso"))
			rhs_solver_params["Pardiso"] = {};
		rhs_solver_params["Pardiso"]["mtype"] = -2;

		return std::make_shared<assembler::RhsAssembler>(
			*assembler, *mesh_, nullptr,
			dirichlet_nodes, neumann_nodes,
			dirichlet_nodes_position, neumann_nodes_position,
			n_bases, 1, bases, geom_bases(), ass_vals_cache, *problem,
			args["space"]["advanced"]["bc_method"],
			rhs_solver_params);
	}

	void BilaplacianVarForm::build_basis(mesh::Mesh &mesh, const bool iso_parametric, const json &args)
	{
		VarForm::build_basis(mesh, iso_parametric, args);

		if (disc_orders.maxCoeff() != disc_orders.minCoeff())
			log_and_throw_error("p refinement not supported in mixed formulation!");

		igl::Timer timer;
		timer.start();

		const int prev_bases = n_bases;
		const auto &all_boundary = total_local_boundary;
		const int prev_b_size = int(all_boundary.size());
		const bool has_polys = mesh.has_poly();
		const bool use_corner_quadrature = args["space"]["advanced"]["use_corner_quadrature"];
		const int quadrature_order = args["space"]["advanced"]["quadrature_order"].get<int>();
		const int mass_quadrature_order = args["space"]["advanced"]["mass_quadrature_order"].get<int>();
		const int order = args["space"]["pressure_discr_order"];
		std::vector<mesh::LocalBoundary> pressure_local_boundary;
		std::map<int, basis::InterfaceData> pressure_poly_edge_to_data;

		pressure_bases.clear();
		n_pressure_bases = 0;
		if (mesh.is_volume())
		{
			const mesh::Mesh3D &tmp_mesh = dynamic_cast<const mesh::Mesh3D &>(mesh);
			n_pressure_bases = basis::LagrangeBasis3d::build_bases(
				tmp_mesh, assembler->name(), quadrature_order, mass_quadrature_order,
				order, order,
				args["space"]["basis_type"] == "Bernstein", false,
				has_polys, false, use_corner_quadrature,
				pressure_bases, pressure_local_boundary, pressure_poly_edge_to_data, pressure_mesh_nodes);
		}
		else
		{
			const mesh::Mesh2D &tmp_mesh = dynamic_cast<const mesh::Mesh2D &>(mesh);
			n_pressure_bases = basis::LagrangeBasis2d::build_bases(
				tmp_mesh, assembler->name(), quadrature_order, mass_quadrature_order,
				order,
				args["space"]["basis_type"] == "Bernstein", false,
				has_polys, false, use_corner_quadrature,
				pressure_bases, pressure_local_boundary, pressure_poly_edge_to_data, pressure_mesh_nodes);
		}

		assert(bases.size() == pressure_bases.size());
		for (int i = 0; i < int(pressure_bases.size()); ++i)
		{
			quadrature::Quadrature b_quad;
			bases[i].compute_quadrature(b_quad);
			pressure_bases[i].set_quadrature([b_quad](quadrature::Quadrature &quad) { quad = b_quad; });
		}

		local_boundary.clear();
		for (const auto &lb : all_boundary)
			local_boundary.emplace_back(lb);

		local_neumann_boundary.clear();
		local_pressure_boundary.clear();
		local_pressure_cavity.clear();
		boundary_nodes.clear();
		pressure_boundary_nodes.clear();
		dirichlet_nodes.clear();
		neumann_nodes.clear();

		problem->setup_bc(
			mesh, n_bases,
			bases, geom_bases(), pressure_bases,
			local_boundary,
			boundary_nodes,
			local_neumann_boundary,
			local_pressure_boundary,
			local_pressure_cavity,
			pressure_boundary_nodes,
			dirichlet_nodes, neumann_nodes);

		rebuild_node_positions(bases, dirichlet_nodes, dirichlet_nodes_position);
		rebuild_node_positions(bases, neumann_nodes, neumann_nodes_position);

		const bool has_neumann = !local_neumann_boundary.empty() || int(local_boundary.size()) < prev_b_size;
		use_avg_pressure = !has_neumann;

		for (int i = prev_bases; i < n_bases; ++i)
			boundary_nodes.push_back(i);

		std::sort(boundary_nodes.begin(), boundary_nodes.end());
		boundary_nodes.erase(std::unique(boundary_nodes.begin(), boundary_nodes.end()), boundary_nodes.end());

		if (n_bases <= args["solver"]["advanced"]["cache_size"])
			pressure_ass_vals_cache.init(mesh.is_volume(), pressure_bases, geom_bases());
		else
			pressure_ass_vals_cache.init_empty();

		build_rhs_assembler();

		timer.stop();
		timings.building_basis_time += timer.getElapsedTime();
		logger().info("n pressure bases: {}", n_pressure_bases);
	}

	void BilaplacianVarForm::assemble_rhs(const mesh::Mesh &mesh, const json &args)
	{
		build_rhs_assembler();
		VarForm::assemble_rhs(mesh, args);

		const int prev_size = rhs.rows();
		rhs.conservativeResize(prev_size + n_pressure_bases, rhs.cols());
		if (local_neumann_boundary.empty())
		{
			rhs.bottomRows(n_pressure_bases).setZero();
		}
		else
		{
			Eigen::MatrixXd tmp = Eigen::MatrixXd::Zero(n_pressure_bases, 1);
			auto tmp_rhs_assembler = build_rhs_assembler(n_pressure_bases, pressure_bases, pressure_ass_vals_cache);
			tmp_rhs_assembler->set_bc(
				std::vector<mesh::LocalBoundary>(), std::vector<int>(), n_boundary_samples(), local_neumann_boundary, tmp);
			rhs.bottomRows(n_pressure_bases) = tmp;
		}
	}

	void BilaplacianVarForm::assemble_mass_mat(const mesh::Mesh &mesh, const json &args)
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
		mass_matrix_assembler->assemble(mesh.is_volume(), n_bases, bases, geom_bases(), mass_ass_vals_cache, 0, mass, true);
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

	int BilaplacianVarForm::stacked_ndof() const
	{
		return n_bases + n_pressure_bases;
	}

	void BilaplacianVarForm::prepare_initial_solution(Eigen::MatrixXd &sol) const
	{
		if (sol.size() <= 0)
			initial_solution(sol);
		if (sol.cols() > 1)
			sol.conservativeResize(Eigen::NoChange, 1);
		sol.conservativeResize(stacked_ndof(), sol.cols());
		sol.bottomRows(n_pressure_bases).setZero();
	}

	void BilaplacianVarForm::split_solution(const Eigen::MatrixXd &stacked, Eigen::MatrixXd &primary, Eigen::MatrixXd &pressure) const
	{
		const int cols = std::max(1, int(stacked.cols()));
		primary.setZero(n_bases, cols);
		pressure.setZero(n_pressure_bases, cols);
		const int primary_rows = std::min(n_bases, int(stacked.rows()));
		if (primary_rows > 0)
			primary.topRows(primary_rows) = stacked.topRows(primary_rows);
		if (stacked.rows() > n_bases)
		{
			const int pressure_rows = std::min(n_pressure_bases, int(stacked.rows()) - n_bases);
			if (pressure_rows > 0)
				pressure.topRows(pressure_rows) = stacked.middleRows(n_bases, pressure_rows);
		}
	}

	void BilaplacianVarForm::build_stiffness_mat(StiffnessMatrix &stiffness)
	{
		igl::Timer timer;
		timer.start();
		logger().info("Assembling stiffness mat...");

		StiffnessMatrix main_stiffness, mixed_stiffness, aux_stiffness;
		assembler->assemble(mesh_->is_volume(), n_bases, bases, geom_bases(), ass_vals_cache, 0, main_stiffness);
		mixed_assembler->assemble(mesh_->is_volume(), n_pressure_bases, n_bases, pressure_bases, bases, geom_bases(), pressure_ass_vals_cache, ass_vals_cache, 0, mixed_stiffness);
		pressure_assembler->assemble(mesh_->is_volume(), n_pressure_bases, pressure_bases, geom_bases(), pressure_ass_vals_cache, 0, aux_stiffness);

		assembler::AssemblerUtils::merge_mixed_matrices(
			n_bases, n_pressure_bases, 1, /*add_average=*/false,
			main_stiffness, mixed_stiffness, aux_stiffness, stiffness);

		timer.stop();
		timings.assembling_stiffness_mat_time = timer.getElapsedTime();
		logger().info(" took {}s", timings.assembling_stiffness_mat_time);
		stats.nn_zero = stiffness.nonZeros();
		stats.num_dofs = stiffness.rows();
		stats.mat_size = (long long)stiffness.rows() * (long long)stiffness.cols();
		write_matrix_market(args, stiffness);
	}

	void BilaplacianVarForm::solve_linear_system(
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
			boundary_nodes,
			x,
			n_bases,
			args["output"]["data"]["stiffness_mat"],
			compute_spectrum,
			/*is_fluid=*/false,
			/*use_avg_pressure=*/false);
		sol = x;
		solver->get_info(stats.solver_info);
	}

	void BilaplacianVarForm::solve_static_linear(Eigen::MatrixXd &sol)
	{
		auto solver = polysolve::linear::Solver::create(args["solver"]["linear"], logger());
		logger().info("{}...", solver->name());
		rhs_assembler->set_bc(
			local_boundary, boundary_nodes, n_boundary_samples(),
			(assembler->name() != "Bilaplacian") ? local_neumann_boundary : std::vector<mesh::LocalBoundary>(), rhs);
		StiffnessMatrix A;
		build_stiffness_mat(A);
		Eigen::VectorXd b = rhs;
		solve_linear_system(solver, A, b, args["output"]["advanced"]["spectrum"], sol);
	}

	void BilaplacianVarForm::solve_transient_linear(Eigen::MatrixXd &sol)
	{
		auto solver = polysolve::linear::Solver::create(args["solver"]["linear"], logger());
		logger().info("{}...", solver->name());

		Eigen::MatrixXd value, pressure;
		split_solution(sol, value, pressure);
		auto bdf = make_bdf_time_integrator();
		bdf->init(
			value,
			Eigen::MatrixXd::Zero(value.rows(), value.cols()),
			Eigen::MatrixXd::Zero(value.rows(), value.cols()),
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
				local_boundary, boundary_nodes, mass_matrix_assembler->density(), n_boundary_samples(), local_neumann_boundary, rhs, time,
				current_rhs);
			rhs_assembler->set_bc(
				local_boundary, boundary_nodes, n_boundary_samples(), local_neumann_boundary, current_rhs, value, time);

			if (current_rhs.rows() != stacked_ndof())
			{
				const int old_rows = current_rhs.rows();
				current_rhs.conservativeResize(stacked_ndof(), current_rhs.cols());
				if (stacked_ndof() > old_rows)
					current_rhs.bottomRows(stacked_ndof() - old_rows).setZero();
			}
			current_rhs.bottomRows(n_pressure_bases).setZero();

			StiffnessMatrix A = expanded_mass / bdf->beta_dt() + stiffness;
			Eigen::VectorXd b = Eigen::VectorXd::Zero(stacked_ndof());
			b.head(n_bases) = (mass * bdf->weighted_sum_x_prevs()) / bdf->beta_dt();
			for (int i : boundary_nodes)
				b[i] = 0;
			b += current_rhs;

			solve_linear_system(solver, A, b, args["output"]["advanced"]["spectrum"].get<bool>() && t == time_steps, sol);
			split_solution(sol, value, pressure);
			bdf->update_quantities(value.col(0));

			save_timestep(time, t, t0, dt, sol);
			save_step_state(t0, dt, t, time_integrator.get());
			logger().info("{}/{}  t={}", t, time_steps, time);
			notify_time_step(t);
		}
	}

	void BilaplacianVarForm::solve_problem(Eigen::MatrixXd &sol)
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

	std::vector<io::OutputField> BilaplacianVarForm::output_fields(
		const io::OutputSample &sample,
		const Eigen::MatrixXd &solution,
		const io::OutputFieldOptions &options) const
	{
		Eigen::MatrixXd value, pressure;
		split_solution(solution, value, pressure);
		auto fields = ScalarVarForm::output_fields(sample, value, options);
		const bool export_pressure_gradient =
			!options.fields.empty() && options.export_field("pressure_gradient");
		if (mesh_ && (options.export_field("pressure") || export_pressure_gradient || (!options.fields.empty() && options.export_field("auxiliary"))))
		{
			Eigen::MatrixXd values, gradients;
			if (sample_scalar_field(
					*mesh_, pressure_bases, geom_bases(), sample, pressure, values,
					export_pressure_gradient ? &gradients : nullptr))
			{
				if (options.export_field("pressure"))
					fields.push_back({"pressure", values, io::OutputField::Association::Point});
				if (export_pressure_gradient)
					fields.push_back({"pressure_gradient", gradients, io::OutputField::Association::Point});
				if (!options.fields.empty() && options.export_field("auxiliary"))
					fields.push_back({"auxiliary", values, io::OutputField::Association::Point});
			}
		}
		return fields;
	}
} // namespace polyfem::varform
