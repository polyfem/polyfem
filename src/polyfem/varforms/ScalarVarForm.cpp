#include "ScalarVarForm.hpp"

#include <polyfem/assembler/AssemblerUtils.hpp>
#include <polyfem/assembler/GenericProblem.hpp>

#include <polyfem/io/Evaluator.hpp>

#include <polyfem/problem/ProblemFactory.hpp>

#include <polyfem/time_integrator/BDF.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/Timer.hpp>

#include <unsupported/Eigen/SparseExtra>

#include <polysolve/linear/FEMSolver.hpp>

namespace polyfem::varform
{
	namespace
	{
		void write_matrix_market(const json &args, const StiffnessMatrix &stiffness)
		{
			const std::string full_mat_path = args["output"]["data"]["full_mat"];
			if (!full_mat_path.empty())
				Eigen::saveMarket(stiffness, full_mat_path);
		}
	} // namespace

	void ScalarVarForm::reset()
	{
		VarForm::reset();
		time_integrator = nullptr;
	}

	void ScalarVarForm::init(const std::string &formulation, const Units &units, const json &args, const std::string &out_path)
	{
		VarForm::init(formulation, units, args, out_path);
		const bool is_time_dependent = args.contains("time") && !args["time"].is_null();

		assembler = assembler::AssemblerUtils::make_assembler(formulation);
		assert(assembler->name() == formulation);
		assert(assembler->is_linear());
		assert(!assembler->is_tensor());
		mass_matrix_assembler = std::make_shared<assembler::Mass>();
		pure_mass_matrix_assembler = std::make_shared<assembler::HRZMass>();

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

	void ScalarVarForm::save_json(const Eigen::MatrixXd &solution, std::ostream &out) const
	{
		save_json_stats(solution, 0, out);
	}

	std::vector<io::OutputField> ScalarVarForm::output_fields(
		const io::OutputSample &sample,
		const Eigen::MatrixXd &solution,
		const io::OutputFieldOptions &options) const
	{
		std::vector<io::OutputField> fields = common_output_fields(sample, solution, options);
		if (!mesh_ || !problem || solution.size() <= 0)
			return fields;

		assert(problem->is_scalar());
		const bool has_element_samples = sample.local_points.rows() > 0 && sample.local_points.rows() == sample.element_ids.size();
		const int output_rows = sample.points.rows() > 0 ? sample.points.rows() : std::max<int>(sample.local_points.rows(), sample.node_ids.size());

		const auto sample_dof_field = [&](const Eigen::MatrixXd &dof_values, Eigen::MatrixXd &values, Eigen::MatrixXd *gradients = nullptr) -> bool {
			if (dof_values.size() <= 0)
				return false;

			if (has_element_samples)
			{
				values.resize(sample.local_points.rows(), 1);
				if (gradients)
					gradients->resize(sample.local_points.rows(), mesh_->dimension());
				for (int i = 0; i < sample.local_points.rows(); ++i)
				{
					const int element_id = sample.element_ids(i);
					if (element_id < 0)
					{
						values(i) = 0;
						if (gradients)
							gradients->row(i).setZero();
						continue;
					}

					Eigen::MatrixXd local_sol, local_grad;
					io::Evaluator::interpolate_at_local_vals(
						*mesh_, 1, scalar_space.bases, geom_bases(),
						element_id, sample.local_points.row(i), dof_values, local_sol, local_grad);
					values(i) = local_sol(0);
					if (gradients)
						gradients->row(i) = local_grad;
				}

				if (output_rows > values.rows())
				{
					const int previous_rows = values.rows();
					values.conservativeResize(output_rows, Eigen::NoChange);
					values.bottomRows(output_rows - previous_rows).setZero();
					if (gradients)
					{
						gradients->conservativeResize(output_rows, Eigen::NoChange);
						gradients->bottomRows(output_rows - previous_rows).setZero();
					}
				}
				return true;
			}

			if (sample.node_ids.size() > 0)
			{
				values.resize(sample.node_ids.size(), 1);
				for (int i = 0; i < sample.node_ids.size(); ++i)
				{
					const int node_id = sample.node_ids(i);
					if (node_id < 0 || node_id >= dof_values.rows())
						return false;
					values(i) = dof_values(node_id);
				}
				return sample.points.rows() == 0 || sample.points.rows() == values.rows();
			}

			return false;
		};

		const bool export_solution_gradient =
			!options.fields.empty() && options.export_field("solution_gradient");
		if (options.export_field("solution") || export_solution_gradient)
		{
			Eigen::MatrixXd values, gradients;
			if (sample_dof_field(
					solution, values,
					export_solution_gradient ? &gradients : nullptr))
			{
				if (options.export_field("solution"))
					fields.push_back({"solution", values, io::OutputField::Association::Point});
				if (export_solution_gradient)
					fields.push_back({"solution_gradient", gradients, io::OutputField::Association::Point});
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

	void ScalarVarForm::build_stiffness_mat(StiffnessMatrix &stiffness)
	{
		igl::Timer timer;
		timer.start();
		logger().info("Assembling stiffness mat...");
		assert(assembler->is_linear());
		assert(problem->is_scalar());

		assembler->assemble(mesh_->is_volume(), scalar_space.n_bases, scalar_space.bases, geom_bases(), scalar_caches.values, 0, stiffness);

		timer.stop();
		timings.assembling_stiffness_mat_time = timer.getElapsedTime();
		logger().info(" took {}s", timings.assembling_stiffness_mat_time);

		stats.nn_zero = stiffness.nonZeros();
		stats.num_dofs = stiffness.rows();
		stats.mat_size = (long long)stiffness.rows() * (long long)stiffness.cols();
		logger().info("sparsity: {}/{}", stats.nn_zero, stats.mat_size);

		write_matrix_market(args, stiffness);
	}

	void ScalarVarForm::solve_linear_system(
		const std::unique_ptr<polysolve::linear::Solver> &solver,
		StiffnessMatrix &A,
		Eigen::VectorXd &b,
		const bool compute_spectrum,
		Eigen::MatrixXd &sol)
	{
		assert(assembler->is_linear());
		assert(problem->is_scalar());
		assert(rhs_assembler != nullptr);

		Eigen::VectorXd x;
		stats.spectrum = dirichlet_solve(
			*solver,
			A,
			b,
			boundary.boundary_nodes,
			x,
			scalar_space.n_bases,
			args["output"]["data"]["stiffness_mat"],
			compute_spectrum,
			/*is_problem_mixed=*/false,
			/*use_avg_pressure=*/false);

		sol = x;
		solver->get_info(stats.solver_info);

		const auto error = (A * x - b).norm();
		if (error > 1e-4)
			logger().error("Solver error: {}", error);
		else
			logger().debug("Solver error: {}", error);
	}

	void ScalarVarForm::solve_static(Eigen::MatrixXd &sol)
	{
		auto solver = polysolve::linear::Solver::create(args["solver"]["linear"], logger());
		logger().info("{}...", solver->name());

		rhs_assembler->set_bc(
			boundary.local_boundary, boundary.boundary_nodes, n_boundary_samples(),
			(assembler->name() != "Bilaplacian") ? boundary.local_neumann_boundary : std::vector<mesh::LocalBoundary>(), rhs);

		StiffnessMatrix A;
		build_stiffness_mat(A);

		Eigen::VectorXd b = rhs;
		solve_linear_system(solver, A, b, args["output"]["advanced"]["spectrum"], sol);
	}

	void ScalarVarForm::solve_transient(Eigen::MatrixXd &sol)
	{
		assert(problem->is_time_dependent());
		assert(rhs_assembler != nullptr);

		auto solver = polysolve::linear::Solver::create(args["solver"]["linear"], logger());
		logger().info("{}...", solver->name());

		auto bdf = std::make_shared<time_integrator::BDF>();
		bdf->set_parameters(args["time"]);
		bdf->init(sol, Eigen::VectorXd::Zero(sol.size()), Eigen::VectorXd::Zero(sol.size()), dt);
		time_integrator = bdf;

		save_timestep(t0, 0, t0, dt, sol);

		Eigen::MatrixXd current_rhs = rhs;

		StiffnessMatrix stiffness;
		build_stiffness_mat(stiffness);

		const QuadratureOrders n_b_samples = n_boundary_samples();
		for (int t = 1; t <= time_steps; ++t)
		{
			const double time = t0 + t * dt;

			rhs_assembler->compute_energy_grad(
				boundary.local_boundary, boundary.boundary_nodes, mass_matrix_assembler->density(), n_b_samples,
				boundary.local_neumann_boundary, rhs, time, current_rhs);

			rhs_assembler->set_bc(
				boundary.local_boundary, boundary.boundary_nodes, n_b_samples, boundary.local_neumann_boundary, current_rhs, sol, time);

			StiffnessMatrix A = mass / bdf->beta_dt() + stiffness;
			Eigen::VectorXd b = (mass * bdf->weighted_sum_x_prevs()) / bdf->beta_dt();
			for (int i : boundary.boundary_nodes)
				b[i] = 0;
			b += current_rhs;

			solve_linear_system(solver, A, b, args["output"]["advanced"]["spectrum"].get<bool>() && t == time_steps, sol);

			bdf->update_quantities(sol);
			save_timestep(time, t, t0, dt, sol);
			save_step_state(t0, dt, t, time_integrator.get());

			logger().info("{}/{}  t={}", t, time_steps, time);
			notify_time_step(t);
		}
	}

	void ScalarVarForm::solve_problem(Eigen::MatrixXd &sol)
	{
		stats.spectrum.setZero();

		igl::Timer timer;
		timer.start();
		logger().info("Solving {}", assembler->name());

		{
			POLYFEM_SCOPED_TIMER("Setup RHS");

			if (sol.size() <= 0)
				initial_solution(sol);

			if (sol.cols() > 1)
				sol.conservativeResize(Eigen::NoChange, 1);
		}

		time_integrator = nullptr;
		if (problem->is_time_dependent())
			solve_transient(sol);
		else
			solve_static(sol);

		timer.stop();
		timings.solving_time = timer.getElapsedTime();
		logger().info(" took {}s", timings.solving_time);
	}
} // namespace polyfem::varform
