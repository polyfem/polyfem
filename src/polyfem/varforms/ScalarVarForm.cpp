#include "ScalarVarForm.hpp"

#include <polyfem/assembler/AssemblerUtils.hpp>
#include <polyfem/assembler/GenericProblem.hpp>

#include <polyfem/io/Evaluator.hpp>
#include <polyfem/io/MatrixIO.hpp>

#include <polyfem/problem/ProblemFactory.hpp>
#include <polyfem/refinement/APriori.hpp>

#include <polyfem/time_integrator/BDF.hpp>
#include <polyfem/utils/Jacobian.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/MatrixUtils.hpp>
#include <polyfem/utils/Timer.hpp>

#include <unsupported/Eigen/SparseExtra>

#include <polysolve/linear/FEMSolver.hpp>

#include <algorithm>

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
		space_.reset();
		boundary_.reset();
		ass_vals_cache_.init_empty();
		mass_ass_vals_cache_.init_empty(true);
		pure_mass_ass_vals_cache_.init_empty(true);
		rhs_assembler_ = nullptr;
		mass_.resize(0, 0);
		pure_mass_.resize(0, 0);
		avg_mass_ = 0;
		rhs_.resize(0, 0);
		primary_assembler_ = nullptr;
		mass_assembler_ = nullptr;
		pure_mass_assembler_ = nullptr;
		t0 = 0;
		time_steps = 0;
		dt = 0;
		time_integrator = nullptr;
	}

	void ScalarVarForm::init(const std::string &formulation, const Units &units, const json &args, const std::string &out_path)
	{
		VarForm::init(formulation, units, args, out_path);
		const bool is_time_dependent = args.contains("time") && !args["time"].is_null();

		primary_assembler_ = assembler::AssemblerUtils::make_assembler(formulation);
		assert(primary_assembler_->name() == formulation);
		assert(primary_assembler_->is_linear());
		assert(!primary_assembler_->is_tensor());
		mass_assembler_ = std::make_shared<assembler::Mass>();
		pure_mass_assembler_ = std::make_shared<assembler::HRZMass>();

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

		problem->set_units(*primary_assembler_, units);

		t0 = is_time_dependent ? args["time"]["t0"].get<double>() : 0.0;
		time_steps = is_time_dependent ? args["time"]["time_steps"].get<int>() : 0;
		dt = is_time_dependent ? args["time"]["dt"].get<double>() : 0.0;
	}

	void ScalarVarForm::load_mesh(const mesh::Mesh &mesh, const json &args)
	{
		set_materials(*primary_assembler_, 1);
		set_materials(*mass_assembler_, 1);
		pure_mass_assembler_->set_size(mass_assembler_->size());

		problem->init(mesh);
	}

	void ScalarVarForm::build_basis(mesh::Mesh &mesh, const bool iso_parametric, const json &args)
	{
		assert(problem);
		assert(primary_assembler_);
		assert(mass_assembler_);
		assert(pure_mass_assembler_);

		Eigen::VectorXi space_disc_orders;
		assign_discr_orders(args["space"]["discr_order"], mesh, space_disc_orders);

		if (args["space"]["use_p_ref"])
		{
			refinement::APriori::p_refine(
				mesh,
				args["space"]["advanced"]["B"],
				args["space"]["advanced"]["h1_formula"],
				args["space"]["discr_order"],
				args["space"]["advanced"]["discr_order_max"],
				stats,
				space_disc_orders);

			logger().info("min p: {} max p: {}", space_disc_orders.minCoeff(), space_disc_orders.maxCoeff());
		}

		build_fe_space(
			mesh,
			iso_parametric,
			space_disc_orders,
			args["space"]["basis_type"],
			args["space"]["poly_basis_type"],
			*primary_assembler_,
			/*value_dim=*/1,
			args["space"]["advanced"]["quadrature_order"],
			args["space"]["advanced"]["mass_quadrature_order"],
			args["space"]["advanced"]["use_corner_quadrature"],
			args["space"]["advanced"]["n_harmonic_samples"],
			args["space"]["advanced"]["integral_constraints"],
			space_,
			boundary_);

		boundary_.clear_boundary_conditions();

		problem->update_nodes(space_.space_in_node_to_node);
		mesh.update_nodes(space_.space_in_node_to_node);

		problem->setup_bc(
			mesh,
			assembler::BoundaryKind::Dirichlet,
			/*fe_space_id=*/-1,
			space_.basis_list(),
			boundary_.total_local_boundary,
			boundary_.local_boundary,
			boundary_.boundary_nodes);
		std::vector<int> unused_neumann_boundary_nodes;
		problem->setup_bc(
			mesh,
			assembler::BoundaryKind::Neumann,
			/*fe_space_id=*/-1,
			space_.basis_list(),
			boundary_.total_local_boundary,
			boundary_.local_neumann_boundary,
			unused_neumann_boundary_nodes);

		problem->setup_nodal_bc(
			mesh,
			assembler::BoundaryKind::Dirichlet,
			/*fe_space_id=*/-1,
			space_.n_bases,
			boundary_.dirichlet_nodes);
		problem->setup_nodal_bc(
			mesh,
			assembler::BoundaryKind::Neumann,
			/*fe_space_id=*/-1,
			space_.n_bases,
			boundary_.neumann_nodes);

		for (const int n_id : boundary_.dirichlet_nodes)
		{
			const int tag = mesh.get_node_id(n_id);
			if (problem->is_nodal_dimension_dirichlet(n_id, tag, 0))
				boundary_.boundary_nodes.push_back(n_id);
		}

		boundary_.normalize_boundary_nodes();

		rebuild_node_positions(space_.basis_list(), boundary_.dirichlet_nodes, boundary_.dirichlet_nodes_position);
		rebuild_node_positions(space_.basis_list(), boundary_.neumann_nodes, boundary_.neumann_nodes_position);

		const auto &current_bases = space_.geometry_basis_list();
		if (args["space"]["advanced"]["count_flipped_els"])
			stats.count_flipped_elements(mesh, current_bases);

		const int n_samples = 10;
		stats.compute_mesh_size(mesh, current_bases, n_samples, args["output"]["advanced"]["curved_mesh_size"]);

		logger().info("flipped elements {}", stats.n_flipped);
		logger().info("h: {}", stats.mesh_size);

		if (space_.n_bases <= args["solver"]["advanced"]["cache_size"])
		{
			igl::Timer timer;
			timer.start();
			logger().info("Building cache...");
			ass_vals_cache_.init(mesh.is_volume(), space_.basis_list(), current_bases);
			mass_ass_vals_cache_.init(mesh.is_volume(), space_.basis_list(), current_bases, true);
			pure_mass_ass_vals_cache_.init(mesh.is_volume(), space_.basis_list(), current_bases, true);
			logger().info(" took {}s", timer.getElapsedTime());
		}
		else
		{
			ass_vals_cache_.init_empty();
			mass_ass_vals_cache_.init_empty(true);
			pure_mass_ass_vals_cache_.init_empty(true);
		}
	}

	void ScalarVarForm::build_rhs_assembler()
	{
		json rhs_solver_params = args["solver"]["linear"];
		if (!rhs_solver_params.contains("Pardiso"))
			rhs_solver_params["Pardiso"] = {};
		rhs_solver_params["Pardiso"]["mtype"] = -2;

		rhs_assembler_ = std::make_shared<assembler::RhsAssembler>(
			*primary_assembler_, *mesh_, nullptr,
			boundary_.dirichlet_nodes, boundary_.neumann_nodes,
			boundary_.dirichlet_nodes_position, boundary_.neumann_nodes_position,
			space_.n_bases, /*size=*/1, space_.basis_list(), space_.geometry_basis_list(), mass_ass_vals_cache_, *problem,
			args["space"]["advanced"]["bc_method"],
			rhs_solver_params);
	}

	void ScalarVarForm::assemble_rhs(const mesh::Mesh &mesh)
	{
		igl::Timer timer;
		json p_params = {};
		p_params["formulation"] = primary_assembler_->name();
		p_params["root_path"] = root_path;
		{
			RowVectorNd min, max, delta;
			mesh.bounding_box(min, max);
			delta = (max - min) / 2. + min;
			if (mesh.is_volume())
				p_params["bbox_center"] = {delta(0), delta(1), delta(2)};
			else
				p_params["bbox_center"] = {delta(0), delta(1)};
		}
		problem->set_parameters(p_params, root_path);

		rhs_.resize(0, 0);

		timer.start();
		logger().info("Assigning rhs...");

		build_rhs_assembler();
		assert(rhs_assembler_ != nullptr);
		rhs_assembler_->assemble(mass_assembler_->density(), rhs_);
		rhs_ *= -1;

		timings.assigning_rhs_time = timer.getElapsedTime();
		logger().info(" took {}s", timings.assigning_rhs_time);
	}

	void ScalarVarForm::assemble_mass_mat(const mesh::Mesh &mesh, const json &args)
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

		assert(mass_.size() > 0);

		avg_mass_ = 0;
		for (int k = 0; k < mass_.outerSize(); ++k)
		{
			for (StiffnessMatrix::InnerIterator it(mass_, k); it; ++it)
			{
				assert(it.col() == k);
				avg_mass_ += it.value();
			}
		}

		avg_mass_ /= mass_.rows();
		logger().info("average mass {}", avg_mass_);

		if (args["solver"]["advanced"]["lump_mass_matrix"])
			mass_ = utils::lump_matrix(mass_);

		timer.stop();
		timings.assembling_mass_mat_time = timer.getElapsedTime();
		logger().info(" took {}s", timings.assembling_mass_mat_time);

		stats.nn_zero = mass_.nonZeros();
		stats.num_dofs = mass_.rows();
		stats.mat_size = (long long)mass_.rows() * (long long)mass_.cols();
		logger().info("sparsity: {}/{}", stats.nn_zero, stats.mat_size);
	}

	void ScalarVarForm::prepare_initial_solution(Eigen::MatrixXd &solution) const
	{
		assert(rhs_assembler_ != nullptr);

		const bool was_solution_loaded = read_initial_x_from_file(
			resolve_input_path(args["input"]["data"]["state"]), "u",
			args["input"]["data"]["reorder"], space_.space_in_node_to_node,
			/*dim=*/1, solution);

		if (!was_solution_loaded)
		{
			if (problem->is_time_dependent())
				rhs_assembler_->initial_solution(solution);
			else
			{
				solution.resize(rhs_.size(), 1);
				solution.setZero();
			}
		}
	}

	void ScalarVarForm::save_json(const Eigen::MatrixXd &solution, std::ostream &out) const
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
		const int primary_size = space_.n_bases;
		const Eigen::MatrixXd stats_solution =
			solution.rows() >= primary_size
				? solution.topRows(primary_size).eval()
				: solution;

		nlohmann::json j;
		stats.save_json(
			args, space_.n_bases, /*n_auxiliary_bases=*/0,
			stats_solution, *mesh_, space_.disc_orders, space_.disc_ordersq, *problem,
			timings, primary_assembler_ ? primary_assembler_->name() : name(), space_.is_iso_parametric(),
			args["output"]["advanced"]["sol_at_node"], j);
		out << j.dump(4) << std::endl;
	}

	io::OutputSpace ScalarVarForm::output_space() const
	{
		Eigen::VectorXi output_orders = space_.disc_orders;
		if (mesh_ && space_.disc_ordersq.size() == space_.disc_orders.size())
		{
			for (int e = 0; e < output_orders.size(); ++e)
			{
				if (mesh_->is_prism(e))
					output_orders(e) = std::max(space_.disc_orders(e), space_.disc_ordersq(e));
			}
		}

		return {
			mesh_.get(),
			&space_.geometry_basis_list(),
			output_orders,
			&space_.polys,
			&space_.polys_3d,
			&boundary_.total_local_boundary,
			nullptr,
			nullptr,
			&boundary_.dirichlet_nodes,
			&boundary_.dirichlet_nodes_position};
	}

	io::OutStatsData ScalarVarForm::compute_errors(const Eigen::MatrixXd &solution)
	{
		if (!args["output"]["advanced"]["compute_error"])
			return stats;

		double tend = 0;
		if (!args["time"].is_null())
			tend = args["time"]["tend"];

		stats.compute_errors(space_.n_bases, space_.basis_list(), space_.geometry_basis_list(), *mesh_, *problem, tend, solution);
		return stats;
	}

	void ScalarVarForm::export_data(const Eigen::MatrixXd &solution) const
	{
		const io::OutputSpace space = output_space();
		if (!space.mesh)
		{
			logger().error("Load the mesh first!");
			return;
		}
		if (solution.size() <= 0)
		{
			logger().error("Solve the problem first!");
			return;
		}

		ensure_output_sampler();

		const std::string vis_mesh_path = resolve_output_path(args["output"]["paraview"]["file_name"]);
		const bool has_time = args.contains("time") && !args["time"].is_null();
		double tend = has_time ? args["time"]["tend"].get<double>() : 1.0;
		double dt = 1;
		if (has_time)
			dt = args["time"]["dt"];

		const auto opts = export_options(space);
		output_geometry_.export_data(
			space,
			output_field_function(solution, opts),
			has_time,
			tend, dt,
			opts,
			vis_mesh_path);

		const std::string solution_path = resolve_output_path(args["output"]["data"]["solution"]);
		if (!solution_path.empty())
		{
			const int primary_ndof = std::min<int>(solution.rows(), space_.n_bases);
			const Eigen::MatrixXd primary_solution = solution.topRows(primary_ndof);
			if (opts.reorder_output && space_.space_in_node_to_node.size() > 0)
			{
				const Eigen::MatrixXd nodal_solution = utils::unflatten(primary_solution, 1);
				Eigen::MatrixXd reordered = Eigen::MatrixXd::Zero(nodal_solution.rows(), nodal_solution.cols());
				for (int input_node = 0; input_node < space_.space_in_node_to_node.size(); ++input_node)
				{
					const int node = space_.space_in_node_to_node(input_node);
					if (node >= 0 && node < nodal_solution.rows() && input_node < reordered.rows())
						reordered.row(input_node) = nodal_solution.row(node);
				}
				io::write_matrix(solution_path, reordered);
			}
			else
			{
				io::write_matrix(solution_path, primary_solution);
			}
		}

		const std::string nodes_path = resolve_output_path(args["output"]["data"]["nodes"]);
		if (!nodes_path.empty())
		{
			Eigen::MatrixXd nodes = Eigen::MatrixXd::Zero(space_.n_bases, mesh_->dimension());
			for (const basis::ElementBases &element_bases : space_.basis_list())
				for (const basis::Basis &basis : element_bases.bases)
					for (const auto &global : basis.global())
						nodes.row(global.index) = global.node;
			io::write_matrix(nodes_path, nodes);
		}

		const std::string stress_path = resolve_output_path(args["output"]["data"]["stress_mat"]);
		const std::string mises_path = resolve_output_path(args["output"]["data"]["mises"]);
		if ((!stress_path.empty() || !mises_path.empty()) && primary_assembler_)
		{
			Eigen::MatrixXd stress;
			Eigen::VectorXd mises;
			io::Evaluator::compute_stress_at_quadrature_points(
				*mesh_, problem->is_scalar(), space_.basis_list(), space_.geometry_basis_list(),
				space_.disc_orders, space_.disc_ordersq, *primary_assembler_, solution, tend,
				stress, mises);
			if (!stress_path.empty())
				io::write_matrix(stress_path, stress);
			if (!mises_path.empty())
				io::write_matrix(mises_path, mises);
		}
	}

	std::vector<io::OutputField> ScalarVarForm::output_fields(
		const io::OutputSample &sample,
		const Eigen::MatrixXd &solution,
		const io::OutputFieldOptions &options) const
	{
		std::vector<io::OutputField> fields;
		if (!mesh_ || !problem || solution.size() <= 0)
			return fields;

		assert(problem->is_scalar());
		const bool has_element_samples = sample.local_points.rows() > 0 && sample.local_points.rows() == sample.element_ids.size();
		const int output_rows = sample.points.rows() > 0 ? sample.points.rows() : std::max<int>(sample.local_points.rows(), sample.node_ids.size());
		const int primary_ndof = std::min<int>(solution.rows(), space_.n_bases);
		const Eigen::MatrixXd primary_solution = solution.topRows(primary_ndof);

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
						*mesh_, 1, space_.basis_list(), space_.geometry_basis_list(),
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

		const auto &paraview_options = args["output"]["paraview"]["options"];
		if (has_element_samples && problem->has_exact_sol() && sample.points.rows() == output_rows)
		{
			Eigen::MatrixXd exact;
			problem->exact(sample.points, sample.time, exact);
			if (exact.rows() == output_rows)
			{
				if (options.export_field("exact"))
					fields.push_back({"exact", exact, io::OutputField::Association::Point});
				if (options.export_field("error"))
				{
					Eigen::MatrixXd values;
					if (sample_dof_field(primary_solution, values))
						fields.push_back({"error", (values - exact).rowwise().norm(), io::OutputField::Association::Point});
				}
			}
		}

		if ((paraview_options["nodes"] || (!options.fields.empty() && options.export_field("nodes")))
			&& has_element_samples
			&& sample.primitive_ids.size() == 0)
		{
			Eigen::MatrixXd dof_ids(primary_ndof, 1);
			dof_ids.col(0).setLinSpaced(primary_ndof, 0, primary_ndof - 1);
			Eigen::MatrixXd values;
			if (sample_dof_field(dof_ids, values))
				fields.push_back({"nodes", values, io::OutputField::Association::Point});
		}

		if ((paraview_options["jacobian_validity"] || (!options.fields.empty() && options.export_field("validity")))
			&& has_element_samples
			&& mesh_->dimension() == 1
			&& sample.primitive_ids.size() == 0)
		{
			const auto invalid_elements = utils::count_invalid(mesh_->dimension(), space_.basis_list(), space_.geometry_basis_list(), primary_solution);
			Eigen::MatrixXd validity = Eigen::MatrixXd::Zero(output_rows, 1);
			for (int i = 0; i < sample.element_ids.size(); ++i)
				validity(i) = std::find(invalid_elements.begin(), invalid_elements.end(), sample.element_ids(i)) != invalid_elements.end();
			fields.push_back({"validity", validity, io::OutputField::Association::Point});
		}

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

		if (paraview_options["material"] && has_element_samples)
		{
			const auto &params = primary_assembler_->parameters();
			std::map<std::string, Eigen::MatrixXd> param_values;
			for (const auto &[p, _] : params)
				param_values[p].setZero(output_rows, 1);

			Eigen::MatrixXd rhos = Eigen::MatrixXd::Zero(output_rows, 1);
			const auto &density = mass_assembler_->density();
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
		assert(primary_assembler_->is_linear());
		assert(problem->is_scalar());

		primary_assembler_->assemble(mesh_->is_volume(), space_.n_bases, space_.basis_list(), space_.geometry_basis_list(), ass_vals_cache_, 0, stiffness);

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
		assert(primary_assembler_->is_linear());
		assert(problem->is_scalar());
		assert(rhs_assembler_ != nullptr);

		Eigen::VectorXd x;
		stats.spectrum = dirichlet_solve(
			*solver,
			A,
			b,
			boundary_.boundary_nodes,
			x,
			space_.n_bases,
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

		const int gdiscr_order = mesh_->orders().size() <= 0 ? 1 : mesh_->orders().maxCoeff();
		const QuadratureOrders boundary_samples = n_boundary_samples(space_.disc_orders.maxCoeff(), gdiscr_order);

		rhs_assembler_->set_bc(
			boundary_.local_boundary, boundary_.boundary_nodes, boundary_samples,
			(primary_assembler_->name() != "Bilaplacian") ? boundary_.local_neumann_boundary : std::vector<mesh::LocalBoundary>(), rhs_);

		StiffnessMatrix A;
		build_stiffness_mat(A);

		Eigen::VectorXd b = rhs_;
		solve_linear_system(solver, A, b, args["output"]["advanced"]["spectrum"], sol);
	}

	void ScalarVarForm::solve_transient(Eigen::MatrixXd &sol)
	{
		assert(problem->is_time_dependent());
		assert(rhs_assembler_ != nullptr);

		auto solver = polysolve::linear::Solver::create(args["solver"]["linear"], logger());
		logger().info("{}...", solver->name());

		auto bdf = make_bdf_time_integrator();
		bdf->init(sol, Eigen::VectorXd::Zero(sol.size()), Eigen::VectorXd::Zero(sol.size()), dt);
		time_integrator = bdf;

		save_timestep(t0, 0, t0, dt, sol);

		Eigen::MatrixXd current_rhs = rhs_;

		StiffnessMatrix stiffness;
		build_stiffness_mat(stiffness);

		const int gdiscr_order = mesh_->orders().size() <= 0 ? 1 : mesh_->orders().maxCoeff();
		const QuadratureOrders n_b_samples = n_boundary_samples(space_.disc_orders.maxCoeff(), gdiscr_order);
		for (int t = 1; t <= time_steps; ++t)
		{
			const double time = t0 + t * dt;

			rhs_assembler_->compute_energy_grad(
				boundary_.local_boundary, boundary_.boundary_nodes, mass_assembler_->density(), n_b_samples,
				boundary_.local_neumann_boundary, rhs_, time, current_rhs);

			rhs_assembler_->set_bc(
				boundary_.local_boundary, boundary_.boundary_nodes, n_b_samples, boundary_.local_neumann_boundary, current_rhs, sol, time);

			StiffnessMatrix A = mass_ / bdf->beta_dt() + stiffness;
			Eigen::VectorXd b = (mass_ * bdf->weighted_sum_x_prevs()) / bdf->beta_dt();
			for (int i : boundary_.boundary_nodes)
				b[i] = 0;
			b += current_rhs;

			solve_linear_system(solver, A, b, args["output"]["advanced"]["spectrum"].get<bool>() && t == time_steps, sol);

			bdf->update_quantities(sol);
			save_timestep(time, t, t0, dt, sol);
			save_step_state(t0, dt, t, time_integrator.get());

			logger().info("{}/{}  t={}", t, time_steps, time);
			notify_time_step(t, time_steps, t0, dt);
		}
	}

	void ScalarVarForm::solve_problem(Eigen::MatrixXd &sol)
	{
		stats.spectrum.setZero();

		igl::Timer timer;
		timer.start();
		logger().info("Solving {}", primary_assembler_->name());

		{
			POLYFEM_SCOPED_TIMER("Setup RHS");

			if (sol.size() <= 0)
				prepare_initial_solution(sol);

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
