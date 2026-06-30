#include "FluidVarForm.hpp"

#include <polyfem/assembler/AssemblerUtils.hpp>
#include <polyfem/assembler/GenericProblem.hpp>
#include <polyfem/assembler/Stokes.hpp>
#include <polyfem/basis/LagrangeBasis2d.hpp>
#include <polyfem/basis/LagrangeBasis3d.hpp>
#include <polyfem/io/Evaluator.hpp>
#include <polyfem/io/MatrixIO.hpp>
#include <polyfem/mesh/mesh2D/Mesh2D.hpp>
#include <polyfem/mesh/mesh3D/Mesh3D.hpp>
#include <polyfem/problem/KernelProblem.hpp>
#include <polyfem/problem/ProblemFactory.hpp>
#include <polyfem/refinement/APriori.hpp>
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

	void FluidVarForm::reset()
	{
		VarForm::reset();
		space_.reset();
		pressure_space_.reset();
		velocity_space_id_ = -1;
		pressure_space_id_ = -1;
		boundary_.reset();
		pressure_boundary_.reset();
		ass_vals_cache_.init_empty();
		pressure_ass_vals_cache_.init_empty();
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
		mixed_assembler_ = nullptr;
		pressure_assembler_ = nullptr;
		use_avg_pressure = true;
		t0 = 0;
		time_steps = 0;
		dt = 0;
		time_integrator = nullptr;
	}

	void FluidVarForm::init(const std::string &formulation, const Units &units, const json &args, const std::string &out_path)
	{
		VarForm::init(formulation, units, args, out_path);
		const bool is_time_dependent = args.contains("time") && !args["time"].is_null();
		const json &discr_orders = args.at("space").at("discr_order");

		const json &materials = args.at("materials");
		if (materials.is_array() && materials.empty())
			log_and_throw_error("Fluid formulations require at least one material.");
		const json &first_material = materials.is_array() ? materials.at(0) : materials;
		velocity_space_id_ = first_material.at("velocity_space_id").get<int>();
		pressure_space_id_ = first_material.at("pressure_space_id").get<int>();
		if (velocity_space_id_ == pressure_space_id_)
			log_and_throw_error("Fluid velocity and pressure must use different FE space IDs.");

		if (discr_orders.is_array())
		{
			bool has_velocity_space = false;
			bool has_pressure_space = false;
			for (const json &entry : discr_orders)
			{
				const int fe_space_id = entry.at("fe_space").get<int>();
				has_velocity_space |= fe_space_id == velocity_space_id_;
				has_pressure_space |= fe_space_id == pressure_space_id_;
			}
			if (!has_velocity_space || !has_pressure_space)
				log_and_throw_error("Fluid discretization-order lists must explicitly name the velocity and pressure FE spaces.");
		}

		if (materials.is_array())
		{
			for (const json &material : materials)
			{
				if (material.at("velocity_space_id").get<int>() != velocity_space_id_
					|| material.at("pressure_space_id").get<int>() != pressure_space_id_)
					log_and_throw_error("All fluid materials must use the same velocity and pressure FE space IDs.");
			}
		}

		primary_assembler_ = assembler::AssemblerUtils::make_assembler(formulation);
		mass_assembler_ = std::make_shared<assembler::Mass>();
		pure_mass_assembler_ = std::make_shared<assembler::HRZMass>();
		mixed_assembler_ = assembler::AssemblerUtils::make_mixed_assembler(formulation);
		pressure_assembler_ = assembler::AssemblerUtils::make_assembler(assembler::AssemblerUtils::other_assembler_name(formulation));

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
				problem = std::make_shared<problem::KernelProblem>("Kernel", *primary_assembler_);
				problem->clear();
			}
			else
			{
				problem = problem::ProblemFactory::factory().get_problem(args["preset_problem"]["type"]);
				problem->clear();
			}
			problem->set_parameters(args["preset_problem"], root_path);
		}

		problem->set_units(*primary_assembler_, units);

		t0 = is_time_dependent ? args["time"]["t0"].get<double>() : 0.0;
		time_steps = is_time_dependent ? args["time"]["time_steps"].get<int>() : 0;
		dt = is_time_dependent ? args["time"]["dt"].get<double>() : 0.0;

		assert(primary_assembler_->is_fluid());
	}

	void FluidVarForm::save_json(const Eigen::MatrixXd &solution, std::ostream &out) const
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

	io::OutputSpace FluidVarForm::output_space() const
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

	io::OutStatsData FluidVarForm::compute_errors(const Eigen::MatrixXd &solution)
	{
		if (!args["output"]["advanced"]["compute_error"])
			return stats;

		double tend = 0;
		if (!args["time"].is_null())
			tend = args["time"]["tend"];

		Eigen::MatrixXd velocity, pressure;
		split_solution(solution, velocity, pressure);
		stats.compute_errors(space_.n_bases, space_.basis_list(), space_.geometry_basis_list(), *mesh_, *problem, tend, velocity);
		return stats;
	}

	void FluidVarForm::export_data(const Eigen::MatrixXd &solution) const
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

		Eigen::MatrixXd velocity, pressure;
		split_solution(solution, velocity, pressure);

		const std::string solution_path = resolve_output_path(args["output"]["data"]["solution"]);
		if (!solution_path.empty())
		{
			const int primary_rows = std::min<int>(velocity.rows(), primary_ndof());
			const Eigen::MatrixXd primary_solution = velocity.topRows(primary_rows);
			if (opts.reorder_output && space_.space_in_node_to_node.size() > 0)
			{
				const Eigen::MatrixXd nodal_solution = utils::unflatten(primary_solution, mesh_->dimension());
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
				space_.disc_orders, space_.disc_ordersq, *primary_assembler_, velocity, tend,
				stress, mises);
			if (!stress_path.empty())
				io::write_matrix(stress_path, stress);
			if (!mises_path.empty())
				io::write_matrix(mises_path, mises);
		}
	}

	void FluidVarForm::load_mesh(const mesh::Mesh &mesh, const json &args)
	{
		set_materials(*primary_assembler_, mesh.dimension());
		set_materials(*mass_assembler_, mesh.dimension());
		pure_mass_assembler_->set_size(mass_assembler_->size());
		problem->init(mesh);

		if (mixed_assembler_)
			mixed_assembler_->set_size(mesh.dimension());
		if (pressure_assembler_)
			set_materials(*pressure_assembler_, 1);
	}

	int FluidVarForm::primary_ndof() const
	{
		return mesh_ ? space_.n_bases * mesh_->dimension() : 0;
	}

	int FluidVarForm::pressure_block_size() const
	{
		return pressure_space_.n_bases + ((use_avg_pressure && primary_assembler_ && primary_assembler_->is_fluid()) ? 1 : 0);
	}

	int FluidVarForm::stacked_ndof() const
	{
		return primary_ndof() + pressure_block_size();
	}

	void FluidVarForm::build_rhs_assembler()
	{
		json rhs_solver_params = args["solver"]["linear"];
		if (!rhs_solver_params.contains("Pardiso"))
			rhs_solver_params["Pardiso"] = {};
		rhs_solver_params["Pardiso"]["mtype"] = -2;

		rhs_assembler_ = std::make_shared<assembler::RhsAssembler>(
			*primary_assembler_, *mesh_, nullptr, // no obstacle for the rhs assembler
			boundary_.dirichlet_nodes, boundary_.neumann_nodes,
			boundary_.dirichlet_nodes_position, boundary_.neumann_nodes_position,
			space_.n_bases, mesh_->dimension(), space_.basis_list(), space_.geometry_basis_list(), mass_ass_vals_cache_, *problem,
			args["space"]["advanced"]["bc_method"],
			rhs_solver_params,
			velocity_space_id_);
	}

	void FluidVarForm::build_basis(mesh::Mesh &mesh, const bool iso_parametric, const json &args)
	{
		assert(problem);
		assert(primary_assembler_);
		assert(mass_assembler_);
		assert(pure_mass_assembler_);

		Eigen::VectorXi space_disc_orders;
		assign_discr_orders(args["space"]["discr_order"], velocity_space_id_, mesh, space_disc_orders);

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
			mesh.dimension(),
			args["space"]["advanced"]["quadrature_order"],
			args["space"]["advanced"]["mass_quadrature_order"],
			args["space"]["advanced"]["use_corner_quadrature"],
			args["space"]["advanced"]["n_harmonic_samples"],
			args["space"]["advanced"]["integral_constraints"],
			space_,
			boundary_);

		problem->update_nodes(space_.space_in_node_to_node);
		mesh.update_nodes(space_.space_in_node_to_node);

		const auto &current_bases = space_.geometry_basis_list();
		if (args["space"]["advanced"]["count_flipped_els"])
			stats.count_flipped_elements(mesh, current_bases);

		const int n_samples = 10;
		stats.compute_mesh_size(mesh, current_bases, n_samples, args["output"]["advanced"]["curved_mesh_size"]);

		logger().info("flipped elements {}", stats.n_flipped);
		logger().info("h: {}", stats.mesh_size);

		if (space_.disc_orders.maxCoeff() != space_.disc_orders.minCoeff())
			log_and_throw_error("p refinement not supported in mixed formulation!");
		if (!space_.poly_edge_to_data.empty())
			log_and_throw_error("Polygonal bases are not supported in mixed formulations!");

		if (space_.n_bases <= args["solver"]["advanced"]["cache_size"])
		{
			igl::Timer cache_timer;
			cache_timer.start();
			logger().info("Building cache...");
			ass_vals_cache_.init(mesh.is_volume(), space_.basis_list(), current_bases);
			mass_ass_vals_cache_.init(mesh.is_volume(), space_.basis_list(), current_bases, true);
			pure_mass_ass_vals_cache_.init(mesh.is_volume(), space_.basis_list(), current_bases, true);
			logger().info(" took {}s", cache_timer.getElapsedTime());
		}
		else
		{
			ass_vals_cache_.init_empty();
			mass_ass_vals_cache_.init_empty(true);
			pure_mass_ass_vals_cache_.init_empty(true);
		}

		const auto &all_boundary = boundary_.total_local_boundary;
		const int prev_bases = space_.n_bases;
		const int prev_b_size = int(all_boundary.size());
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

		const bool has_neumann = !boundary_.local_neumann_boundary.empty() || int(boundary_.local_boundary.size()) < prev_b_size;
		use_avg_pressure = !has_neumann;

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

	void FluidVarForm::assemble_rhs(const mesh::Mesh &mesh)
	{
		build_rhs_assembler();

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

		assert(rhs_assembler_ != nullptr);
		rhs_assembler_->assemble(mass_assembler_->density(), rhs_);
		rhs_ *= -1;

		timings.assigning_rhs_time = timer.getElapsedTime();
		logger().info(" took {}s", timings.assigning_rhs_time);

		const int prev_size = rhs_.rows();
		rhs_.conservativeResize(prev_size + pressure_block_size(), rhs_.cols());
		rhs_.bottomRows(pressure_block_size()).setZero();
	}

	void FluidVarForm::assemble_mass_mat(const mesh::Mesh &mesh, const json &args)
	{
		if (!problem->is_time_dependent())
		{
			avg_mass_ = 1;
			timings.assembling_mass_mat_time = 0;
			if (!primary_assembler_->is_linear())
				pure_mass_assembler_->assemble(mesh.is_volume(), space_.n_bases, space_.basis_list(), space_.geometry_basis_list(), pure_mass_ass_vals_cache_, 0, pure_mass_, true);
			return;
		}

		mass_.resize(0, 0);
		igl::Timer timer;
		timer.start();
		logger().info("Assembling mass mat...");

		StiffnessMatrix velocity_mass;
		mass_assembler_->assemble(mesh.is_volume(), space_.n_bases, space_.basis_list(), space_.geometry_basis_list(), mass_ass_vals_cache_, 0, velocity_mass, true);
		if (!primary_assembler_->is_linear())
			pure_mass_assembler_->assemble(mesh.is_volume(), space_.n_bases, space_.basis_list(), space_.geometry_basis_list(), pure_mass_ass_vals_cache_, 0, pure_mass_, true);

		std::vector<Eigen::Triplet<double>> blocks;
		blocks.reserve(velocity_mass.nonZeros());
		for (int k = 0; k < velocity_mass.outerSize(); ++k)
			for (StiffnessMatrix::InnerIterator it(velocity_mass, k); it; ++it)
				blocks.emplace_back(it.row(), it.col(), it.value());

		mass_.resize(primary_ndof(), primary_ndof());
		mass_.setFromTriplets(blocks.begin(), blocks.end());
		mass_.makeCompressed();

		avg_mass_ = 0;
		for (int k = 0; k < velocity_mass.outerSize(); ++k)
			for (StiffnessMatrix::InnerIterator it(velocity_mass, k); it; ++it)
			{
				assert(it.col() == k);
				avg_mass_ += it.value();
			}
		avg_mass_ /= std::max(1, int(velocity_mass.rows()));
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

	void FluidVarForm::prepare_initial_solution(Eigen::MatrixXd &sol) const
	{
		if (sol.size() <= 0)
		{
			assert(rhs_assembler_ != nullptr);
			const bool was_solution_loaded = read_initial_x_from_file(
				resolve_input_path(args["input"]["data"]["state"]), "u",
				args["input"]["data"]["reorder"], space_.space_in_node_to_node,
				mesh_->dimension(), sol);

			if (!was_solution_loaded)
			{
				if (problem->is_time_dependent())
					rhs_assembler_->initial_solution(sol);
				else
				{
					sol.resize(rhs_.size(), 1);
					sol.setZero();
				}
			}
		}
		if (sol.cols() > 1)
			sol.conservativeResize(Eigen::NoChange, 1);
		sol.conservativeResize(stacked_ndof(), sol.cols());
		sol.bottomRows(pressure_block_size()).setZero();
	}

	void FluidVarForm::split_solution(const Eigen::MatrixXd &stacked, Eigen::MatrixXd &primary, Eigen::MatrixXd &pressure) const
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

	void FluidVarForm::build_stiffness_mat(StiffnessMatrix &stiffness)
	{
		igl::Timer timer;
		timer.start();
		logger().info("Assembling stiffness mat...");

		StiffnessMatrix velocity_stiffness, mixed_stiffness, pressure_stiffness;
		primary_assembler_->assemble(mesh_->is_volume(), space_.n_bases, space_.basis_list(), space_.geometry_basis_list(), ass_vals_cache_, 0, velocity_stiffness);
		mixed_assembler_->assemble(mesh_->is_volume(), pressure_space_.n_bases, space_.n_bases, pressure_space_.basis_list(), space_.basis_list(), space_.geometry_basis_list(), pressure_ass_vals_cache_, ass_vals_cache_, 0, mixed_stiffness);
		pressure_assembler_->assemble(mesh_->is_volume(), pressure_space_.n_bases, pressure_space_.basis_list(), space_.geometry_basis_list(), pressure_ass_vals_cache_, 0, pressure_stiffness);

		assembler::AssemblerUtils::merge_mixed_matrices(
			space_.n_bases, pressure_space_.n_bases, mesh_->dimension(), use_avg_pressure,
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
			boundary_.boundary_nodes,
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
		std::vector<io::OutputField> fields;
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
						*mesh_, field_dim, space_.basis_list(), space_.geometry_basis_list(),
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
					*mesh_, pressure_space_.basis_list(), space_.geometry_basis_list(), sample, pressure, values,
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

	void StokesVarForm::solve_static_linear(Eigen::MatrixXd &sol)
	{
		auto solver = polysolve::linear::Solver::create(args["solver"]["linear"], logger());
		logger().info("{}...", solver->name());

		const int gdiscr_order = mesh_->orders().size() <= 0 ? 1 : mesh_->orders().maxCoeff();
		const QuadratureOrders boundary_samples = n_boundary_samples(space_.disc_orders.maxCoeff(), gdiscr_order);
		rhs_assembler_->set_bc(
			boundary_.local_boundary, boundary_.boundary_nodes, boundary_samples,
			boundary_.local_neumann_boundary, rhs_);

		StiffnessMatrix A;
		build_stiffness_mat(A);
		Eigen::VectorXd b = rhs_;
		solve_linear_system(solver, A, b, args["output"]["advanced"]["spectrum"], sol);
	}

	void StokesVarForm::solve_transient_linear(Eigen::MatrixXd &sol)
	{
		auto solver = polysolve::linear::Solver::create(args["solver"]["linear"], logger());
		logger().info("{}...", solver->name());

		Eigen::MatrixXd velocity, pressure;
		split_solution(sol, velocity, pressure);

		auto bdf = make_bdf_time_integrator();
		bdf->init(
			velocity,
			Eigen::MatrixXd::Zero(velocity.rows(), velocity.cols()),
			Eigen::MatrixXd::Zero(velocity.rows(), velocity.cols()),
			dt);
		time_integrator = bdf;

		save_timestep(t0, 0, t0, dt, sol);

		Eigen::MatrixXd current_rhs = rhs_;
		StiffnessMatrix stiffness, expanded_mass;
		build_stiffness_mat(stiffness);
		expand_primary_matrix(stacked_ndof(), mass_, expanded_mass);
		const int gdiscr_order = mesh_->orders().size() <= 0 ? 1 : mesh_->orders().maxCoeff();
		const QuadratureOrders boundary_samples = n_boundary_samples(space_.disc_orders.maxCoeff(), gdiscr_order);

		for (int t = 1; t <= time_steps; ++t)
		{
			const double time = t0 + t * dt;
			rhs_assembler_->compute_energy_grad(
				boundary_.local_boundary, boundary_.boundary_nodes, mass_assembler_->density(), boundary_samples, boundary_.local_neumann_boundary, rhs_, time,
				current_rhs);
			rhs_assembler_->set_bc(
				boundary_.local_boundary, boundary_.boundary_nodes, boundary_samples, boundary_.local_neumann_boundary, current_rhs, velocity, time);

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
			b.head(primary_ndof()) = (mass_ * bdf->weighted_sum_x_prevs()) / bdf->beta_dt();
			for (int i : boundary_.boundary_nodes)
				b[i] = 0;
			b += current_rhs;

			solve_linear_system(solver, A, b, args["output"]["advanced"]["spectrum"].get<bool>() && t == time_steps, sol);
			split_solution(sol, velocity, pressure);
			bdf->update_quantities(velocity.col(0));

			save_timestep(time, t, t0, dt, sol);
			save_step_state(t0, dt, t, time_integrator.get());
			logger().info("{}/{}  t={}", t, time_steps, time);
			notify_time_step(t, time_steps, t0, dt);
		}
	}

	void StokesVarForm::solve_problem(Eigen::MatrixXd &sol)
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

	void NavierStokesVarForm::solve_static(Eigen::MatrixXd &sol)
	{
		assert(rhs_assembler_ != nullptr);
		const int gdiscr_order = mesh_->orders().size() <= 0 ? 1 : mesh_->orders().maxCoeff();
		const QuadratureOrders boundary_samples = n_boundary_samples(space_.disc_orders.maxCoeff(), gdiscr_order);
		rhs_assembler_->set_bc(
			boundary_.local_boundary, boundary_.boundary_nodes, boundary_samples, boundary_.local_neumann_boundary, rhs_);

		auto velocity_stokes_assembler = std::make_shared<assembler::StokesVelocity>();
		set_materials(*velocity_stokes_assembler, mesh_->dimension());

		Eigen::VectorXd x;
		solver::NavierStokesSolver ns_solver(args["solver"]);
		ns_solver.minimize(
			space_.n_bases, pressure_space_.n_bases,
			space_.basis_list(), pressure_space_.basis_list(),
			space_.geometry_basis_list(),
			*velocity_stokes_assembler,
			*dynamic_cast<assembler::NavierStokesVelocity *>(primary_assembler_.get()),
			*mixed_assembler_,
			*pressure_assembler_,
			ass_vals_cache_,
			pressure_ass_vals_cache_,
			boundary_.boundary_nodes,
			use_avg_pressure,
			mesh_->dimension(),
			mesh_->is_volume(),
			rhs_,
			x);
		sol = x;
		ns_solver.get_info(stats.solver_info);
	}

	void NavierStokesVarForm::solve_transient(Eigen::MatrixXd &sol)
	{

		Eigen::MatrixXd velocity, pressure;
		split_solution(sol, velocity, pressure);

		auto bdf = make_bdf_time_integrator();
		bdf->init(
			velocity,
			Eigen::MatrixXd::Zero(velocity.rows(), velocity.cols()),
			Eigen::MatrixXd::Zero(velocity.rows(), velocity.cols()),
			dt);
		time_integrator = bdf;

		save_timestep(t0, 0, t0, dt, sol);

		Eigen::MatrixXd current_rhs = rhs_;
		StiffnessMatrix velocity_mass;
		mass_assembler_->assemble(mesh_->is_volume(), space_.n_bases, space_.basis_list(), space_.geometry_basis_list(), mass_ass_vals_cache_, 0, velocity_mass, true);

		StiffnessMatrix velocity_stiffness, mixed_stiffness, pressure_stiffness;
		auto velocity_stokes_assembler = std::make_shared<assembler::StokesVelocity>();
		set_materials(*velocity_stokes_assembler, mesh_->dimension());

		mixed_assembler_->assemble(mesh_->is_volume(), pressure_space_.n_bases, space_.n_bases, pressure_space_.basis_list(), space_.basis_list(), space_.geometry_basis_list(), pressure_ass_vals_cache_, ass_vals_cache_, 0, mixed_stiffness);
		pressure_assembler_->assemble(mesh_->is_volume(), pressure_space_.n_bases, pressure_space_.basis_list(), space_.geometry_basis_list(), pressure_ass_vals_cache_, 0, pressure_stiffness);

		solver::TransientNavierStokesSolver ns_solver(args["solver"]);
		for (int t = 1; t <= time_steps; ++t)
		{
			const double time = t0 + t * dt;
			velocity_stokes_assembler->assemble(mesh_->is_volume(), space_.n_bases, space_.basis_list(), space_.geometry_basis_list(), ass_vals_cache_, time, velocity_stiffness);

			logger().info("{}/{} steps, dt={}s t={}s", t, time_steps, dt, time);

			const Eigen::VectorXd prev_sol = bdf->weighted_sum_x_prevs();
			const int gdiscr_order = mesh_->orders().size() <= 0 ? 1 : mesh_->orders().maxCoeff();
			const QuadratureOrders boundary_samples = n_boundary_samples(space_.disc_orders.maxCoeff(), gdiscr_order);
			rhs_assembler_->compute_energy_grad(
				boundary_.local_boundary, boundary_.boundary_nodes, mass_assembler_->density(), boundary_samples, boundary_.local_neumann_boundary, rhs_, time, current_rhs);
			rhs_assembler_->set_bc(
				boundary_.local_boundary, boundary_.boundary_nodes, boundary_samples, boundary_.local_neumann_boundary, current_rhs, velocity, time);

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
				space_.n_bases, pressure_space_.n_bases,
				time,
				space_.basis_list(), space_.geometry_basis_list(),
				*dynamic_cast<assembler::NavierStokesVelocity *>(primary_assembler_.get()),
				ass_vals_cache_,
				boundary_.boundary_nodes,
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
			notify_time_step(t, time_steps, t0, dt);
		}

		ns_solver.get_info(stats.solver_info);
	}

	void NavierStokesVarForm::solve_problem(Eigen::MatrixXd &sol)
	{
		stats.spectrum.setZero();
		igl::Timer timer;
		timer.start();
		logger().info("Solving {}", primary_assembler_->name());

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
