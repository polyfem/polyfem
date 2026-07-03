#include "ThermoElasticVarForm.hpp"

#include <polyfem/assembler/AssemblerUtils.hpp>
#include <polyfem/assembler/GenericProblem.hpp>
#include <polyfem/assembler/Laplacian.hpp>
#include <polyfem/assembler/MatParams.hpp>

#include <polyfem/io/Evaluator.hpp>
#include <polyfem/io/MatrixIO.hpp>

#include <polyfem/mesh/GeometryReader.hpp>

#include <polyfem/refinement/APriori.hpp>

#include <polyfem/solver/ALSolver.hpp>
#include <polyfem/solver/NLProblem.hpp>
#include <polyfem/solver/forms/BodyForm.hpp>
#include <polyfem/solver/forms/ElasticForm.hpp>
#include <polyfem/solver/forms/InertiaForm.hpp>
#include <polyfem/solver/forms/MixedAssemblerForm.hpp>
#include <polyfem/solver/forms/StackedForm.hpp>
#include <polyfem/solver/forms/lagrangian/AugmentedLagrangianForm.hpp>
#include <polyfem/solver/forms/lagrangian/BCLagrangianForm.hpp>
#include <polyfem/solver/forms/lagrangian/StackedAugmentedLagrangianForm.hpp>

#include <polyfem/time_integrator/ImplicitTimeIntegrator.hpp>

#include <polyfem/utils/Jacobian.hpp>
#include <polyfem/utils/JSONUtils.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/MatrixUtils.hpp>
#include <polyfem/utils/Timer.hpp>

#include <igl/Timer.h>

#include <polysolve/linear/Solver.hpp>
#include <polysolve/nonlinear/Solver.hpp>

#include <algorithm>
#include <cassert>
#include <limits>
#include <map>
#include <vector>

namespace polyfem::varform
{
	namespace
	{
		json first_material(const json &materials)
		{
			return materials.is_array() ? materials.front() : materials;
		}

		void disable_newton_psd_projection(json &solver_params)
		{
			const auto disable_for_newton = [](json &params) {
				if (!params.contains("Newton") || params["Newton"].is_null())
					params["Newton"] = json::object();
				params["Newton"]["use_psd_projection"] = false;
			};

			if (solver_params.contains("solver") && solver_params["solver"].is_array())
			{
				for (json &strategy : solver_params["solver"])
				{
					const std::string type = strategy.value("type", "");
					if (type == "Newton" || type == "SparseNewton" || type == "sparse_newton"
						|| type == "DenseNewton" || type == "dense_newton")
						disable_for_newton(strategy);
				}
			}
			else
			{
				disable_for_newton(solver_params);
			}
		}

		void assert_same_space_ids(
			const json &materials,
			const int displacement_space_id,
			const int temperature_space_id)
		{
			for (const json &material : utils::json_as_array(materials))
			{
				if (material.at("displacement_space_id").get<int>() != displacement_space_id
					|| material.at("temperature_space_id").get<int>() != temperature_space_id)
				{
					log_and_throw_error("All ThermoElasticity materials must use the same FE space ids.");
				}
			}
		}

		json elastic_material_from_thermo_material(const json &material)
		{
			if (!material.contains("elastic_material") || !material["elastic_material"].is_object())
				log_and_throw_error("ThermoElasticity requires elastic_material to be an elastic material object.");

			json elastic_material = material["elastic_material"];
			const std::string type = elastic_material.value("type", "");
			if (!assembler::AssemblerUtils::is_elastic_material(type))
				log_and_throw_error("ThermoElasticity elastic_material must be an elastic material, got '{}'.", type);

			if (material.contains("id"))
				elastic_material["id"] = material["id"];
			if (material.contains("rho"))
				elastic_material["rho"] = material["rho"];

			return elastic_material;
		}

		json solver_params_for_residual_mode(const json &solver_params, const bool is_residual)
		{
			json params = solver_params;
			if (is_residual)
				disable_newton_psd_projection(params);
			return params;
		}

		std::string elastic_formulation_from_thermo_materials(const json &materials)
		{
			std::string formulation;
			for (const json &material : utils::json_as_array(materials))
			{
				const json elastic_material = elastic_material_from_thermo_material(material);
				const std::string type = elastic_material["type"];
				if (formulation.empty())
					formulation = type;
				else if (formulation != type)
					formulation = "MultiModels";
			}

			return formulation;
		}

		StiffnessMatrix block_diag(const StiffnessMatrix &a, const StiffnessMatrix &b)
		{
			std::vector<Eigen::Triplet<double>> entries;
			entries.reserve(a.nonZeros() + b.nonZeros());

			for (int k = 0; k < a.outerSize(); ++k)
				for (StiffnessMatrix::InnerIterator it(a, k); it; ++it)
					entries.emplace_back(it.row(), it.col(), it.value());

			for (int k = 0; k < b.outerSize(); ++k)
				for (StiffnessMatrix::InnerIterator it(b, k); it; ++it)
					entries.emplace_back(a.rows() + it.row(), a.cols() + it.col(), it.value());

			StiffnessMatrix out(a.rows() + b.rows(), a.cols() + b.cols());
			out.setFromTriplets(entries.begin(), entries.end());
			out.makeCompressed();
			return out;
		}

		StiffnessMatrix identity_mass(const int size)
		{
			return utils::sparse_identity(size, size);
		}

	} // namespace

	void ThermoElasticVarForm::reset()
	{
		NonlinearElasticVarForm::reset();
		temperature_space_.reset();
		temperature_boundary_.reset();
		temperature_problem_ = nullptr;
		temperature_ass_vals_cache_.init_empty();
		temperature_mass_ass_vals_cache_.init_empty(true);
		temperature_pure_mass_ass_vals_cache_.init_empty(true);
		temperature_assembler_ = nullptr;
		thermoelastic_assembler_ = nullptr;
		temperature_mass_assembler_ = nullptr;
		temperature_pure_mass_assembler_ = nullptr;
		temperature_rhs_assembler_ = nullptr;
		temperature_rhs_density_ = std::make_shared<assembler::NoDensity>();
		temperature_mass_.resize(0, 0);
		temperature_pure_mass_.resize(0, 0);
		stacked_lumped_mass_.resize(0, 0);
		temperature_rhs_.resize(0, 0);
		temperature_time_integrator_ = nullptr;
		temperature_form_ = nullptr;
		thermoelastic_form_ = nullptr;
		temperature_body_form_ = nullptr;
		temperature_inertia_form_ = nullptr;
		stacked_form_ = nullptr;
		displacement_space_id_ = -1;
		temperature_space_id_ = -1;
		elastic_formulation_ = "NeoHookean";
	}

	void ThermoElasticVarForm::init(
		const std::string &formulation,
		const Units &units,
		const json &args,
		const std::string &out_path)
	{
		VarForm::init(formulation, units, args, out_path);
		read_material_space_ids(args);

		const bool is_time_dependent = args.contains("time") && !args["time"].is_null();

		primary_assembler_ = assembler::AssemblerUtils::make_assembler(elastic_formulation_);
		if (args["solver"]["advanced"]["check_inversion"] == "Conservative")
		{
			if (auto elastic_assembler = std::dynamic_pointer_cast<assembler::ElasticityAssembler>(primary_assembler_))
				elastic_assembler->set_use_robust_jacobian();
		}

		temperature_assembler_ = std::make_shared<assembler::Laplacian>("conductivity");
		thermoelastic_assembler_ = assembler::AssemblerUtils::make_mixed_nl_assembler("ThermoElasticity");
		mass_assembler_ = std::make_shared<assembler::Mass>();
		pure_mass_assembler_ = std::make_shared<assembler::HRZMass>();
		temperature_mass_assembler_ = std::make_shared<assembler::Mass>(std::make_shared<assembler::ThermalMassDensity>());
		temperature_pure_mass_assembler_ = std::make_shared<assembler::HRZMass>();

		problem = std::make_shared<assembler::GenericTensorProblem>("ThermoElasticDisplacement");
		problem->clear();
		temperature_problem_ = std::make_shared<assembler::GenericScalarProblem>("ThermoElasticTemperature");
		temperature_problem_->clear();

		json tmp;
		tmp["is_time_dependent"] = is_time_dependent;
		problem->set_parameters(tmp, root_path);
		temperature_problem_->set_parameters(tmp, root_path);

		auto bc = args["boundary_conditions"];
		bc["root_path"] = root_path;
		problem->set_parameters(bc, root_path);
		temperature_problem_->set_parameters(bc, root_path);
		problem->set_parameters(args["initial_conditions"], root_path);
		temperature_problem_->set_parameters(args["initial_conditions"], root_path);
		problem->set_parameters(args["output"], root_path);
		temperature_problem_->set_parameters(args["output"], root_path);

		problem->set_units(*primary_assembler_, units);
		temperature_problem_->set_units(*temperature_assembler_, units);

		t0 = is_time_dependent ? args["time"]["t0"].get<double>() : 0.0;
		time_steps = is_time_dependent ? args["time"]["time_steps"].get<int>() : 0;
		dt = is_time_dependent ? args["time"]["dt"].get<double>() : 0.0;
		contact_dhat_was_explicit_ = args["contact"].value("_dhat_was_explicit", false);
		this->args["contact"].erase("_dhat_was_explicit");
	}

	void ThermoElasticVarForm::read_material_space_ids(const json &args)
	{
		const json material = first_material(args.at("materials"));
		displacement_space_id_ = material.at("displacement_space_id").get<int>();
		temperature_space_id_ = material.at("temperature_space_id").get<int>();
		if (displacement_space_id_ == temperature_space_id_)
			log_and_throw_error("ThermoElasticity requires distinct displacement and temperature FE spaces.");

		elastic_formulation_ = elastic_formulation_from_thermo_materials(args.at("materials"));
		assert_same_space_ids(args.at("materials"), displacement_space_id_, temperature_space_id_);
	}

	json ThermoElasticVarForm::elastic_material_args() const
	{
		if (args["materials"].is_array())
		{
			json materials = json::array();
			for (const json &material : args["materials"])
				materials.push_back(elastic_material_from_thermo_material(material));
			return materials;
		}

		return elastic_material_from_thermo_material(args["materials"]);
	}

	json ThermoElasticVarForm::time_integrator_args(const int fe_space_id) const
	{
		const json &integrators = args["time"]["integrator"];
		if (!integrators.is_array())
			return integrators;

		for (const json &integrator : integrators)
		{
			if (integrator.value("fe_space", -1) == fe_space_id)
			{
				json copy = integrator;
				copy.erase("fe_space");
				return copy;
			}
		}

		log_and_throw_error("Missing time integrator for FE space {}.", fe_space_id);
	}

	void ThermoElasticVarForm::load_mesh(const mesh::Mesh &mesh, const json &args)
	{
		assert(mesh_);
		std::vector<int> body_ids(mesh.n_elements());
		for (int i = 0; i < mesh.n_elements(); ++i)
			body_ids[i] = mesh.get_body_id(i);

		const json elastic_materials = elastic_material_args();

		primary_assembler_->set_size(mesh.dimension());
		primary_assembler_->set_materials(body_ids, elastic_materials, units, root_path);
		thermoelastic_assembler_->set_size(mesh.dimension());
		thermoelastic_assembler_->set_materials(body_ids, args["materials"], units, root_path);
		mass_assembler_->set_size(mesh.dimension());
		mass_assembler_->set_materials(body_ids, elastic_materials, units, root_path);
		pure_mass_assembler_->set_size(mass_assembler_->size());

		temperature_assembler_->set_size(1);
		temperature_assembler_->set_materials(body_ids, args["materials"], units, root_path);
		temperature_mass_assembler_->set_size(1);
		temperature_mass_assembler_->set_materials(body_ids, args["materials"], units, root_path);
		temperature_pure_mass_assembler_->set_size(1);

		problem->init(mesh);
		temperature_problem_->init(mesh);

		logger().info("Loading obstacles...");
		obstacle = mesh::read_obstacle_geometry(
			units,
			args["geometry"],
			utils::json_as_array(args["boundary_conditions"]["obstacle_displacements"]),
			utils::json_as_array(args["boundary_conditions"]["dirichlet_boundary"]),
			root_path, mesh.dimension());
	}

	void ThermoElasticVarForm::build_basis(mesh::Mesh &mesh, const bool iso_parametric, const json &args)
	{
		assert(problem);
		assert(temperature_problem_);
		assert(primary_assembler_);
		assert(temperature_assembler_);

		Eigen::VectorXi displacement_orders;
		assign_discr_orders(args["space"]["discr_order"], displacement_space_id_, mesh, displacement_orders);

		if (args["space"]["use_p_ref"])
		{
			refinement::APriori::p_refine(
				mesh,
				args["space"]["advanced"]["B"],
				args["space"]["advanced"]["h1_formula"],
				args["space"]["discr_order"],
				args["space"]["advanced"]["discr_order_max"],
				stats,
				displacement_orders);

			logger().info("min p: {} max p: {}", displacement_orders.minCoeff(), displacement_orders.maxCoeff());
		}

		build_fe_space(
			mesh,
			iso_parametric,
			displacement_orders,
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
		build_displacement_boundary(mesh);

		const int n_fe_bases = space_.n_bases;
		space_.n_bases += obstacle.n_vertices();

		logger().info("Building collision mesh...");
		build_collision_mesh(mesh, args);
		preprocess_contact_parameters();
		logger().info("Done!");

		for (int i = n_fe_bases; i < space_.n_bases; ++i)
		{
			for (int d = 0; d < mesh.dimension(); ++d)
				boundary_.boundary_nodes.push_back(i * mesh.dimension() + d);
		}
		boundary_.normalize_boundary_nodes();

		build_temperature_basis(mesh, iso_parametric, args);
		build_temperature_boundary(mesh);

		const auto &current_bases = space_.geometry_basis_list();
		if (args["space"]["advanced"]["count_flipped_els"])
			stats.count_flipped_elements(mesh, current_bases);

		const int n_samples = 10;
		stats.compute_mesh_size(mesh, current_bases, n_samples, args["output"]["advanced"]["curved_mesh_size"]);
		logger().info("flipped elements {}", stats.n_flipped);
		logger().info("h: {}", stats.mesh_size);

		if (std::max(space_.n_bases, temperature_space_.n_bases) <= args["solver"]["advanced"]["cache_size"])
		{
			igl::Timer timer;
			timer.start();
			logger().info("Building cache...");
			ass_vals_cache_.init(mesh.is_volume(), space_.basis_list(), current_bases);
			mass_ass_vals_cache_.init(mesh.is_volume(), space_.basis_list(), current_bases, true);
			pure_mass_ass_vals_cache_.init(mesh.is_volume(), space_.basis_list(), current_bases, true);
			temperature_ass_vals_cache_.init(mesh.is_volume(), temperature_space_.basis_list(), temperature_space_.geometry_basis_list());
			temperature_mass_ass_vals_cache_.init(mesh.is_volume(), temperature_space_.basis_list(), temperature_space_.geometry_basis_list(), true);
			temperature_pure_mass_ass_vals_cache_.init(mesh.is_volume(), temperature_space_.basis_list(), temperature_space_.geometry_basis_list(), true);
			logger().info(" took {}s", timer.getElapsedTime());
		}
		else
		{
			ass_vals_cache_.init_empty();
			mass_ass_vals_cache_.init_empty(true);
			pure_mass_ass_vals_cache_.init_empty(true);
			temperature_ass_vals_cache_.init_empty();
			temperature_mass_ass_vals_cache_.init_empty(true);
			temperature_pure_mass_ass_vals_cache_.init_empty(true);
		}
	}

	void ThermoElasticVarForm::build_displacement_boundary(mesh::Mesh &mesh)
	{
		boundary_.clear_boundary_conditions();

		problem->setup_bc(
			mesh,
			assembler::BoundaryKind::Dirichlet,
			displacement_space_id_,
			space_.basis_list(),
			boundary_.total_local_boundary,
			boundary_.local_boundary,
			boundary_.boundary_nodes,
			mesh.dimension());
		std::vector<int> unused_neumann_boundary_nodes;
		problem->setup_bc(
			mesh,
			assembler::BoundaryKind::Neumann,
			displacement_space_id_,
			space_.basis_list(),
			boundary_.total_local_boundary,
			boundary_.local_neumann_boundary,
			unused_neumann_boundary_nodes,
			mesh.dimension());

		problem->setup_nodal_bc(
			mesh,
			assembler::BoundaryKind::Dirichlet,
			displacement_space_id_,
			space_.n_bases,
			boundary_.dirichlet_nodes);
		problem->setup_nodal_bc(
			mesh,
			assembler::BoundaryKind::Neumann,
			displacement_space_id_,
			space_.n_bases,
			boundary_.neumann_nodes);

		for (const int n_id : boundary_.dirichlet_nodes)
		{
			const int tag = mesh.get_node_id(n_id);
			for (int d = 0; d < mesh.dimension(); ++d)
				if (problem->is_nodal_dimension_dirichlet(n_id, tag, d, displacement_space_id_))
					boundary_.boundary_nodes.push_back(n_id * mesh.dimension() + d);
		}

		boundary_.normalize_boundary_nodes();
		rebuild_node_positions(space_.basis_list(), boundary_.dirichlet_nodes, boundary_.dirichlet_nodes_position);
		rebuild_node_positions(space_.basis_list(), boundary_.neumann_nodes, boundary_.neumann_nodes_position);
	}

	void ThermoElasticVarForm::build_temperature_basis(mesh::Mesh &mesh, const bool iso_parametric, const json &args)
	{
		Eigen::VectorXi temperature_orders;
		assign_discr_orders(args["space"]["discr_order"], temperature_space_id_, mesh, temperature_orders);

		build_fe_space(
			mesh,
			iso_parametric,
			temperature_orders,
			args["space"]["basis_type"],
			args["space"]["poly_basis_type"],
			*temperature_assembler_,
			/*value_dim=*/1,
			args["space"]["advanced"]["quadrature_order"],
			args["space"]["advanced"]["mass_quadrature_order"],
			args["space"]["advanced"]["use_corner_quadrature"],
			args["space"]["advanced"]["n_harmonic_samples"],
			args["space"]["advanced"]["integral_constraints"],
			temperature_space_,
			temperature_boundary_,
			space_.geometry);

		logger().info("n temperature bases: {}", temperature_space_.n_bases);
	}

	void ThermoElasticVarForm::build_temperature_boundary(mesh::Mesh &mesh)
	{
		temperature_boundary_.clear_boundary_conditions();

		temperature_problem_->update_nodes(temperature_space_.space_in_node_to_node);

		temperature_problem_->setup_bc(
			mesh,
			assembler::BoundaryKind::Dirichlet,
			temperature_space_id_,
			temperature_space_.basis_list(),
			temperature_boundary_.total_local_boundary,
			temperature_boundary_.local_boundary,
			temperature_boundary_.boundary_nodes,
			/*value_dim=*/1);
		std::vector<int> unused_neumann_boundary_nodes;
		temperature_problem_->setup_bc(
			mesh,
			assembler::BoundaryKind::Neumann,
			temperature_space_id_,
			temperature_space_.basis_list(),
			temperature_boundary_.total_local_boundary,
			temperature_boundary_.local_neumann_boundary,
			unused_neumann_boundary_nodes,
			/*value_dim=*/1);

		temperature_problem_->setup_nodal_bc(
			mesh,
			assembler::BoundaryKind::Dirichlet,
			temperature_space_id_,
			temperature_space_.n_bases,
			temperature_boundary_.dirichlet_nodes);
		temperature_problem_->setup_nodal_bc(
			mesh,
			assembler::BoundaryKind::Neumann,
			temperature_space_id_,
			temperature_space_.n_bases,
			temperature_boundary_.neumann_nodes);

		for (const int n_id : temperature_boundary_.dirichlet_nodes)
			temperature_boundary_.boundary_nodes.push_back(n_id);

		temperature_boundary_.normalize_boundary_nodes();
		rebuild_node_positions(temperature_space_.basis_list(), temperature_boundary_.dirichlet_nodes, temperature_boundary_.dirichlet_nodes_position);
		rebuild_node_positions(temperature_space_.basis_list(), temperature_boundary_.neumann_nodes, temperature_boundary_.neumann_nodes_position);
	}

	void ThermoElasticVarForm::build_rhs_assembler()
	{
		json rhs_solver_params = args["solver"]["linear"];
		if (!rhs_solver_params.contains("Pardiso"))
			rhs_solver_params["Pardiso"] = {};
		rhs_solver_params["Pardiso"]["mtype"] = -2;

		solve_data.rhs_assembler = std::make_shared<assembler::RhsAssembler>(
			*primary_assembler_, *mesh_, &obstacle,
			boundary_.dirichlet_nodes, boundary_.neumann_nodes,
			boundary_.dirichlet_nodes_position, boundary_.neumann_nodes_position,
			space_.n_bases, mesh_->dimension(), space_.basis_list(), space_.geometry_basis_list(),
			mass_ass_vals_cache_, *problem,
			args["space"]["advanced"]["bc_method"],
			rhs_solver_params,
			displacement_space_id_);
		rhs_assembler_ = solve_data.rhs_assembler;

		temperature_rhs_assembler_ = std::make_shared<assembler::RhsAssembler>(
			*temperature_assembler_, *mesh_, nullptr,
			temperature_boundary_.dirichlet_nodes, temperature_boundary_.neumann_nodes,
			temperature_boundary_.dirichlet_nodes_position, temperature_boundary_.neumann_nodes_position,
			temperature_space_.n_bases, /*size=*/1,
			temperature_space_.basis_list(), temperature_space_.geometry_basis_list(),
			temperature_mass_ass_vals_cache_, *temperature_problem_,
			args["space"]["advanced"]["bc_method"],
			rhs_solver_params,
			temperature_space_id_);
	}

	void ThermoElasticVarForm::assemble_rhs(const mesh::Mesh &mesh)
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
		temperature_problem_->set_parameters(p_params, root_path);

		rhs_.resize(0, 0);
		temperature_rhs_.resize(0, 0);

		timer.start();
		logger().info("Assigning rhs...");

		build_rhs_assembler();
		assert(rhs_assembler_ != nullptr);
		assert(temperature_rhs_assembler_ != nullptr);
		assert(temperature_rhs_density_ != nullptr);
		rhs_assembler_->assemble(mass_assembler_->density(), rhs_);
		rhs_ *= -1;
		temperature_rhs_assembler_->assemble(*temperature_rhs_density_, temperature_rhs_);
		temperature_rhs_ *= -1;

		timings.assigning_rhs_time = timer.getElapsedTime();
		logger().info(" took {}s", timings.assigning_rhs_time);
	}

	void ThermoElasticVarForm::assemble_mass_mat(const mesh::Mesh &mesh, const json &args)
	{
		mass_.resize(0, 0);
		pure_mass_.resize(0, 0);
		temperature_mass_.resize(0, 0);
		temperature_pure_mass_.resize(0, 0);

		igl::Timer timer;
		timer.start();
		logger().info("Assembling mass mat...");

		mass_assembler_->assemble(mesh.is_volume(), space_.n_bases, space_.basis_list(), space_.geometry_basis_list(), mass_ass_vals_cache_, 0, mass_, true);
		pure_mass_assembler_->assemble(mesh.is_volume(), space_.n_bases, space_.basis_list(), space_.geometry_basis_list(), pure_mass_ass_vals_cache_, 0, pure_mass_, true);
		temperature_mass_assembler_->assemble(mesh.is_volume(), temperature_space_.n_bases, temperature_space_.basis_list(), temperature_space_.geometry_basis_list(), temperature_mass_ass_vals_cache_, 0, temperature_mass_, true);
		temperature_pure_mass_assembler_->assemble(mesh.is_volume(), temperature_space_.n_bases, temperature_space_.basis_list(), temperature_space_.geometry_basis_list(), temperature_pure_mass_ass_vals_cache_, 0, temperature_pure_mass_, true);

		assert(mass_.size() > 0);
		avg_mass_ = 0;
		for (int k = 0; k < mass_.outerSize(); ++k)
			for (StiffnessMatrix::InnerIterator it(mass_, k); it; ++it)
				avg_mass_ += it.value();
		avg_mass_ /= mass_.rows();
		logger().info("average mass {}", avg_mass_);

		if (args["solver"]["advanced"]["lump_mass_matrix"])
		{
			mass_ = utils::lump_matrix(mass_);
			temperature_mass_ = utils::lump_matrix(temperature_mass_);
		}

		stacked_lumped_mass_ = block_diag(
			pure_mass_.size() > 0 ? pure_mass_ : identity_mass(displacement_ndof()),
			temperature_pure_mass_.size() > 0 ? temperature_pure_mass_ : identity_mass(temperature_ndof()));

		timer.stop();
		timings.assembling_mass_mat_time = timer.getElapsedTime();
		logger().info(" took {}s", timings.assembling_mass_mat_time);

		stats.nn_zero = stacked_lumped_mass_.nonZeros();
		stats.num_dofs = stacked_lumped_mass_.rows();
		stats.mat_size = (long long)stacked_lumped_mass_.rows() * (long long)stacked_lumped_mass_.cols();
		logger().info("sparsity: {}/{}", stats.nn_zero, stats.mat_size);
	}

	void ThermoElasticVarForm::initial_temperature_solution(Eigen::MatrixXd &solution) const
	{
		assert(temperature_rhs_assembler_ != nullptr);

		const bool was_solution_loaded = read_initial_x_from_file(
			resolve_input_path(args["input"]["data"]["state"]), "temperature",
			args["input"]["data"]["reorder"], temperature_space_.space_in_node_to_node,
			/*dim=*/1, solution);

		if (!was_solution_loaded)
			temperature_rhs_assembler_->initial_solution(solution);
	}

	Eigen::MatrixXd ThermoElasticVarForm::stacked_solution(
		const Eigen::MatrixXd &displacement,
		const Eigen::MatrixXd &temperature) const
	{
		assert(displacement.rows() == displacement_ndof());
		assert(temperature.rows() == temperature_ndof());
		assert(displacement.cols() == temperature.cols());

		Eigen::MatrixXd solution(displacement.rows() + temperature.rows(), displacement.cols());
		solution << displacement, temperature;
		return solution;
	}

	void ThermoElasticVarForm::split_solution(
		const Eigen::MatrixXd &solution,
		Eigen::MatrixXd &displacement,
		Eigen::MatrixXd &temperature) const
	{
		assert(solution.rows() == total_ndof());
		displacement = solution.topRows(displacement_ndof());
		temperature = solution.bottomRows(temperature_ndof());
	}

	void ThermoElasticVarForm::build_forms(Eigen::MatrixXd &solution, const double t)
	{
		assert(solution.cols() == 1);
		assert(solution.rows() == total_ndof());

		const bool is_time_dependent = problem->is_time_dependent();
		const double form_dt = is_time_dependent ? dt : 0.0;

		stacked_form_ = std::make_shared<solver::StackedForm>();
		const auto displacement_block = stacked_form_->add_block(displacement_ndof());
		const auto temperature_block = stacked_form_->add_block(temperature_ndof());

		Eigen::MatrixXd displacement, temperature;
		split_solution(solution, displacement, temperature);

		if (is_time_dependent)
		{
			solve_data.time_integrator = time_integrator::ImplicitTimeIntegrator::construct_time_integrator(
				time_integrator_args(displacement_space_id_),
				time_integrator::ImplicitTimeIntegrator::DynamicOrder::Second);

			Eigen::MatrixXd displacement_solution, displacement_velocity, displacement_acceleration;
			initial_elastic_solution(displacement_solution);
			displacement_solution.col(0) = displacement;
			initial_velocity(displacement_velocity);
			initial_acceleration(displacement_acceleration);
			solve_data.time_integrator->init(displacement_solution, displacement_velocity, displacement_acceleration, dt);
		}
		else
		{
			solve_data.time_integrator = nullptr;
		}

		init_forms(args, mesh_->dimension(), displacement, t);
		for (const auto &form : forms)
		{
			assert(form);
			stacked_form_->add(displacement_block, form);
		}
		solve_data.al_form.clear();

		temperature_form_ = std::make_shared<solver::ElasticForm>(
			temperature_space_.n_bases, *temperature_space_.bases, temperature_space_.geometry_basis_list(),
			*temperature_assembler_, temperature_ass_vals_cache_, t, form_dt, mesh_->is_volume(),
			/*jacobian_threshold=*/0.0, solver::ElementInversionCheck::Discrete);
		stacked_form_->add(temperature_block, temperature_form_);

		assert(thermoelastic_assembler_);
		thermoelastic_form_ = std::make_shared<solver::MixedAssemblerForm>(
			space_.n_bases, temperature_space_.n_bases,
			space_.basis_list(), temperature_space_.basis_list(), space_.geometry_basis_list(),
			*thermoelastic_assembler_, ass_vals_cache_, temperature_ass_vals_cache_,
			t, form_dt, mesh_->is_volume());
		stacked_form_->add(displacement_block, temperature_block, thermoelastic_form_);

		const int gdiscr_order = mesh_->orders().size() <= 0 ? 1 : mesh_->orders().maxCoeff();
		const QuadratureOrders temperature_boundary_samples =
			n_boundary_samples(temperature_space_.disc_orders.maxCoeff(), gdiscr_order);
		temperature_body_form_ = std::make_shared<solver::BodyForm>(
			temperature_ndof(), 0,
			temperature_boundary_.boundary_nodes, temperature_boundary_.local_boundary,
			temperature_boundary_.local_neumann_boundary, temperature_boundary_samples,
			temperature_rhs_, *temperature_rhs_assembler_,
			*temperature_rhs_density_,
			/*is_formulation_mixed=*/false, is_time_dependent);
		temperature_body_form_->update_quantities(t, temperature);
		stacked_form_->add(temperature_block, temperature_body_form_);

		if (is_time_dependent)
		{
			temperature_time_integrator_ = time_integrator::ImplicitTimeIntegrator::construct_time_integrator(
				time_integrator_args(temperature_space_id_),
				time_integrator::ImplicitTimeIntegrator::DynamicOrder::First);

			Eigen::MatrixXd temperature_solution;
			initial_temperature_solution(temperature_solution);
			temperature_solution.col(0) = temperature;
			Eigen::MatrixXd temperature_velocity = Eigen::MatrixXd::Zero(temperature_solution.rows(), temperature_solution.cols());
			Eigen::MatrixXd temperature_acceleration = Eigen::MatrixXd::Zero(temperature_solution.rows(), temperature_solution.cols());
			temperature_time_integrator_->init(temperature_solution, temperature_velocity, temperature_acceleration, dt);

			temperature_inertia_form_ = std::make_shared<solver::InertiaForm>(temperature_mass_, *temperature_time_integrator_);
			if (!temperature_boundary_.boundary_nodes.empty())
			{
				temperature_inertia_form_->set_x_tilde_updater(
					[this, temperature_boundary_samples](
						const double t,
						const Eigen::VectorXd &,
						Eigen::VectorXd &target) {
						Eigen::MatrixXd projected_target = target;
						const std::vector<mesh::LocalBoundary> empty_neumann_boundary;
						temperature_rhs_assembler_->set_bc(
							temperature_boundary_.local_boundary,
							temperature_boundary_.boundary_nodes,
							temperature_boundary_samples,
							empty_neumann_boundary,
							projected_target,
							Eigen::MatrixXd(),
							t);
						assert(projected_target.cols() == 1);
						target = projected_target.col(0);
					});
			}
			stacked_form_->add(temperature_block, temperature_inertia_form_);

			update_transient_form_weights();
		}
		else
		{
			temperature_time_integrator_ = nullptr;
			temperature_inertia_form_ = nullptr;
		}

		forms.clear();
		forms.push_back(stacked_form_);
		for (const auto &form : forms)
			form->set_output_dir(output_path);

		solve_data.al_form.clear();
		if (!boundary_.boundary_nodes.empty() || !temperature_boundary_.boundary_nodes.empty())
		{
			auto stacked_al = std::make_shared<solver::StackedAugmentedLagrangianForm>();
			const auto displacement_al_block = stacked_al->add_block(displacement_block.size());
			const auto temperature_al_block = stacked_al->add_block(temperature_block.size());

			if (!boundary_.boundary_nodes.empty())
			{
				stacked_al->add(
					displacement_al_block,
					std::make_shared<solver::BCLagrangianForm>(
						displacement_block.size(),
						boundary_.boundary_nodes, boundary_.local_boundary, boundary_.local_neumann_boundary,
						elastic_boundary_samples(), mass_, *rhs_assembler_,
						obstacle.n_vertices() * mesh_->dimension(), is_time_dependent, t));
			}

			if (!temperature_boundary_.boundary_nodes.empty())
			{
				stacked_al->add(
					temperature_al_block,
					std::make_shared<solver::BCLagrangianForm>(
						temperature_block.size(),
						temperature_boundary_.boundary_nodes, temperature_boundary_.local_boundary,
						temperature_boundary_.local_neumann_boundary, temperature_boundary_samples,
						temperature_mass_, *temperature_rhs_assembler_,
						/*obstacle_ndof=*/0, is_time_dependent, t));
			}

			solve_data.al_form.push_back(stacked_al);
		}
	}

	void ThermoElasticVarForm::update_transient_form_weights()
	{
		assert(problem->is_time_dependent());
		assert(solve_data.time_integrator);
		assert(temperature_time_integrator_);

		solve_data.update_dt();

		const double displacement_scaling = solve_data.time_integrator->acceleration_scaling();
		const double temperature_scaling = temperature_time_integrator_->acceleration_scaling();

		if (temperature_form_)
			temperature_form_->set_weight(temperature_scaling);
		if (temperature_body_form_)
			temperature_body_form_->set_weight(temperature_scaling);
		if (thermoelastic_form_)
			thermoelastic_form_->set_row_weights(displacement_scaling, temperature_scaling);
	}

	void ThermoElasticVarForm::solve_nonlinear_step(const int step, Eigen::MatrixXd &solution)
	{
		assert(solve_data.nl_problem != nullptr);
		solver::NLProblem &nl_problem = *solve_data.nl_problem;

		const json nonlinear_params = solver_params_for_residual_mode(args["solver"]["nonlinear"], nl_problem.is_residual());
		const json al_nonlinear_params = solver_params_for_residual_mode(args["solver"]["augmented_lagrangian"]["nonlinear"], nl_problem.is_residual());

		std::shared_ptr<polysolve::nonlinear::Solver> nl_solver =
			polysolve::nonlinear::Solver::create(
				nonlinear_params, args["solver"]["linear"],
				units.characteristic_length(), logger());

		if (nl_problem.uses_lagging())
			nl_problem.init_lagging(solution);

		const auto update_displacement_barrier_stiffness = [&](const Eigen::VectorXd &x) {
			const Eigen::VectorXd displacement = x.head(displacement_ndof());
			solve_data.update_barrier_stiffness(displacement);
		};

		if (!solve_data.al_form.empty())
		{
			solver::ALSolver al_solver(
				solve_data.al_form,
				args["solver"]["augmented_lagrangian"]["initial_weight"],
				args["solver"]["augmented_lagrangian"]["scaling"],
				args["solver"]["augmented_lagrangian"]["max_weight"],
				args["solver"]["augmented_lagrangian"]["eta"],
				update_displacement_barrier_stiffness);

			al_solver.post_subsolve = [&](const double al_weight) {
				stats.solver_info.push_back(
					{{"type", al_weight > 0 ? "al" : "rc"},
					 {"t", step},
					 {"info", nl_solver->info()}});
				if (al_weight > 0)
					stats.solver_info.back()["weight"] = al_weight;
				save_subsolve(stats.solver_info.size(), step, solution);
			};

			al_solver.solve_al(
				nl_problem, solution, al_nonlinear_params,
				args["solver"]["linear"], units.characteristic_length(), nl_solver);
			al_solver.solve_reduced(
				nl_problem, solution, nonlinear_params,
				args["solver"]["linear"], units.characteristic_length(), nl_solver);
			return;
		}

		Eigen::VectorXd x = solution;
		nl_problem.init(x);
		update_displacement_barrier_stiffness(x);
		nl_problem.normalize_forms();
		nl_solver->minimize(nl_problem, x);
		nl_problem.finish();
		solution = x;
		stats.solver_info.push_back({{"type", "rc"}, {"t", step}, {"info", nl_solver->info()}});
		save_subsolve(stats.solver_info.size(), step, solution);
	}

	void ThermoElasticVarForm::solve_problem(Eigen::MatrixXd &sol)
	{
		stats.spectrum.setZero();

		igl::Timer timer;
		timer.start();
		logger().info("Solving ThermoElasticity");

		{
			POLYFEM_SCOPED_TIMER("Setup RHS");

			if (sol.size() <= 0)
			{
				Eigen::MatrixXd displacement, temperature;
				initial_elastic_solution(displacement);
				initial_temperature_solution(temperature);
				const int cols = std::max(displacement.cols(), temperature.cols());
				if (displacement.cols() != cols)
					displacement.conservativeResize(Eigen::NoChange, cols);
				if (temperature.cols() != cols)
					temperature.conservativeResize(Eigen::NoChange, cols);
				sol = stacked_solution(displacement, temperature);
			}

			if (sol.cols() > 1)
				sol.conservativeResize(Eigen::NoChange, 1);
		}

		build_forms(sol, problem->is_time_dependent() ? t0 + dt : 1.0);

		double characteristic_length = 0;
		if (args["solver"]["advanced"]["characteristic_length"] > 0)
			characteristic_length = args["solver"]["advanced"]["characteristic_length"];
		else
		{
			RowVectorNd min, max;
			mesh_->bounding_box(min, max);
			characteristic_length = (max - min).norm();
		}

		double characteristic_force_density = 0;
		if (args["solver"]["advanced"]["characteristic_force_density"] <= 0)
		{
			logger().warn("No user-specified force density was provided, defaulting to 10000.");
			characteristic_force_density = 10000;
		}
		else
			characteristic_force_density = args["solver"]["advanced"]["characteristic_force_density"];

		solve_data.nl_problem = std::make_shared<solver::NLProblem>(
			total_ndof(), nullptr, problem->is_time_dependent() ? t0 + dt : 1.0,
			forms, solve_data.al_form,
			polysolve::linear::Solver::create(args["solver"]["linear"], logger()),
			characteristic_length, characteristic_force_density,
			stacked_lumped_mass_.size() > 0 ? stacked_lumped_mass_ : identity_mass(total_ndof()),
			mesh_->dimension(),
			problem->is_time_dependent());
		solve_data.nl_problem->init(sol);
		solve_data.nl_problem->update_quantities(problem->is_time_dependent() ? t0 + dt : 1.0, sol);
		stats.solver_info = json::array();

		if (!problem->is_time_dependent())
		{
			solve_nonlinear_step(0, sol);
		}
		else
		{
			save_timestep(t0, 0, t0, dt, sol);
			for (int t = 1; t <= time_steps; ++t)
			{
				const double time = t0 + dt * t;
				solve_nonlinear_step(t, sol);

				save_timestep(time, t, t0, dt, sol);

				Eigen::MatrixXd displacement, temperature;
				split_solution(sol, displacement, temperature);
				solve_data.time_integrator->update_quantities(displacement);
				temperature_time_integrator_->update_quantities(temperature);
				update_transient_form_weights();
				solve_data.update_barrier_stiffness(displacement);
				solve_data.nl_problem->update_quantities(t0 + (t + 1) * dt, sol);

				logger().info("{}/{}  t={}", t, time_steps, time);
				notify_time_step(t, time_steps, t0, dt);
				save_step_state(t0, dt, t, nullptr);
			}
		}

		timer.stop();
		timings.solving_time = timer.getElapsedTime();
		logger().info(" took {}s", timings.solving_time);
	}

	io::OutStatsData ThermoElasticVarForm::compute_errors(const Eigen::MatrixXd &solution)
	{
		if (!args["output"]["advanced"]["compute_error"])
			return stats;

		Eigen::MatrixXd displacement, temperature;
		split_solution(solution, displacement, temperature);

		double tend = 0;
		if (!args["time"].is_null())
			tend = args["time"]["tend"];

		stats.compute_errors(space_.n_bases, space_.basis_list(), space_.geometry_basis_list(), *mesh_, *problem, tend, displacement);
		return stats;
	}

	std::vector<io::OutputField> ThermoElasticVarForm::output_fields(
		const io::OutputSample &sample,
		const Eigen::MatrixXd &solution,
		const io::OutputFieldOptions &options) const
	{
		Eigen::MatrixXd displacement, temperature;
		split_solution(solution, displacement, temperature);

		std::vector<io::OutputField> fields =
			NonlinearElasticVarForm::output_fields(sample, displacement, options);
		fields.erase(
			std::remove_if(
				fields.begin(), fields.end(),
				[](const io::OutputField &field) {
					return field.name == "solution" || field.name == "solution_gradient";
				}),
			fields.end());

		if (!mesh_ || temperature.size() <= 0)
			return fields;
		if (sample.domain == io::OutputSample::Domain::Contact)
			return fields;

		const bool has_element_samples = sample.local_points.rows() > 0 && sample.local_points.rows() == sample.element_ids.size();
		const int output_rows = sample.points.rows() > 0 ? sample.points.rows() : std::max<int>(sample.local_points.rows(), sample.node_ids.size());

		const auto has_field = [&](const std::string &name) {
			return std::any_of(fields.begin(), fields.end(), [&](const io::OutputField &field) {
				return field.name == name;
			});
		};

		const auto append_material_fields = [&](const assembler::Assembler &assembler) {
			const auto &paraview_options = args["output"]["paraview"]["options"];
			if (!paraview_options["material"] || !has_element_samples)
				return;

			const auto params = assembler.parameters();
			std::map<std::string, Eigen::MatrixXd> param_values;
			for (const auto &[p, _] : params)
				param_values[p].setZero(output_rows, 1);

			for (int i = 0; i < sample.local_points.rows(); ++i)
			{
				const int element_id = sample.element_ids(i);
				if (element_id < 0)
					continue;

				for (const auto &[p, func] : params)
					param_values.at(p)(i) = func(sample.local_points.row(i), sample.points.row(i), sample.time, element_id);
			}

			for (const auto &[name, values] : param_values)
				if (options.export_field(name) && !has_field(name))
					fields.push_back({name, values, io::OutputField::Association::Point});
		};

		const auto sample_temperature = [&](Eigen::MatrixXd &values, Eigen::MatrixXd *gradients = nullptr) -> bool {
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
						*mesh_, 1, temperature_space_.basis_list(), temperature_space_.geometry_basis_list(),
						element_id, sample.local_points.row(i), temperature, local_sol, local_grad);
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
					if (node_id < 0 || node_id >= temperature.rows())
						return false;
					values(i) = temperature(node_id);
				}
				return sample.points.rows() == 0 || sample.points.rows() == values.rows();
			}

			return false;
		};

		const bool export_temperature_gradient =
			!options.fields.empty() && options.export_field("temperature_gradient");
		if (options.export_field("temperature") || export_temperature_gradient)
		{
			Eigen::MatrixXd values, gradients;
			if (sample_temperature(values, export_temperature_gradient ? &gradients : nullptr))
			{
				if (options.export_field("temperature"))
					fields.push_back({"temperature", values, io::OutputField::Association::Point});
				if (export_temperature_gradient)
					fields.push_back({"temperature_gradient", gradients, io::OutputField::Association::Point});
			}
		}

		if (thermoelastic_assembler_)
			append_material_fields(*thermoelastic_assembler_);
		if (temperature_assembler_)
			append_material_fields(*temperature_assembler_);
		if (temperature_mass_assembler_)
			append_material_fields(*temperature_mass_assembler_);

		return fields;
	}
} // namespace polyfem::varform
