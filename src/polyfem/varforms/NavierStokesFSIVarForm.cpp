#include "NavierStokesFSIVarForm.hpp"

#include <polyfem/assembler/AssemblerUtils.hpp>
#include <polyfem/assembler/GenericProblem.hpp>
#include <polyfem/assembler/NavierStokesFSI.hpp>
#include <polyfem/io/Evaluator.hpp>
#include <polyfem/io/MatrixIO.hpp>
#include <polyfem/solver/ALSolver.hpp>
#include <polyfem/solver/NLProblem.hpp>
#include <polyfem/solver/forms/BodyForm.hpp>
#include <polyfem/solver/forms/ElasticForm.hpp>
#include <polyfem/solver/forms/NavierStokesForm.hpp>
#include <polyfem/solver/forms/NavierStokesFSIForm.hpp>
#include <polyfem/solver/forms/StackedForm.hpp>
#include <polyfem/solver/forms/lagrangian/BCLagrangianForm.hpp>
#include <polyfem/solver/forms/lagrangian/StackedAugmentedLagrangianForm.hpp>
#include <polyfem/time_integrator/BDF.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/MatrixUtils.hpp>

#include <igl/Timer.h>
#include <polysolve/linear/FEMSolver.hpp>
#include <polysolve/nonlinear/Solver.hpp>
#include <spdlog/fmt/fmt.h>

namespace polyfem::varform
{
	namespace
	{
		json first_material(const json &materials)
		{
			return materials.is_array() ? materials.at(0) : materials;
		}

		json mesh_material(const json &material)
		{
			json result = material.at("mesh_material");
			if (material.contains("id"))
				result["id"] = material["id"];
			return result;
		}

		json residual_solver_params(const json &input)
		{
			json params = input;
			params["solver"] = "Newton";
			params["line_search"]["method"] = "ResidualBacktracking";
			if (!params.contains("Newton") || params["Newton"].is_null())
				params["Newton"] = json::object();
			params["Newton"]["force_psd_projection"] = false;
			params["Newton"]["use_psd_projection"] = true;
			return params;
		}

		StiffnessMatrix residual_mass(
			const StiffnessMatrix &velocity_mass,
			const int pressure_size,
			const StiffnessMatrix &mesh_mass,
			const bool add_average)
		{
			const int mesh_offset = velocity_mass.rows() + pressure_size;
			const int total = mesh_offset + mesh_mass.rows() + (add_average ? 1 : 0);
			std::vector<Eigen::Triplet<double>> entries;
			entries.reserve(velocity_mass.nonZeros() + pressure_size + mesh_mass.nonZeros() + (add_average ? 1 : 0));
			for (int k = 0; k < velocity_mass.outerSize(); ++k)
				for (StiffnessMatrix::InnerIterator it(velocity_mass, k); it; ++it)
					entries.emplace_back(it.row(), it.col(), it.value());
			for (int i = 0; i < pressure_size; ++i)
				entries.emplace_back(velocity_mass.rows() + i, velocity_mass.rows() + i, 1);
			for (int k = 0; k < mesh_mass.outerSize(); ++k)
				for (StiffnessMatrix::InnerIterator it(mesh_mass, k); it; ++it)
					entries.emplace_back(mesh_offset + it.row(), mesh_offset + it.col(), it.value());
			if (add_average)
				entries.emplace_back(total - 1, total - 1, 1);
			StiffnessMatrix result(total, total);
			result.setFromTriplets(entries.begin(), entries.end());
			result.makeCompressed();
			return result;
		}
	} // namespace

	void NavierStokesFSIVarForm::reset()
	{
		FluidVarForm::reset();
		mesh_displacement_space_id_ = -1;
		mesh_elastic_formulation_ = "NeoHookean";
		mesh_displacement_space_.reset();
		mesh_displacement_boundary_.reset();
		mesh_displacement_problem_ = nullptr;
		mesh_displacement_ass_vals_cache_.init_empty();
		mesh_displacement_mass_ass_vals_cache_.init_empty(true);
		mesh_displacement_pure_mass_ass_vals_cache_.init_empty(true);
		mesh_elastic_assembler_ = nullptr;
		mesh_mass_assembler_ = nullptr;
		mesh_pure_mass_assembler_ = nullptr;
		mesh_rhs_assembler_ = nullptr;
		mesh_rhs_.resize(0, 0);
		fluid_zero_rhs_.resize(0, 0);
		mesh_pure_mass_.resize(0, 0);
		ale_assemblers_.clear();
		mesh_displacement_time_integrator_ = nullptr;
		fsi_forms_.clear();
		fsi_al_forms_.clear();
		fsi_problem_ = nullptr;
		ale_form_ = nullptr;
		auxiliary_form_ = nullptr;
		mesh_elastic_form_ = nullptr;
		fluid_neumann_form_ = nullptr;
		mesh_body_form_ = nullptr;
		average_pressure_form_ = nullptr;
	}

	void NavierStokesFSIVarForm::init(
		const std::string &formulation,
		const Units &units,
		const json &args,
		const std::string &out_path)
	{
		if (!args.contains("time") || args["time"].is_null())
			log_and_throw_error("NavierStokesFSI is only available for time-dependent problems.");
		FluidVarForm::init(formulation, units, args, out_path);

		const json &materials = args.at("materials");
		const json material = first_material(materials);
		mesh_displacement_space_id_ = material.at("mesh_displacement_space_id").get<int>();
		if (mesh_displacement_space_id_ == velocity_space_id_
			|| mesh_displacement_space_id_ == pressure_space_id_)
			log_and_throw_error("NavierStokesFSI requires distinct velocity, pressure, and mesh-displacement FE spaces.");
		mesh_elastic_formulation_ = material.at("mesh_material").at("type").get<std::string>();
		if (!assembler::AssemblerUtils::is_elastic_material(mesh_elastic_formulation_))
			log_and_throw_error("NavierStokesFSI mesh_material must be an elastic material, got {}.", mesh_elastic_formulation_);

		if (materials.is_array())
			for (const json &entry : materials)
			{
				if (entry.at("mesh_displacement_space_id").get<int>() != mesh_displacement_space_id_)
					log_and_throw_error("All NavierStokesFSI materials must use the same mesh-displacement FE space.");
				if (entry.at("mesh_material").at("type").get<std::string>() != mesh_elastic_formulation_)
					log_and_throw_error("All NavierStokesFSI regions must use the same mesh elastic formulation.");
			}

		if (args.at("space").at("discr_order").is_array())
		{
			bool found = false;
			for (const json &entry : args.at("space").at("discr_order"))
				found |= entry.at("fe_space").get<int>() == mesh_displacement_space_id_;
			if (!found)
				log_and_throw_error("NavierStokesFSI discretization orders must name the mesh-displacement FE space.");
		}

		mesh_elastic_assembler_ = assembler::AssemblerUtils::make_assembler(mesh_elastic_formulation_);
		mesh_mass_assembler_ = std::make_shared<assembler::Mass>();
		mesh_pure_mass_assembler_ = std::make_shared<assembler::HRZMass>();
		ale_assemblers_ = {
			std::make_shared<assembler::NavierStokesFSIVelocity>(),
			std::make_shared<assembler::NavierStokesFSIMixed>(),
			std::make_shared<assembler::NavierStokesFSIPressure>(),
			std::make_shared<assembler::NavierStokesFSIInertia>()};

		mesh_displacement_problem_ = std::make_shared<assembler::GenericTensorProblem>("NavierStokesFSIMeshDisplacement");
		mesh_displacement_problem_->clear();
		mesh_displacement_problem_->set_parameters({{"is_time_dependent", true}}, root_path);
		auto boundary_conditions = args["boundary_conditions"];
		boundary_conditions["root_path"] = root_path;
		mesh_displacement_problem_->set_parameters(boundary_conditions, root_path);
		mesh_displacement_problem_->set_parameters(args["initial_conditions"], root_path);
		mesh_displacement_problem_->set_parameters(args["output"], root_path);
		mesh_displacement_problem_->set_units(*mesh_elastic_assembler_, units);
	}

	json NavierStokesFSIVarForm::mesh_material_args() const
	{
		if (args["materials"].is_array())
		{
			json result = json::array();
			for (const json &material : args["materials"])
				result.push_back(mesh_material(material));
			return result;
		}
		return mesh_material(args["materials"]);
	}

	json NavierStokesFSIVarForm::time_integrator_args(const int fe_space_id) const
	{
		const json &integrators = args["time"]["integrator"];
		if (!integrators.is_array())
			return integrators;
		for (const json &integrator : integrators)
			if (integrator.value("fe_space", -1) == fe_space_id)
			{
				json result = integrator;
				result.erase("fe_space");
				return result;
			}
		log_and_throw_error("Missing time integrator for FE space {}.", fe_space_id);
	}

	void NavierStokesFSIVarForm::load_mesh(const mesh::Mesh &mesh, const json &args)
	{
		FluidVarForm::load_mesh(mesh, args);
		std::vector<int> body_ids(mesh.n_elements());
		for (int e = 0; e < mesh.n_elements(); ++e)
			body_ids[e] = mesh.get_body_id(e);
		for (const auto &assembler : ale_assemblers_)
		{
			assembler->set_size(mesh.dimension());
			assembler->set_materials(body_ids, this->args["materials"], units, root_path);
		}
		const json mesh_materials = mesh_material_args();
		mesh_elastic_assembler_->set_size(mesh.dimension());
		mesh_elastic_assembler_->set_materials(body_ids, mesh_materials, units, root_path);
		mesh_mass_assembler_->set_size(mesh.dimension());
		mesh_mass_assembler_->set_materials(body_ids, mesh_materials, units, root_path);
		mesh_pure_mass_assembler_->set_size(mesh.dimension());
		mesh_displacement_problem_->init(mesh);
	}

	void NavierStokesFSIVarForm::build_basis(mesh::Mesh &mesh, const bool iso_parametric, const json &args)
	{
		FluidVarForm::build_basis(mesh, iso_parametric, args);
		Eigen::VectorXi orders;
		assign_discr_orders(args["space"]["discr_order"], mesh_displacement_space_id_, mesh, orders);
		build_fe_space(
			mesh, iso_parametric, orders,
			args["space"]["basis_type"], args["space"]["poly_basis_type"],
			*mesh_elastic_assembler_, mesh.dimension(),
			args["space"]["advanced"]["quadrature_order"],
			args["space"]["advanced"]["mass_quadrature_order"],
			args["space"]["advanced"]["use_corner_quadrature"],
			args["space"]["advanced"]["n_harmonic_samples"],
			args["space"]["advanced"]["integral_constraints"],
			mesh_displacement_space_, mesh_displacement_boundary_, space_.geometry);
		build_mesh_displacement_boundary(mesh);

		if (std::max({space_.n_bases, pressure_space_.n_bases, mesh_displacement_space_.n_bases})
			<= args["solver"]["advanced"]["cache_size"])
		{
			mesh_displacement_ass_vals_cache_.init(mesh.is_volume(), mesh_displacement_space_.basis_list(), space_.geometry_basis_list());
			mesh_displacement_mass_ass_vals_cache_.init(mesh.is_volume(), mesh_displacement_space_.basis_list(), space_.geometry_basis_list(), true);
			mesh_displacement_pure_mass_ass_vals_cache_.init(mesh.is_volume(), mesh_displacement_space_.basis_list(), space_.geometry_basis_list(), true);
		}
		else
		{
			mesh_displacement_ass_vals_cache_.init_empty();
			mesh_displacement_mass_ass_vals_cache_.init_empty(true);
			mesh_displacement_pure_mass_ass_vals_cache_.init_empty(true);
		}
		build_rhs_assembler();
		logger().info("n mesh displacement bases: {}", mesh_displacement_space_.n_bases);
	}

	void NavierStokesFSIVarForm::build_mesh_displacement_boundary(mesh::Mesh &mesh)
	{
		mesh_displacement_boundary_.clear_boundary_conditions();
		mesh_displacement_problem_->update_nodes(mesh_displacement_space_.space_in_node_to_node);
		mesh_displacement_problem_->setup_bc(
			mesh, assembler::BoundaryKind::Dirichlet, mesh_displacement_space_id_,
			mesh_displacement_space_.basis_list(), mesh_displacement_boundary_.total_local_boundary,
			mesh_displacement_boundary_.local_boundary, mesh_displacement_boundary_.boundary_nodes,
			mesh.dimension());
		std::vector<int> unused;
		mesh_displacement_problem_->setup_bc(
			mesh, assembler::BoundaryKind::Neumann, mesh_displacement_space_id_,
			mesh_displacement_space_.basis_list(), mesh_displacement_boundary_.total_local_boundary,
			mesh_displacement_boundary_.local_neumann_boundary, unused, mesh.dimension());
		mesh_displacement_problem_->setup_nodal_bc(mesh, assembler::BoundaryKind::Dirichlet, mesh_displacement_space_id_, mesh_displacement_space_.n_bases, mesh_displacement_boundary_.dirichlet_nodes);
		mesh_displacement_problem_->setup_nodal_bc(mesh, assembler::BoundaryKind::Neumann, mesh_displacement_space_id_, mesh_displacement_space_.n_bases, mesh_displacement_boundary_.neumann_nodes);
		for (const int node : mesh_displacement_boundary_.dirichlet_nodes)
		{
			const int tag = mesh.get_node_id(node);
			for (int d = 0; d < mesh.dimension(); ++d)
				if (mesh_displacement_problem_->is_nodal_dimension_dirichlet(node, tag, d, mesh_displacement_space_id_))
					mesh_displacement_boundary_.boundary_nodes.push_back(node * mesh.dimension() + d);
		}
		mesh_displacement_boundary_.normalize_boundary_nodes();
		rebuild_node_positions(mesh_displacement_space_.basis_list(), mesh_displacement_boundary_.dirichlet_nodes, mesh_displacement_boundary_.dirichlet_nodes_position);
		rebuild_node_positions(mesh_displacement_space_.basis_list(), mesh_displacement_boundary_.neumann_nodes, mesh_displacement_boundary_.neumann_nodes_position);
	}

	void NavierStokesFSIVarForm::build_rhs_assembler()
	{
		FluidVarForm::build_rhs_assembler();
		if (mesh_displacement_space_.n_bases <= 0 || !mesh_)
			return;
		json solver_params = args["solver"]["linear"];
		if (!solver_params.contains("Pardiso"))
			solver_params["Pardiso"] = {};
		solver_params["Pardiso"]["mtype"] = -2;
		mesh_rhs_assembler_ = std::make_shared<assembler::RhsAssembler>(
			*mesh_elastic_assembler_, *mesh_, nullptr,
			mesh_displacement_boundary_.dirichlet_nodes, mesh_displacement_boundary_.neumann_nodes,
			mesh_displacement_boundary_.dirichlet_nodes_position, mesh_displacement_boundary_.neumann_nodes_position,
			mesh_displacement_space_.n_bases, mesh_->dimension(),
			mesh_displacement_space_.basis_list(), space_.geometry_basis_list(),
			mesh_displacement_mass_ass_vals_cache_, *mesh_displacement_problem_,
			args["space"]["advanced"]["bc_method"], solver_params,
			mesh_displacement_space_id_);
	}

	void NavierStokesFSIVarForm::assemble_rhs(const mesh::Mesh &mesh)
	{
		FluidVarForm::assemble_rhs(mesh);
		assert(mesh_rhs_assembler_);
		mesh_rhs_assembler_->assemble(mesh_mass_assembler_->density(), mesh_rhs_);
		mesh_rhs_ *= -1;
		const Eigen::MatrixXd velocity_rhs = rhs_.topRows(primary_ndof());
		rhs_.setZero(total_ndof(), 1);
		rhs_.topRows(primary_ndof()) = velocity_rhs;
		rhs_.middleRows(mesh_displacement_offset(), mesh_displacement_ndof()) = mesh_rhs_;
	}

	void NavierStokesFSIVarForm::assemble_mass_mat(const mesh::Mesh &mesh, const json &args)
	{
		FluidVarForm::assemble_mass_mat(mesh, args);
		mesh_pure_mass_assembler_->assemble(
			mesh.is_volume(), mesh_displacement_space_.n_bases,
			mesh_displacement_space_.basis_list(), space_.geometry_basis_list(),
			mesh_displacement_pure_mass_ass_vals_cache_, 0, mesh_pure_mass_, true);
	}

	int NavierStokesFSIVarForm::mesh_displacement_ndof() const
	{
		return mesh_ ? mesh_displacement_space_.n_bases * mesh_->dimension() : 0;
	}

	int NavierStokesFSIVarForm::total_ndof() const
	{
		return primary_ndof() + pressure_space_.n_bases + mesh_displacement_ndof() + (use_avg_pressure ? 1 : 0);
	}

	void NavierStokesFSIVarForm::prepare_fsi_initial_solution(Eigen::MatrixXd &sol) const
	{
		if (sol.size() == 0)
		{
			Eigen::MatrixXd velocity, mesh_displacement;
			const std::string state_path = resolve_input_path(args["input"]["data"]["state"]);
			const bool loaded_velocity = read_initial_x_from_file(
				state_path, "u", args["input"]["data"]["reorder"],
				space_.space_in_node_to_node, mesh_->dimension(), velocity);
			const bool loaded_mesh_displacement = read_initial_x_from_file(
				state_path, "mesh_u", args["input"]["data"]["reorder"],
				mesh_displacement_space_.space_in_node_to_node, mesh_->dimension(), mesh_displacement);
			if (!loaded_velocity)
				rhs_assembler_->initial_solution(velocity);
			if (!loaded_mesh_displacement)
				mesh_rhs_assembler_->initial_solution(mesh_displacement);
			sol.setZero(total_ndof(), 1);
			sol.topRows(primary_ndof()) = velocity.topRows(primary_ndof()).leftCols(1);
			sol.middleRows(mesh_displacement_offset(), mesh_displacement_ndof()) =
				mesh_displacement.topRows(mesh_displacement_ndof()).leftCols(1);
		}
		else
		{
			if (sol.cols() > 1)
				sol.conservativeResize(Eigen::NoChange, 1);
			if (sol.rows() != total_ndof())
			{
				const Eigen::MatrixXd input = sol;
				sol.setZero(total_ndof(), 1);
				const int rows = std::min<int>(input.rows(), primary_ndof() + pressure_space_.n_bases);
				if (rows > 0)
					sol.topRows(rows) = input.topRows(rows);
			}
		}
	}

	void NavierStokesFSIVarForm::build_forms(Eigen::MatrixXd &sol, const double t)
	{
		const int dim = mesh_->dimension();
		const Eigen::VectorXd velocity = sol.topRows(primary_ndof());
		const Eigen::VectorXd mesh_displacement = sol.middleRows(mesh_displacement_offset(), mesh_displacement_ndof());

		auto velocity_bdf = time_integrator::ImplicitTimeIntegrator::construct_bdf_integrator(
			time_integrator_args(velocity_space_id_), time_integrator::ImplicitTimeIntegrator::DynamicOrder::First);
		auto mesh_bdf = time_integrator::ImplicitTimeIntegrator::construct_bdf_integrator(
			time_integrator_args(mesh_displacement_space_id_), time_integrator::ImplicitTimeIntegrator::DynamicOrder::First);
		Eigen::MatrixXd velocity_initial_velocity, mesh_initial_velocity;
		rhs_assembler_->initial_velocity(velocity_initial_velocity);
		mesh_rhs_assembler_->initial_velocity(mesh_initial_velocity);
		Eigen::MatrixXd velocity_history = velocity;
		Eigen::MatrixXd velocity_history_velocity = velocity_initial_velocity;
		Eigen::MatrixXd velocity_history_acceleration = Eigen::MatrixXd::Zero(primary_ndof(), 1);
		Eigen::MatrixXd mesh_history = mesh_displacement;
		Eigen::MatrixXd mesh_history_velocity = mesh_initial_velocity;
		Eigen::MatrixXd mesh_history_acceleration = Eigen::MatrixXd::Zero(mesh_displacement_ndof(), 1);
		const std::string state_path = resolve_input_path(args["input"]["data"]["state"]);
		if (read_initial_x_from_file(
				state_path, "u", args["input"]["data"]["reorder"],
				space_.space_in_node_to_node, dim, velocity_history))
		{
			if (!read_initial_x_from_file(
					state_path, "v", args["input"]["data"]["reorder"],
					space_.space_in_node_to_node, dim, velocity_history_velocity))
				velocity_history_velocity.setZero(velocity_history.rows(), velocity_history.cols());
			if (!read_initial_x_from_file(
					state_path, "a", args["input"]["data"]["reorder"],
					space_.space_in_node_to_node, dim, velocity_history_acceleration))
				velocity_history_acceleration.setZero(velocity_history.rows(), velocity_history.cols());
		}
		if (read_initial_x_from_file(
				state_path, "mesh_u", args["input"]["data"]["reorder"],
				mesh_displacement_space_.space_in_node_to_node, dim, mesh_history))
		{
			if (!read_initial_x_from_file(
					state_path, "mesh_v", args["input"]["data"]["reorder"],
					mesh_displacement_space_.space_in_node_to_node, dim, mesh_history_velocity))
				mesh_history_velocity.setZero(mesh_history.rows(), mesh_history.cols());
			if (!read_initial_x_from_file(
					state_path, "mesh_a", args["input"]["data"]["reorder"],
					mesh_displacement_space_.space_in_node_to_node, dim, mesh_history_acceleration))
				mesh_history_acceleration.setZero(mesh_history.rows(), mesh_history.cols());
		}
		velocity_bdf->init(velocity_history, velocity_history_velocity, velocity_history_acceleration, dt);
		mesh_bdf->init(mesh_history, mesh_history_velocity, mesh_history_acceleration, dt);
		time_integrator = velocity_bdf;
		mesh_displacement_time_integrator_ = mesh_bdf;

		ale_form_ = std::make_shared<solver::NavierStokesFSIForm>(
			total_ndof(), space_.n_bases, pressure_space_.n_bases, mesh_displacement_space_.n_bases,
			space_.basis_list(), pressure_space_.basis_list(), mesh_displacement_space_.basis_list(),
			space_.geometry_basis_list(), ass_vals_cache_, pressure_ass_vals_cache_, mesh_displacement_ass_vals_cache_,
			ale_assemblers_, time_integrator.get(), mesh_displacement_time_integrator_.get(),
			t, dt, mesh_->is_volume(),
			[this](const int element, const Eigen::MatrixXd &points, const double time, Eigen::MatrixXd &value) {
				problem->rhs(*primary_assembler_, *mesh_, element, points, time, value, velocity_space_id_);
			});
		const int gorder = mesh_->orders().size() == 0 ? 1 : mesh_->orders().maxCoeff();
		const QuadratureOrders velocity_samples = n_boundary_samples(space_.disc_orders.maxCoeff(), gorder);
		ale_form_->set_velocity_tilde_updater(
			[this, velocity_samples](const double time, const Eigen::VectorXd &, Eigen::VectorXd &target) {
				Eigen::MatrixXd projected = target;
				const std::vector<mesh::LocalBoundary> empty_neumann;
				rhs_assembler_->set_bc(boundary_.local_boundary, boundary_.boundary_nodes, velocity_samples, empty_neumann, projected, Eigen::MatrixXd(), time);
				target = projected.col(0);
			});

		auxiliary_form_ = std::make_shared<solver::StackedForm>();
		const auto velocity_block = auxiliary_form_->add_block(primary_ndof());
		const auto pressure_block = auxiliary_form_->add_block(pressure_space_.n_bases);
		const auto mesh_block = auxiliary_form_->add_block(mesh_displacement_ndof());

		const solver::ElementInversionCheck check = args["solver"]["advanced"]["check_inversion"];
		mesh_elastic_form_ = std::make_shared<solver::ElasticForm>(
			mesh_displacement_space_.n_bases, *mesh_displacement_space_.bases, space_.geometry_basis_list(),
			*mesh_elastic_assembler_, mesh_displacement_ass_vals_cache_, t, dt, mesh_->is_volume(),
			args["solver"]["advanced"]["jacobian_threshold"], check);
		auxiliary_form_->add(mesh_block, mesh_elastic_form_);

		fluid_zero_rhs_ = Eigen::MatrixXd::Zero(primary_ndof(), 1);
		fluid_neumann_form_ = std::make_shared<solver::BodyForm>(
			primary_ndof(), 0, boundary_.boundary_nodes, boundary_.local_boundary,
			boundary_.local_neumann_boundary, velocity_samples, fluid_zero_rhs_, *rhs_assembler_,
			mass_assembler_->density(), false, true);
		fluid_neumann_form_->update_quantities(t, velocity);
		auxiliary_form_->add(velocity_block, fluid_neumann_form_);

		const QuadratureOrders mesh_samples = n_boundary_samples(mesh_displacement_space_.disc_orders.maxCoeff(), gorder);
		mesh_body_form_ = std::make_shared<solver::BodyForm>(
			mesh_displacement_ndof(), 0,
			mesh_displacement_boundary_.boundary_nodes, mesh_displacement_boundary_.local_boundary,
			mesh_displacement_boundary_.local_neumann_boundary, mesh_samples,
			mesh_rhs_, *mesh_rhs_assembler_, mesh_mass_assembler_->density(), false, true);
		mesh_body_form_->update_quantities(t, mesh_displacement);
		auxiliary_form_->add(mesh_block, mesh_body_form_);

		if (use_avg_pressure)
		{
			const auto average_block = auxiliary_form_->add_block(1);
			average_pressure_form_ = std::make_shared<solver::AveragePressureForm>(pressure_space_.n_bases);
			auxiliary_form_->add(pressure_block, average_block, average_pressure_form_);
		}

		fsi_forms_ = {ale_form_, auxiliary_form_};
		for (const auto &form : fsi_forms_)
			form->set_output_dir(output_path);
		fsi_al_forms_.clear();
		if (!boundary_.boundary_nodes.empty() || !mesh_displacement_boundary_.boundary_nodes.empty())
		{
			auto stacked_al = std::make_shared<solver::StackedAugmentedLagrangianForm>();
			const auto velocity_al = stacked_al->add_block(primary_ndof());
			stacked_al->add_block(pressure_space_.n_bases);
			const auto mesh_al = stacked_al->add_block(mesh_displacement_ndof());
			if (use_avg_pressure)
				stacked_al->add_block(1);
			if (!boundary_.boundary_nodes.empty())
				stacked_al->add(velocity_al, std::make_shared<solver::BCLagrangianForm>(
												 primary_ndof(), boundary_.boundary_nodes, boundary_.local_boundary, boundary_.local_neumann_boundary,
												 velocity_samples, pure_mass_, *rhs_assembler_, 0, true, t));
			if (!mesh_displacement_boundary_.boundary_nodes.empty())
				stacked_al->add(mesh_al, std::make_shared<solver::BCLagrangianForm>(
											 mesh_displacement_ndof(), mesh_displacement_boundary_.boundary_nodes,
											 mesh_displacement_boundary_.local_boundary, mesh_displacement_boundary_.local_neumann_boundary,
											 mesh_samples, mesh_pure_mass_, *mesh_rhs_assembler_, 0, true, t));
			fsi_al_forms_.push_back(stacked_al);
		}

		fsi_problem_ = std::make_shared<solver::NLProblem>(
			total_ndof(), nullptr, t, fsi_forms_, fsi_al_forms_,
			polysolve::linear::Solver::create(args["solver"]["linear"], logger()),
			units.characteristic_length(), 1,
			residual_mass(pure_mass_, pressure_space_.n_bases, mesh_pure_mass_, use_avg_pressure),
			dim, true);
		fsi_problem_->init(sol);
		fsi_problem_->update_quantities(t, sol);
		update_transient_form_weights();
		stats.solver_info = json::array();
	}

	void NavierStokesFSIVarForm::update_transient_form_weights()
	{
		const double scale = time_integrator->acceleration_scaling();
		if (fluid_neumann_form_)
			fluid_neumann_form_->set_weight(scale);
		if (average_pressure_form_)
			average_pressure_form_->set_weight(scale);
	}

	void NavierStokesFSIVarForm::solve_nonlinear_step(const int step, Eigen::MatrixXd &sol)
	{
		const json nonlinear_params = residual_solver_params(args["solver"]["nonlinear"]);
		const json al_params = residual_solver_params(args["solver"]["augmented_lagrangian"]["nonlinear"]);
		std::shared_ptr<polysolve::nonlinear::Solver> nonlinear_solver = polysolve::nonlinear::Solver::create(
			nonlinear_params, args["solver"]["linear"], units.characteristic_length(), logger());
		solver::ALSolver al_solver(
			fsi_al_forms_, args["solver"]["augmented_lagrangian"]["initial_weight"],
			args["solver"]["augmented_lagrangian"]["scaling"],
			args["solver"]["augmented_lagrangian"]["max_weight"],
			args["solver"]["augmented_lagrangian"]["eta"], [](const Eigen::VectorXd &) {});
		al_solver.post_subsolve = [&](const double weight) {
			stats.solver_info.push_back({{"type", weight > 0 ? "al" : "rc"}, {"t", step}, {"info", nonlinear_solver->info()}});
			if (weight > 0)
				stats.solver_info.back()["weight"] = weight;
			save_subsolve(stats.solver_info.size(), step, sol);
		};
		if (!fsi_al_forms_.empty())
			al_solver.solve_al(*fsi_problem_, sol, al_params, args["solver"]["linear"], units.characteristic_length(), nonlinear_solver);
		al_solver.solve_reduced(*fsi_problem_, sol, nonlinear_params, args["solver"]["linear"], units.characteristic_length(), nonlinear_solver);
	}

	void NavierStokesFSIVarForm::solve_problem(Eigen::MatrixXd &sol)
	{
		igl::Timer timer;
		timer.start();
		prepare_fsi_initial_solution(sol);
		build_forms(sol, t0 + dt);
		save_timestep(t0, 0, t0, dt, sol);
		for (int step = 1; step <= time_steps; ++step)
		{
			const double time = t0 + step * dt;
			logger().info("{}/{} steps, dt={}s t={}s", step, time_steps, dt, time);
			solve_nonlinear_step(step, sol);
			time_integrator->update_quantities(sol.topRows(primary_ndof()));
			mesh_displacement_time_integrator_->update_quantities(
				sol.middleRows(mesh_displacement_offset(), mesh_displacement_ndof()));
			update_transient_form_weights();
			fsi_problem_->update_quantities(t0 + (step + 1) * dt, sol);
			save_timestep(time, step, t0, dt, sol);
			save_step_state(t0, dt, step, time_integrator.get());
			save_mesh_integrator_state(step);
			notify_time_step(step, time_steps, t0, dt);
		}
		timer.stop();
		timings.solving_time = timer.getElapsedTime();
	}

	void NavierStokesFSIVarForm::save_mesh_integrator_state(const int step) const
	{
		assert(mesh_displacement_time_integrator_);
		const std::string state_path = resolve_output_path(
			fmt::format(args["output"]["data"]["state"].get<std::string>(), output_file_index(step)));
		if (state_path.empty())
			return;

		const auto save_history = [&](const std::string &name, const std::deque<Eigen::VectorXd> &history) {
			Eigen::MatrixXd values(history.front().size(), history.size());
			for (int i = 0; i < int(history.size()); ++i)
				values.col(i) = history[i];
			io::write_matrix(state_path, name, values, /*replace=*/false);
		};
		save_history("mesh_u", mesh_displacement_time_integrator_->x_prevs());
		save_history("mesh_v", mesh_displacement_time_integrator_->v_prevs());
		save_history("mesh_a", mesh_displacement_time_integrator_->a_prevs());
	}

	std::vector<io::OutputField> NavierStokesFSIVarForm::output_fields(
		const io::OutputSample &sample,
		const Eigen::MatrixXd &solution,
		const io::OutputFieldOptions &options) const
	{
		std::vector<io::OutputField> fields = FluidVarForm::output_fields(sample, solution, options);
		if (!mesh_ || solution.rows() < mesh_displacement_offset() + mesh_displacement_ndof()
			|| !options.export_field("mesh_displacement"))
			return fields;

		const int dim = mesh_->dimension();
		const Eigen::MatrixXd mesh_displacement =
			solution.middleRows(mesh_displacement_offset(), mesh_displacement_ndof());
		const bool has_element_samples =
			sample.local_points.rows() > 0 && sample.local_points.rows() == sample.element_ids.size();
		const int output_rows = sample.points.rows() > 0
									? sample.points.rows()
									: std::max<int>(sample.local_points.rows(), sample.node_ids.size());
		Eigen::MatrixXd values;

		if (has_element_samples)
		{
			values.setZero(output_rows, dim);
			for (int i = 0; i < sample.local_points.rows(); ++i)
			{
				const int element = sample.element_ids(i);
				if (element < 0)
					continue;
				Eigen::MatrixXd local_value, local_gradient;
				io::Evaluator::interpolate_at_local_vals(
					*mesh_, dim,
					mesh_displacement_space_.basis_list(), space_.geometry_basis_list(),
					element, sample.local_points.row(i), mesh_displacement,
					local_value, local_gradient);
				for (int d = 0; d < dim; ++d)
					values(i, d) = local_value(d);
			}
		}
		else if (sample.node_ids.size() > 0)
		{
			values.resize(sample.node_ids.size(), dim);
			for (int i = 0; i < sample.node_ids.size(); ++i)
			{
				const int node = sample.node_ids(i);
				if (node < 0 || node * dim + dim > mesh_displacement.rows())
					return fields;
				values.row(i) = mesh_displacement.block(node * dim, 0, dim, 1).transpose();
			}
		}
		else
		{
			return fields;
		}

		fields.push_back({"mesh_displacement", values, io::OutputField::Association::Point});
		return fields;
	}
} // namespace polyfem::varform
