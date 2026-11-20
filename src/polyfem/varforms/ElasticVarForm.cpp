#include "ElasticVarForm.hpp"

#include <polyfem/assembler/ElementAssemblyValues.hpp>

#include <polyfem/assembler/MultiModel.hpp>

#include <polyfem/autogen/auto_p_bases.hpp>
#include <polyfem/autogen/auto_q_bases.hpp>
#include <polyfem/autogen/prism_bases.hpp>

#include <polyfem/basis/Basis.hpp>

#include <polyfem/io/MshWriter.hpp>
#include <polyfem/io/MatrixIO.hpp>
#include <polyfem/io/Evaluator.hpp>

#include <polyfem/mesh/Obstacle.hpp>

#include <polyfem/problem/KernelProblem.hpp>
#include <polyfem/problem/ProblemFactory.hpp>

#include <polyfem/refinement/APriori.hpp>

#include <polyfem/solver/forms/ContactForm.hpp>

#include <polyfem/time_integrator/ImplicitTimeIntegrator.hpp>
#include <polyfem/utils/BoundarySampler.hpp>
#include <polyfem/utils/Jacobian.hpp>
#include <polyfem/utils/MatrixUtils.hpp>

#include <algorithm>
#include <map>
#include <ostream>

#include <igl/Timer.h>

#include <spdlog/fmt/fmt.h>

namespace polyfem::varform
{
	void ElasticVarForm::serialize_checkpoint(
		io::CheckpointWriter &writer,
		const Eigen::MatrixXd &solution,
		const io::CheckpointMetadata &metadata) const
	{
		VarForm::serialize_checkpoint(writer, solution, metadata);
		write_checkpoint_ordering(writer, "primary", space_);
	}

	void ElasticVarForm::deserialize_checkpoint(
		const io::CheckpointReader &reader,
		Eigen::MatrixXd &solution)
	{
		VarForm::deserialize_checkpoint(reader, solution);
		validate_checkpoint_solution(solution, space_.ndof());
		reorder_checkpoint_block(
			reader, "primary", space_,
			mesh_->dimension(), 0, solution);
	}

	void ElasticVarForm::reset()
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
	}

	void ElasticVarForm::init(const std::string &formulation, const Units &units, const json &args, const std::string &out_path)
	{
		VarForm::init(formulation, units, args, out_path);
		const bool is_time_dependent = args.contains("time") && !args["time"].is_null();

		primary_assembler_ = assembler::AssemblerUtils::make_assembler(formulation);
		assert(primary_assembler_->name() == formulation);
		if (args["solver"]["advanced"]["check_inversion"] == "Conservative")
		{
			if (auto elastic_assembler = std::dynamic_pointer_cast<assembler::ElasticityAssembler>(primary_assembler_))
				elastic_assembler->set_use_robust_jacobian();
		}
		mass_assembler_ = std::make_shared<assembler::Mass>();
		pure_mass_assembler_ = std::make_shared<assembler::HRZMass>();

		if (!args.contains("preset_problem"))
		{
			problem = std::make_shared<assembler::GenericTensorProblem>("GenericTensor");

			problem->clear();
			json tmp;
			tmp["is_time_dependent"] = is_time_dependent;
			problem->set_parameters(tmp, resources_);

			// important for the BC

			auto bc = args["boundary_conditions"];
			problem->set_parameters(bc, resources_);
			problem->set_parameters(args["initial_conditions"], resources_);
			problem->set_parameters(args["output"], resources_);
		}
		else
		{
			if (args["preset_problem"]["type"] == "Kernel")
			{
				problem = std::make_shared<problem::KernelProblem>("Kernel", *primary_assembler_);
				problem->clear();
				problem::KernelProblem &kprob = *dynamic_cast<problem::KernelProblem *>(problem.get());
			}
			else
			{
				problem = problem::ProblemFactory::factory().get_problem(args["preset_problem"]["type"]);
				problem->clear();
			}
			// important for the BC
			problem->set_parameters(args["preset_problem"], resources_);
		}

		problem->set_units(*primary_assembler_, units);

		t0 = is_time_dependent ? args["time"]["t0"].get<double>() : 0.0;
		time_steps = is_time_dependent ? args["time"]["time_steps"].get<int>() : 0;
		dt = is_time_dependent ? args["time"]["dt"].get<double>() : 0.0;
	}

	void ElasticVarForm::load_mesh(const mesh::Mesh &mesh, const json &args)
	{
		set_materials(*primary_assembler_, mesh.dimension());
		set_materials(*mass_assembler_, mesh.dimension());
		pure_mass_assembler_->set_size(mass_assembler_->size());

		problem->init(mesh);

		if (assembler::MultiModel *mm = dynamic_cast<assembler::MultiModel *>(primary_assembler_.get()))
		{
			assert(args["materials"].is_array());

			std::vector<std::string> materials(mesh.n_elements());

			std::map<int, std::string> mats;

			for (const auto &m : args["materials"])
				mats[m["id"].get<int>()] = m["type"];

			for (int i = 0; i < materials.size(); ++i)
				materials[i] = mats.at(mesh.get_body_id(i));

			mm->init_multimodels(materials);
		}
	}

	void ElasticVarForm::build_basis(mesh::Mesh &mesh, const bool iso_parametric, const json &args)
	{
		build_elastic_basis(mesh, iso_parametric, args, -1);
	}

	void ElasticVarForm::build_elastic_basis(mesh::Mesh &mesh, const bool iso_parametric, const json &args, const int fe_space_id)
	{
		assert(problem);
		assert(primary_assembler_);
		assert(mass_assembler_);
		assert(pure_mass_assembler_);

		Eigen::VectorXi space_disc_orders, space_disc_ordersq;
		assign_discr_orders(args["space"], fe_space_id, mesh, space_disc_orders, space_disc_ordersq);

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
			space_disc_ordersq,
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

		boundary_.clear_boundary_conditions();
		problem->setup_bc(
			mesh, assembler::BoundaryKind::Dirichlet, fe_space_id,
			space_.basis_list(), boundary_.total_local_boundary,
			boundary_.local_boundary, boundary_.boundary_nodes, mesh.dimension());
		problem->setup_bc(
			mesh, assembler::BoundaryKind::Neumann, fe_space_id,
			boundary_.total_local_boundary, boundary_.local_neumann_boundary);

		problem->setup_nodal_bc(
			space_.mesh_nodes.get(), assembler::BoundaryKind::Dirichlet, fe_space_id, space_.n_bases, boundary_.dirichlet_nodes);
		problem->setup_nodal_bc(
			space_.mesh_nodes.get(), assembler::BoundaryKind::Neumann, fe_space_id, space_.n_bases, boundary_.neumann_nodes);

		for (const int n_id : boundary_.dirichlet_nodes)
		{
			const int tag = space_.node_tag(n_id);
			for (int d = 0; d < mesh.dimension(); ++d)
				if (problem->is_nodal_dimension_dirichlet(n_id, tag, d, fe_space_id))
					boundary_.boundary_nodes.push_back(n_id * mesh.dimension() + d);
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

	void ElasticVarForm::build_rhs_assembler()
	{
		json rhs_solver_params = args["solver"]["linear"];
		if (!rhs_solver_params.contains("Pardiso"))
			rhs_solver_params["Pardiso"] = {};
		rhs_solver_params["Pardiso"]["mtype"] = -2;

		rhs_assembler_ = std::make_shared<assembler::RhsAssembler>(
			*primary_assembler_, *mesh_, space_.mesh_nodes.get(), nullptr,
			boundary_.dirichlet_nodes, boundary_.neumann_nodes,
			boundary_.dirichlet_nodes_position, boundary_.neumann_nodes_position,
			space_.n_bases, mesh_->dimension(), space_.basis_list(), space_.geometry_basis_list(), mass_ass_vals_cache_, *problem, resources_,
			args["space"]["advanced"]["bc_method"],
			rhs_solver_params,
			/*fe_space_id=*/-1);
	}

	void ElasticVarForm::assemble_rhs(const mesh::Mesh &mesh)
	{
		igl::Timer timer;
		json p_params = {};
		p_params["formulation"] = primary_assembler_->name();
		{
			RowVectorNd min, max, delta;
			mesh.bounding_box(min, max);
			delta = (max - min) / 2. + min;
			if (mesh.is_volume())
				p_params["bbox_center"] = {delta(0), delta(1), delta(2)};
			else
				p_params["bbox_center"] = {delta(0), delta(1)};
		}
		problem->set_parameters(p_params, resources_);

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

	void ElasticVarForm::assemble_mass_mat(const mesh::Mesh &mesh, const json &args)
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

		mass_assembler_->assemble(mesh.is_volume(), space_.n_bases, space_.basis_list(), space_.geometry_basis_list(), mass_ass_vals_cache_, 0, mass_, true);
		if (!primary_assembler_->is_linear())
			pure_mass_assembler_->assemble(mesh.is_volume(), space_.n_bases, space_.basis_list(), space_.geometry_basis_list(), pure_mass_ass_vals_cache_, 0, pure_mass_, true);

		assert(mass_.size() > 0);

		avg_mass_ = 0;
		for (int k = 0; k < mass_.outerSize(); ++k)
			for (StiffnessMatrix::InnerIterator it(mass_, k); it; ++it)
			{
				assert(it.col() == k);
				avg_mass_ += it.value();
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

	void ElasticVarForm::initial_velocity(
		Eigen::MatrixXd &velocity,
		const InitialConditionOverride *override,
		const std::string &state_prefix) const
	{
		assert(rhs_assembler_ != nullptr);
		if (override && override->velocity.size() != 0)
		{
			velocity = override->velocity;
			assert(
				velocity.rows() == space_.ndof() && velocity.cols() >= 1
				&& "Initial velocity override must match the simulation DOFs");
			return;
		}

		rhs_assembler_->initial_velocity(velocity);
	}

	void ElasticVarForm::initial_acceleration(
		Eigen::MatrixXd &acceleration,
		const InitialConditionOverride *override,
		const std::string &state_prefix) const
	{
		assert(rhs_assembler_ != nullptr);
		if (override && override->acceleration.size() != 0)
		{
			acceleration = override->acceleration;
			assert(
				acceleration.rows() == space_.ndof() && acceleration.cols() >= 1
				&& "Initial acceleration override must match the simulation DOFs");
			return;
		}

		rhs_assembler_->initial_acceleration(acceleration);
	}

	void ElasticVarForm::initial_solution(
		Eigen::MatrixXd &solution,
		const InitialConditionOverride *override,
		const std::string &state_prefix) const
	{
		assert(rhs_assembler_ != nullptr);
		if (override && override->solution.size() != 0)
		{
			solution = override->solution;
			assert(
				solution.rows() == space_.ndof() && solution.cols() >= 1
				&& "Initial solution override must match the simulation DOFs");
			return;
		}

		if (problem->is_time_dependent())
			rhs_assembler_->initial_solution(solution);
		else
		{
			solution.resize(rhs_.size(), 1);
			solution.setZero();
		}
	}

	QuadratureOrders ElasticVarForm::elastic_boundary_samples() const
	{
		const int gdiscr_order = mesh_->orders().size() <= 0 ? 1 : mesh_->orders().maxCoeff();
		return n_boundary_samples(space_.disc_orders.maxCoeff(), space_.disc_ordersq.maxCoeff(), gdiscr_order);
	}

	std::vector<int> ElasticVarForm::elastic_primitive_to_node() const
	{
		if (!mesh_ || !space_.geometry)
			return {};

		const auto &nodes = space_.is_iso_parametric() ? space_.mesh_nodes : space_.geometry->mesh_nodes;
		if (!nodes)
			return {};

		auto indices = nodes->primitive_to_node();
		indices.resize(mesh_->n_vertices());
		return indices;
	}

	std::vector<int> ElasticVarForm::elastic_node_to_primitive() const
	{
		const auto p2n = elastic_primitive_to_node();
		const int n_geometry_bases = space_.geometry ? space_.geometry->n_bases : 0;
		std::vector<int> indices(n_geometry_bases, -1);
		for (int i = 0; i < int(p2n.size()); ++i)
		{
			if (p2n[i] >= 0 && p2n[i] < int(indices.size()))
				indices[p2n[i]] = i;
		}
		return indices;
	}

	io::OutputFieldFunction ElasticVarForm::snapshot_elastic_output_fields(
		const Eigen::MatrixXd &solution, const io::OutGeometryData::ExportOptions &opts,
		const mesh::Obstacle *obstacle,
		const time_integrator::ImplicitTimeIntegrator *time_integrator,
		const std::vector<std::pair<std::string, std::shared_ptr<solver::Form>>> &named_forms,
		const solver::Form *elastic_form, const solver::ContactForm *contact_form) const
	{
		auto state = std::make_shared<ElasticOutputState>();
		const io::OutputFieldOptions options{opts.fields};
		const auto &flags = args["output"]["paraview"]["options"];
		if (problem->is_time_dependent())
		{
			if (flags["velocity"] && options.export_field("velocity"))
				state->velocity = time_integrator ? time_integrator->v_prev() : Eigen::VectorXd::Zero(solution.size());
			if (flags["acceleration"] && options.export_field("acceleration"))
				state->acceleration = time_integrator ? time_integrator->a_prev() : Eigen::VectorXd::Zero(solution.size());
		}
		if (flags["forces"] && !problem->is_scalar())
		{
			const double scaling = time_integrator ? time_integrator->acceleration_scaling() : 1;
			for (const auto &[name, form] : named_forms)
			{
				const std::string field_name = name + "_forces";
				if (!options.export_field(field_name))
					continue;
				Eigen::VectorXd force;
				if (form && form->enabled())
				{
					form->first_derivative(solution, force);
					force *= -1.0 / scaling;
				}
				else
					force.setZero(solution.size());
				state->forces.push_back({field_name, std::move(force)});
			}
		}
		if (!opts.fields.empty() && options.export_field("gradient_of_elastic_potential") && elastic_form)
		{
			Eigen::VectorXd gradient;
			elastic_form->first_derivative(solution, gradient);
			state->elastic_gradient = std::move(gradient);
		}
		if (!opts.fields.empty() && options.export_field("gradient_of_contact_potential") && contact_form && contact_form->weight() > 0)
		{
			Eigen::VectorXd gradient;
			contact_form->first_derivative(solution, gradient);
			gradient *= -contact_form->barrier_stiffness() / contact_form->weight();
			state->contact_gradient = std::move(gradient);
		}
		auto saved_solution = std::make_shared<Eigen::MatrixXd>(solution);
		return [this, state, saved_solution, obstacle, fields = opts.fields](const io::OutputSample &sample) {
			return elastic_output_fields(sample, *saved_solution,
										 {sample.requested_fields.empty() ? fields : sample.requested_fields},
										 obstacle, nullptr, {}, nullptr, nullptr, state.get());
		};
	}

	std::vector<io::OutputField> ElasticVarForm::elastic_output_fields(
		const io::OutputSample &sample,
		const Eigen::MatrixXd &solution,
		const io::OutputFieldOptions &options,
		const mesh::Obstacle *obstacle,
		const time_integrator::ImplicitTimeIntegrator *time_integrator,
		const std::vector<std::pair<std::string, std::shared_ptr<solver::Form>>> &named_forms,
		const solver::Form *elastic_form,
		const solver::ContactForm *contact_form, const ElasticOutputState *output_state) const
	{
		std::vector<io::OutputField> fields;
		if (!mesh_ || !problem || solution.size() <= 0)
			return fields;

		const bool has_element_samples = sample.local_points.rows() > 0 && sample.local_points.rows() == sample.element_ids.size();
		const int output_rows = sample.points.rows() > 0 ? sample.points.rows() : std::max<int>(sample.local_points.rows(), sample.node_ids.size());

		const int actual_dim = problem->is_scalar() ? 1 : mesh_->dimension();
		const auto &paraview_options = args["output"]["paraview"]["options"];
		const bool material_params = paraview_options["material"];
		const bool body_ids = paraview_options["body_ids"];
		const bool velocity = paraview_options["velocity"];
		const bool acceleration = paraview_options["acceleration"];
		const bool forces = paraview_options["forces"] && !problem->is_scalar();
		const bool tensor_values = paraview_options["tensor_values"] && !problem->is_scalar();
		const bool scalar_values = paraview_options["scalar_values"];
		const bool use_spline = args["space"]["basis_type"] == "Spline";
		const bool explicit_fields = !options.fields.empty();
		StiffnessMatrix interpolation;
		const auto get_interpolation = [&]() -> const StiffnessMatrix & {
			if (interpolation.rows() != sample.local_points.rows())
				interpolation = io::Evaluator::build_interpolation_matrix(
					space_.n_bases, space_.basis_list(), sample.local_points, sample.element_ids);
			return interpolation;
		};

		const auto resize_to_output_rows = [&](Eigen::MatrixXd &values) {
			if (output_rows <= values.rows())
				return;

			const int previous_rows = values.rows();
			values.conservativeResize(output_rows, values.cols());
			values.bottomRows(output_rows - previous_rows).setZero();
		};

		const auto append_obstacle_values = [&](Eigen::MatrixXd &sampled_values, const Eigen::MatrixXd &dof_values) -> bool {
			if (!obstacle || obstacle->n_vertices() <= 0)
			{
				resize_to_output_rows(sampled_values);
				return sample.points.rows() == 0 || sample.points.rows() == sampled_values.rows();
			}

			const bool has_obstacle_rows =
				sample.points.rows() == sampled_values.rows() + obstacle->n_vertices()
				&& sample.points.cols() == obstacle->v().cols()
				&& sample.points.bottomRows(obstacle->n_vertices()).isApprox(obstacle->v());

			if (!has_obstacle_rows)
				return sample.points.rows() == 0 || sample.points.rows() == sampled_values.rows();

			sampled_values.conservativeResize(sampled_values.rows() + obstacle->n_vertices(), sampled_values.cols());
			if (dof_values.rows() >= obstacle->ndof())
			{
				const Eigen::MatrixXd obstacle_values =
					utils::unflatten(dof_values.bottomRows(obstacle->ndof()), sampled_values.cols());
				if (obstacle_values.rows() == obstacle->n_vertices()
					&& obstacle_values.cols() == sampled_values.cols())
					sampled_values.bottomRows(obstacle->n_vertices()) = obstacle_values;
				else
					sampled_values.bottomRows(obstacle->n_vertices()).setZero();
			}
			else
				sampled_values.bottomRows(obstacle->n_vertices()).setZero();
			return true;
		};

		const auto sample_dof_field = [&](const Eigen::MatrixXd &dof_values, const int field_dim, Eigen::MatrixXd &values) -> bool {
			if (dof_values.size() <= 0 || field_dim <= 0)
				return false;

			if (has_element_samples)
			{
				values = get_interpolation() * utils::unflatten(dof_values.topRows(space_.n_bases * field_dim), field_dim);

				return append_obstacle_values(values, dof_values);
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

		const auto append_sampled_dof_field = [&](const std::string &name, const Eigen::MatrixXd &dof_values, const int field_dim) {
			Eigen::MatrixXd values;
			if (sample_dof_field(dof_values, field_dim, values))
				fields.push_back({name, values, io::OutputField::Association::Point});
		};

		std::vector<std::pair<std::string, int>> raw_scalar_fields, raw_tensor_fields;
		bool all_scalar_fields = false, all_tensor_fields = false;
		const bool sample_scalar = scalar_values && !problem->is_scalar() && has_element_samples
								   && (options.fields.empty() || options.export_field("von_mises"));
		const bool sample_tensor = tensor_values && !problem->is_scalar() && has_element_samples
								   && (options.fields.empty() || options.export_field("cauchy_stess") || options.export_field("pk1_stess")
									   || options.export_field("pk2_stess") || options.export_field("F")
									   || options.export_field("energy") || options.export_field("energy_avg"));
		bool stress_sampled = false;
		std::vector<assembler::Assembler::NamedMatrix> sampled_scalar, sampled_tensor;
		const auto sample_stress_values = [&](const bool tensor) {
			if (!stress_sampled)
			{
				const auto scatter_values = [&](const auto &local, auto &points, int begin, int count) {
					if (points.empty())
					{
						points.resize(local.size());
						for (size_t k = 0; k < local.size(); ++k)
						{
							points[k].first = local[k].first;
							points[k].second.setZero(output_rows, local[k].second.cols());
						}
					}
					for (size_t k = 0; k < local.size(); ++k)
						points[k].second.middleRows(begin, count) = local[k].second;
				};
				std::vector<assembler::Assembler::NamedMatrix> local_scalar, local_tensor;
				for (int begin = 0; begin < sample.local_points.rows();)
				{
					const int element_id = sample.element_ids(begin);
					int end = begin + 1;
					while (end < sample.local_points.rows() && sample.element_ids(end) == element_id)
						++end;
					if (element_id >= 0)
					{
						const Eigen::MatrixXd local_points = sample.local_points.middleRows(begin, end - begin);
						const assembler::OutputData data(sample.time, element_id,
														 space_.basis_list()[element_id], space_.geometry_basis_list()[element_id], local_points, solution);
						primary_assembler_->compute_output_values(data,
																  sample_scalar ? &local_scalar : nullptr, sample_tensor ? &local_tensor : nullptr);
						scatter_values(local_scalar, sampled_scalar, begin, end - begin);
						scatter_values(local_tensor, sampled_tensor, begin, end - begin);
					}
					begin = end;
				}
				stress_sampled = true;
			}
			return tensor ? std::move(sampled_tensor) : std::move(sampled_scalar);
		};

		const auto append_scalar_values = [&]() {
			if (!scalar_values || problem->is_scalar() || !has_element_samples)
				return;

			const bool wants_scalar = options.fields.empty() || options.export_field("von_mises");
			if (!wants_scalar)
				return;

			auto point_values = sample_stress_values(false);

			all_scalar_fields = !point_values.empty();
			for (auto &[name, values] : point_values)
			{
				if (options.export_field(name))
				{
					raw_scalar_fields.emplace_back(name, fields.size());
					fields.push_back({name, std::move(values), io::OutputField::Association::Point});
				}
				else
					all_scalar_fields = false;
			}
		};

		const auto append_tensor_values = [&]() {
			if (!tensor_values || problem->is_scalar() || !has_element_samples)
				return;

			const bool wants_tensor = options.fields.empty()
									  || options.export_field("cauchy_stess")
									  || options.export_field("pk1_stess")
									  || options.export_field("pk2_stess")
									  || options.export_field("F")
									  || options.export_field("energy")
									  || options.export_field("energy_avg");
			if (!wants_tensor)
				return;

			auto point_values = sample_stress_values(true);

			all_tensor_fields = !point_values.empty();
			for (const auto &[name, values] : point_values)
			{
				if (name == "energy")
				{
					// Scalar field: must not go through the per-dimension splitting below.
					if (options.export_field("energy"))
						fields.push_back({name, values, io::OutputField::Association::Point});

					if (options.export_field("energy_avg"))
					{
						// Per-element mean of the energy at the sample points.
						Eigen::VectorXd sum = Eigen::VectorXd::Zero(mesh_->n_elements());
						Eigen::VectorXi cnt = Eigen::VectorXi::Zero(mesh_->n_elements());
						for (int i = 0; i < sample.element_ids.size(); ++i)
						{
							const int e = sample.element_ids(i);
							if (e < 0)
								continue;
							sum(e) += values(i, 0);
							cnt(e)++;
						}

						Eigen::MatrixXd avg = Eigen::MatrixXd::Zero(values.rows(), 1);
						for (int i = 0; i < sample.element_ids.size(); ++i)
						{
							const int e = sample.element_ids(i);
							if (e >= 0)
								avg(i) = sum(e) / cnt(e);
						}
						fields.push_back({"energy_avg", avg, io::OutputField::Association::Point});
					}
					continue;
				}

				if (!options.export_field(name))
				{
					all_tensor_fields = false;
					continue;
				}

				raw_tensor_fields.emplace_back(name, fields.size());
				const int stride = mesh_->dimension();
				assert(values.cols() % stride == 0);
				for (int i = 0; i < values.cols(); i += stride)
				{
					const int ii = (i / stride) + 1;
					fields.push_back({fmt::format("{:s}_{:d}", name, ii), values.middleCols(i, stride), io::OutputField::Association::Point});
				}
			}
		};

		const auto append_averaged_values = [&]() {
			if (use_spline || problem->is_scalar() || !has_element_samples || (!scalar_values && !tensor_values))
				return;

			const bool wants_avg = options.fields.empty()
								   || options.export_field("von_mises_avg")
								   || options.export_field("cauchy_stess_avg")
								   || options.export_field("pk1_stess_avg")
								   || options.export_field("pk2_stess_avg")
								   || options.export_field("F_avg");
			if (!wants_avg)
				return;

			Eigen::MatrixXd areas(space_.n_bases, 1);
			areas.setZero();
			std::vector<assembler::Assembler::NamedMatrix> tmp_s, tmp_t;
			std::vector<Eigen::MatrixXd> avg_scalar, avg_tensor;
			// Reuse raw output only when its samples coincide with the averaging nodes.
			std::vector<std::pair<int, int>> sample_ranges(space_.basis_list().size(), {-1, 0});
			for (int begin = 0; begin < sample.element_ids.size();)
			{
				const int element = sample.element_ids(begin);
				int end = begin + 1;
				while (end < sample.element_ids.size() && sample.element_ids(end) == element)
					++end;
				if (element >= 0)
				{
					auto &range = sample_ranges[element];
					range.first = range.second == 0 ? begin : -1;
					range.second += end - begin;
				}
				begin = end;
			}

			for (int e = 0; e < int(space_.basis_list().size()); ++e)
			{
				Eigen::MatrixXd local_pts;
				if (mesh_->is_simplex(e))
				{
					if (mesh_->dimension() == 3)
						autogen::p_nodes_3d(space_.disc_orders(e), local_pts);
					else
						autogen::p_nodes_2d(space_.disc_orders(e), local_pts);
				}
				else if (mesh_->is_cube(e))
				{
					if (mesh_->dimension() == 3)
						autogen::q_nodes_3d(space_.disc_orders(e), local_pts);
					else
						autogen::q_nodes_2d(space_.disc_orders(e), local_pts);
				}
				else if (mesh_->is_prism(e))
				{
					autogen::prism_nodes_3d(space_.disc_orders(e), space_.disc_ordersq(e), local_pts);
				}
				else
				{
					continue;
				}

				const basis::ElementBases &bs = space_.basis_list()[e];
				const basis::ElementBases &gbs = space_.geometry_basis_list()[e];

				assembler::ElementAssemblyValues vals;
				vals.compute(e, mesh_->is_volume(), bs, gbs);
				const double area = (vals.det.array() * vals.quadrature.weights.array()).sum();

				const auto [begin, count] = sample_ranges[e];
				const bool sampled_at_nodes = begin >= 0 && count == local_pts.rows()
											  && (sample.local_points.middleRows(begin, count).array() == local_pts.array()).all();
				const bool reuse_scalar = sampled_at_nodes && all_scalar_fields;
				const bool reuse_tensor = sampled_at_nodes && all_tensor_fields;
				if ((scalar_values && !reuse_scalar) || (tensor_values && !reuse_tensor))
				{
					primary_assembler_->compute_output_values(
						assembler::OutputData(sample.time, e, bs, gbs, local_pts, solution),
						scalar_values && !reuse_scalar ? &tmp_s : nullptr,
						tensor_values && !reuse_tensor ? &tmp_t : nullptr);
					// Energy uses the per-element mean assembled with the raw samples.
					tmp_t.erase(std::remove_if(tmp_t.begin(), tmp_t.end(), [](const auto &field) {
									return field.first == "energy";
								}),
								tmp_t.end());
				}
				if (scalar_values)
				{
					if (reuse_scalar)
					{
						tmp_s.resize(raw_scalar_fields.size());
						for (int k = 0; k < raw_scalar_fields.size(); ++k)
						{
							tmp_s[k].first = raw_scalar_fields[k].first;
							tmp_s[k].second = fields[raw_scalar_fields[k].second].values.middleRows(begin, count);
						}
					}
				}
				if (tensor_values)
				{
					if (reuse_tensor)
					{
						tmp_t.resize(raw_tensor_fields.size());
						for (int k = 0; k < raw_tensor_fields.size(); ++k)
						{
							tmp_t[k].first = raw_tensor_fields[k].first;
							tmp_t[k].second.resize(count, actual_dim * actual_dim);
							for (int d = 0; d < actual_dim; ++d)
								tmp_t[k].second.middleCols(d * actual_dim, actual_dim) =
									fields[raw_tensor_fields[k].second + d].values.middleRows(begin, count);
						}
					}
				}

				if (avg_scalar.empty() && !tmp_s.empty())
				{
					avg_scalar.resize(tmp_s.size());
					for (auto &m : avg_scalar)
						m.setZero(space_.n_bases, 1);
				}
				if (avg_tensor.empty() && !tmp_t.empty())
				{
					avg_tensor.resize(tmp_t.size());
					// Not every tensor is dim x dim (e.g. the scalar "energy"), so size by its own columns.
					for (int k = 0; k < tmp_t.size(); ++k)
						avg_tensor[k].setZero(space_.n_bases, tmp_t[k].second.cols());
				}

				for (size_t j = 0; j < bs.bases.size(); ++j)
				{
					const basis::Basis &b = bs.bases[j];
					if (b.global().size() > 1)
						continue;

					const int index = b.global().front().index;
					areas(index) += area;
					for (int k = 0; k < tmp_s.size(); ++k)
						avg_scalar[k](index) += tmp_s[k].second(j) * area;
					for (int k = 0; k < tmp_t.size(); ++k)
						avg_tensor[k].row(index) += tmp_t[k].second.row(j) * area;
				}
			}

			for (auto &m : avg_scalar)
				for (int i = 0; i < m.rows(); ++i)
					if (areas(i) > 0)
						m(i) /= areas(i);
			for (auto &m : avg_tensor)
				for (int i = 0; i < m.rows(); ++i)
					if (areas(i) > 0)
						m.row(i) /= areas(i);

			for (int k = 0; k < tmp_s.size(); ++k)
			{
				const std::string name = fmt::format("{:s}_avg", tmp_s[k].first);
				if (!options.export_field(name))
					continue;

				Eigen::MatrixXd sampled;
				if (sample_dof_field(avg_scalar[k], 1, sampled))
					fields.push_back({name, sampled, io::OutputField::Association::Point});
			}

			for (int k = 0; k < tmp_t.size(); ++k)
			{
				// "energy_avg" is the per-element mean written by append_tensor_values.
				if (tmp_t[k].first == "energy")
					continue;

				const std::string base_name = fmt::format("{:s}_avg", tmp_t[k].first);
				if (!options.export_field(base_name))
					continue;

				Eigen::MatrixXd sampled;
				if (!sample_dof_field(utils::flatten(avg_tensor[k]), actual_dim * actual_dim, sampled))
					continue;

				const int stride = mesh_->dimension();
				for (int i = 0; i < sampled.cols(); i += stride)
				{
					const int ii = (i / stride) + 1;
					fields.push_back({fmt::format("{:s}_{:d}", base_name, ii), sampled.middleCols(i, stride), io::OutputField::Association::Point});
				}
			}
		};

		const auto append_material_fields = [&]() {
			if (!material_params || !has_element_samples)
				return;

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
		};

		const auto append_body_ids = [&]() {
			if (!body_ids || !options.export_field("body_ids") || !has_element_samples)
				return;

			Eigen::MatrixXd ids = Eigen::MatrixXd::Zero(output_rows, 1);
			for (int i = 0; i < sample.element_ids.size(); ++i)
			{
				const int element_id = sample.element_ids(i);
				if (element_id >= 0)
					ids(i) = mesh_->get_body_id(element_id);
			}
			fields.push_back({"body_ids", ids, io::OutputField::Association::Point});
		};

		const auto compute_traction_forces = [&]() {
			Eigen::MatrixXd traction_forces;
			traction_forces.setZero(space_.n_bases * actual_dim, 1);

			Eigen::MatrixXd uv, points, normals;
			Eigen::VectorXd weights;
			Eigen::VectorXi global_primitive_ids;
			assembler::ElementAssemblyValues vals;

			for (const auto &lb : boundary_.total_local_boundary)
			{
				const int e = lb.element_id();
				const bool has_samples = utils::BoundarySampler::boundary_quadrature(
					lb, elastic_boundary_samples(), *mesh_, false, uv, points, normals, weights, global_primitive_ids);
				if (!has_samples)
					continue;

				const basis::ElementBases &bs = space_.basis_list()[e];
				const basis::ElementBases &gbs = space_.geometry_basis_list()[e];
				vals.compute(e, mesh_->is_volume(), points, bs, gbs);

				for (int n = 0; n < normals.rows(); ++n)
				{
					Eigen::MatrixXd deform_mat = Eigen::MatrixXd::Zero(actual_dim, actual_dim);
					for (const auto &b : vals.basis_values)
					{
						for (const auto &g : b.global)
						{
							for (int d = 0; d < actual_dim; ++d)
								deform_mat.row(d) += solution(g.index * actual_dim + d) * b.grad.row(n);
						}
					}

					Eigen::MatrixXd trafo = vals.jac_it[n].inverse() + deform_mat;
					normals.row(n) = normals.row(n) * trafo.inverse();
					normals.row(n).normalize();
				}

				std::vector<assembler::Assembler::NamedMatrix> tensor_flat;
				primary_assembler_->compute_tensor_value(assembler::OutputData(sample.time, e, bs, gbs, points, solution), tensor_flat);

				for (long n = 0; n < vals.basis_values.size(); ++n)
				{
					const assembler::AssemblyValues &v = vals.basis_values[n];
					const int g_index = v.global[0].index * actual_dim;
					for (int q = 0; q < points.rows(); ++q)
					{
						assert(tensor_flat[0].first == "cauchy_stess");
						Eigen::MatrixXd stress_tensor = utils::unflatten(tensor_flat[0].second.row(q), actual_dim);
						traction_forces.block(g_index, 0, actual_dim, 1) += stress_tensor * normals.row(q).transpose() * v.val(q) * weights(q);
					}
				}
			}

			return traction_forces;
		};

		const auto append_traction_force = [&]() {
			if (problem->is_scalar() || !explicit_fields || !options.export_field("traction_force"))
				return;

			if (has_element_samples && sample.normals.rows() == sample.local_points.rows() && sample.primitive_ids.size() == sample.local_points.rows())
			{
				const Eigen::MatrixXd displaced_normals = displaced_output_normals(sample, solution);
				const Eigen::MatrixXd &normals = displaced_normals.rows() == sample.normals.rows() ? displaced_normals : sample.normals;
				Eigen::MatrixXd values = Eigen::MatrixXd::Zero(output_rows, actual_dim);
				for (int i = 0; i < sample.local_points.rows(); ++i)
				{
					const int element_id = sample.element_ids(i);
					if (element_id < 0)
						continue;

					std::vector<assembler::Assembler::NamedMatrix> tensor_flat;
					primary_assembler_->compute_tensor_value(
						assembler::OutputData(sample.time, element_id, space_.basis_list()[element_id], space_.geometry_basis_list()[element_id], sample.local_points.row(i), solution),
						tensor_flat);

					assert(tensor_flat[0].first == "cauchy_stess");
					Eigen::Map<Eigen::MatrixXd> tensor(tensor_flat[0].second.data(), actual_dim, actual_dim);
					values.row(i) = normals.row(i) * tensor;

					double area = 0;
					const int primitive_id = sample.primitive_ids(i);
					if (mesh_->is_volume())
					{
						if (mesh_->is_simplex(element_id))
							area = mesh_->tri_area(primitive_id);
						else if (mesh_->is_cube(element_id))
							area = mesh_->quad_area(primitive_id);
						else if (mesh_->is_prism(element_id))
							area = mesh_->n_face_vertices(primitive_id) == 4 ? mesh_->quad_area(primitive_id) : mesh_->tri_area(primitive_id);
					}
					else
					{
						area = mesh_->edge_length(primitive_id);
					}
					values.row(i) *= area;
				}
				fields.push_back({"traction_force", values, io::OutputField::Association::Point});
				return;
			}

			append_sampled_dof_field("traction_force", compute_traction_forces(), actual_dim);
		};

		append_scalar_values();
		append_tensor_values();
		append_averaged_values();
		append_material_fields();
		append_body_ids();

		if ((paraview_options["jacobian_validity"] || (explicit_fields && options.export_field("validity")))
			&& has_element_samples
			&& sample.primitive_ids.size() == 0)
		{
			const auto invalid_elements = utils::count_invalid(
				mesh_->dimension(), space_.basis_list(), space_.geometry_basis_list(), solution);
			Eigen::MatrixXd validity = Eigen::MatrixXd::Zero(output_rows, 1);
			for (int i = 0; i < sample.element_ids.size(); ++i)
			{
				validity(i) = std::find(
								  invalid_elements.begin(), invalid_elements.end(), sample.element_ids(i))
							  != invalid_elements.end();
			}
			fields.push_back({"validity", validity, io::OutputField::Association::Point});
		}

		if (problem->is_time_dependent())
		{
			if (velocity && options.export_field("velocity"))
				append_sampled_dof_field(
					"velocity",
					output_state ? output_state->velocity : time_integrator ? time_integrator->v_prev()
																			: Eigen::VectorXd::Zero(solution.size()),
					actual_dim);
			if (acceleration && options.export_field("acceleration"))
				append_sampled_dof_field(
					"acceleration",
					output_state ? output_state->acceleration : time_integrator ? time_integrator->a_prev()
																				: Eigen::VectorXd::Zero(solution.size()),
					actual_dim);
		}

		if (forces && !output_state)
		{
			const double s = time_integrator ? time_integrator->acceleration_scaling() : 1;
			for (const auto &[name, form] : named_forms)
			{
				const std::string field_name = name + "_forces";
				if (!options.export_field(field_name))
					continue;

				Eigen::VectorXd force;
				if (form && form->enabled())
				{
					form->first_derivative(solution, force);
					force *= -1.0 / s;
				}
				else
				{
					force.setZero(solution.size());
				}
				append_sampled_dof_field(field_name, force, actual_dim);
			}
		}

		if (output_state)
			for (const auto &field : output_state->forces)
				if (options.export_field(field.name))
					append_sampled_dof_field(field.name, field.values, actual_dim);

		append_traction_force();
		if (output_state)
		{
			if (output_state->elastic_gradient.size() > 0 && options.export_field("gradient_of_elastic_potential"))
				append_sampled_dof_field("gradient_of_elastic_potential", output_state->elastic_gradient, actual_dim);
			if (output_state->contact_gradient.size() > 0 && options.export_field("gradient_of_contact_potential"))
				append_sampled_dof_field("gradient_of_contact_potential", output_state->contact_gradient, actual_dim);
		}

		if (!output_state && explicit_fields && options.export_field("gradient_of_elastic_potential") && elastic_form)
		{
			Eigen::VectorXd potential_grad;
			elastic_form->first_derivative(solution, potential_grad);
			append_sampled_dof_field("gradient_of_elastic_potential", potential_grad, actual_dim);
		}

		if (!output_state && explicit_fields && options.export_field("gradient_of_contact_potential") && contact_form && contact_form->weight() > 0)
		{
			Eigen::VectorXd potential_grad;
			contact_form->first_derivative(solution, potential_grad);
			potential_grad *= -contact_form->barrier_stiffness() / contact_form->weight();
			append_sampled_dof_field("gradient_of_contact_potential", potential_grad, actual_dim);
		}

		append_primary_output_fields(fields, sample, solution, options, obstacle,
									 has_element_samples && !(explicit_fields && options.export_field("solution_gradient"))
										 ? &get_interpolation()
										 : nullptr);
		return fields;
	}

	void ElasticVarForm::append_primary_output_fields(
		std::vector<io::OutputField> &fields,
		const io::OutputSample &sample,
		const Eigen::MatrixXd &solution,
		const io::OutputFieldOptions &options,
		const mesh::Obstacle *obstacle,
		const StiffnessMatrix *interpolation) const
	{
		if (!mesh_ || solution.size() <= 0)
			return;

		const int dim = mesh_->dimension();
		const bool has_element_samples =
			sample.local_points.rows() > 0
			&& sample.local_points.rows() == sample.element_ids.size();
		const bool export_solution_gradient =
			!options.fields.empty() && options.export_field("solution_gradient");

		Eigen::MatrixXd values, gradients;
		if (has_element_samples)
		{
			if (interpolation)
				values = *interpolation * utils::unflatten(solution.topRows(space_.n_bases * dim), dim);
			else
			{
				values.resize(sample.local_points.rows(), dim);
				if (export_solution_gradient)
					gradients.resize(sample.local_points.rows(), dim * mesh_->dimension());
				for (int i = 0; i < sample.local_points.rows();)
				{
					const int element_id = sample.element_ids(i);
					int end = i + 1;
					while (end < sample.local_points.rows() && sample.element_ids(end) == element_id)
						++end;
					const int count = end - i;
					if (element_id < 0)
					{
						values.middleRows(i, count).setZero();
						if (gradients.rows() > 0)
							gradients.middleRows(i, count).setZero();
						i = end;
						continue;
					}

					Eigen::MatrixXd local_value, local_gradient;
					io::Evaluator::interpolate_at_local_vals(
						*mesh_, dim, space_.basis_list(), space_.geometry_basis_list(),
						element_id, sample.local_points.middleRows(i, count), solution,
						local_value, local_gradient);
					values.middleRows(i, count) = local_value;
					if (gradients.rows() > 0)
						gradients.middleRows(i, count) = local_gradient;
					i = end;
				}
			}

			if (obstacle && obstacle->n_vertices() > 0
				&& sample.points.rows() == values.rows() + obstacle->n_vertices()
				&& sample.points.cols() == obstacle->v().cols()
				&& sample.points.bottomRows(obstacle->n_vertices()).isApprox(obstacle->v()))
			{
				values.conservativeResize(values.rows() + obstacle->n_vertices(), Eigen::NoChange);
				if (solution.rows() >= obstacle->ndof())
					values.bottomRows(obstacle->n_vertices()) =
						utils::unflatten(solution.bottomRows(obstacle->ndof()), dim);
				else
					values.bottomRows(obstacle->n_vertices()).setZero();
				if (gradients.rows() > 0)
				{
					gradients.conservativeResize(values.rows(), Eigen::NoChange);
					gradients.bottomRows(obstacle->n_vertices()).setZero();
				}
			}
		}
		else if (sample.node_ids.size() > 0)
		{
			values.resize(sample.node_ids.size(), dim);
			for (int i = 0; i < sample.node_ids.size(); ++i)
			{
				for (int d = 0; d < dim; ++d)
				{
					const int dof = sample.node_ids(i) * dim + d;
					if (dof < 0 || dof >= solution.rows())
						return;
					values(i, d) = solution(dof);
				}
			}
		}
		else
		{
			return;
		}

		if (sample.points.rows() > 0 && values.rows() != sample.points.rows())
			return;
		if (options.export_field("displacement"))
			fields.push_back({"displacement", values, io::OutputField::Association::Point});
		if (options.export_field("solution"))
			fields.push_back({"solution", values, io::OutputField::Association::Point});
		if (export_solution_gradient && gradients.rows() == values.rows())
			fields.push_back({"solution_gradient", gradients, io::OutputField::Association::Point});

		if (options.export_field("displaced_normals"))
		{
			Eigen::MatrixXd normals = displaced_output_normals(sample, solution);
			if (normals.rows() == values.rows())
				fields.push_back({"displaced_normals", normals, io::OutputField::Association::Point});
		}
	}

	Eigen::MatrixXd ElasticVarForm::displaced_output_normals(
		const io::OutputSample &sample,
		const Eigen::MatrixXd &solution) const
	{
		if (!mesh_
			|| sample.normals.rows() == 0
			|| sample.normals.rows() != sample.local_points.rows()
			|| sample.local_points.rows() != sample.element_ids.size())
			return {};

		const int dim = mesh_->dimension();
		Eigen::MatrixXd displaced_normals = sample.normals;
		for (int i = 0; i < sample.local_points.rows(); ++i)
		{
			const int element_id = sample.element_ids(i);
			if (element_id < 0)
				continue;

			Eigen::MatrixXd local_value, local_gradient;
			io::Evaluator::interpolate_at_local_vals(
				*mesh_, dim, space_.basis_list(), space_.geometry_basis_list(),
				element_id, sample.local_points.row(i), solution,
				local_value, local_gradient);

			Eigen::MatrixXd deformation = Eigen::MatrixXd::Identity(dim, dim);
			for (int d = 0; d < dim; ++d)
				deformation.row(d) += local_gradient.block(0, d * dim, 1, dim);
			displaced_normals.row(i) = sample.normals.row(i) * deformation.inverse();
			displaced_normals.row(i).normalize();
		}
		return displaced_normals;
	}

	io::OutputSpace ElasticVarForm::output_space() const
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
			&boundary_.dirichlet_nodes_position,
			space_.mesh_nodes.get()};
	}

	io::OutStatsData ElasticVarForm::compute_errors(const Eigen::MatrixXd &solution)
	{
		if (!args["output"]["advanced"]["compute_error"])
			return stats;

		double tend = 0;
		if (!args["time"].is_null())
			tend = args["time"]["tend"];

		stats.compute_errors(space_.n_bases, space_.basis_list(), space_.geometry_basis_list(), *mesh_, *problem, tend, solution);
		return stats;
	}

	void ElasticVarForm::export_data(const Eigen::MatrixXd &solution) const
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
			const int dim = mesh_->dimension();
			const int primary_ndof = std::min<int>(solution.rows(), space_.n_bases * dim);
			const Eigen::MatrixXd primary_solution = solution.topRows(primary_ndof);
			if (opts.reorder_output && space_.space_in_node_to_node.size() > 0)
			{
				const Eigen::MatrixXd nodal_solution = utils::unflatten(primary_solution, dim);
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

	void ElasticVarForm::save_json(const Eigen::MatrixXd &solution, std::ostream &out) const
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
		const int primary_size = space_.n_bases * mesh_->dimension();
		const Eigen::MatrixXd stats_solution =
			solution.rows() >= primary_size
				? solution.topRows(primary_size).eval()
				: solution;

		nlohmann::json j;
		stats.save_json(
			args, space_.n_bases, 0,
			stats_solution, *mesh_, space_.disc_orders, space_.disc_ordersq, *problem,
			timings, primary_assembler_ ? primary_assembler_->name() : name(), space_.is_iso_parametric(),
			args["output"]["advanced"]["sol_at_node"], j);
		out << j.dump(4) << std::endl;
	}

	void ElasticVarForm::save_elastic_step_state(
		const double t0,
		const double dt,
		const int t,
		const Eigen::MatrixXd &solution,
		const time_integrator::ImplicitTimeIntegrator *time_integrator) const
	{
		if (!mesh_)
			return;
		save_step_state(t0, dt, t, solution, time_integrator);
	}

	void ElasticVarForm::build_mesh_matrices(Eigen::MatrixXd &V, Eigen::MatrixXi &F) const
	{
		assert(mesh_);
		assert(space_.basis_list().size() == mesh_->n_elements());
		const size_t n_vertices = space_.n_bases - n_obstacle_vertices();
		const int dim = mesh_->dimension();

		V.resize(n_vertices, dim);
		F.resize(space_.basis_list().size(), dim + 1);

		for (int i = 0; i < space_.basis_list().size(); i++)
		{
			const basis::ElementBases &element = space_.basis_list()[i];
			for (int j = 0; j < element.bases.size(); j++)
			{
				const basis::Basis &basis = element.bases[j];
				assert(basis.global().size() == 1);
				V.row(basis.global()[0].index) = basis.global()[0].node;
				if (j < F.cols())
					F(i, j) = basis.global()[0].index;
			}
		}
	}

} // namespace polyfem::varform
