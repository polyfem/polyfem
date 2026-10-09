#include "GrowthElasticVarForm.hpp"

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
#include <polyfem/solver/forms/MixedAssemblerForm.hpp>
#include <polyfem/solver/forms/StackedForm.hpp>
#include <polyfem/solver/forms/lagrangian/AugmentedLagrangianForm.hpp>
#include <polyfem/solver/forms/lagrangian/BCLagrangianForm.hpp>
#include <polyfem/solver/forms/lagrangian/StackedAugmentedLagrangianForm.hpp>

#include <polyfem/utils/JSONUtils.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/MatrixUtils.hpp>
#include <polyfem/utils/Timer.hpp>

#include <igl/Timer.h>

#include <polysolve/linear/Solver.hpp>
#include <polysolve/nonlinear/Solver.hpp>

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iomanip>
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

		json solver_params_for_residual_mode(const json &solver_params, const bool is_residual)
		{
			json params = solver_params;
			if (is_residual)
				disable_newton_psd_projection(params);
			return params;
		}

		void assert_same_space_ids(
			const json &materials,
			const int displacement_space_id,
			const int growth_space_id)
		{
			for (const json &material : utils::json_as_array(materials))
			{
				if (material.at("displacement_space_id").get<int>() != displacement_space_id
					|| material.at("growth_space_id").get<int>() != growth_space_id)
				{
					log_and_throw_error("All GrowthElasticity materials must use the same FE space ids.");
				}
			}
		}

		json elastic_material_from_growth_material(const json &material)
		{
			if (!material.contains("elastic_material") || !material["elastic_material"].is_object())
				log_and_throw_error("GrowthElasticity requires elastic_material to be an elastic material object.");

			json elastic_material = material["elastic_material"];
			const std::string type = elastic_material.value("type", "");
			if (!assembler::AssemblerUtils::is_elastic_material(type))
				log_and_throw_error("GrowthElasticity elastic_material must be an elastic material, got '{}'.", type);

			if (material.contains("id"))
				elastic_material["id"] = material["id"];
			if (material.contains("rho"))
				elastic_material["rho"] = material["rho"];

			return elastic_material;
		}

		std::string elastic_formulation_from_growth_materials(const json &materials)
		{
			std::string formulation;
			for (const json &material : utils::json_as_array(materials))
			{
				const json elastic_material = elastic_material_from_growth_material(material);
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

	void GrowthElasticVarForm::reset()
	{
		NonlinearElasticVarForm::reset();
		growth_space_.reset();
		growth_boundary_.reset();
		growth_problem_ = nullptr;
		growth_ass_vals_cache_.init_empty();
		growth_mass_ass_vals_cache_.init_empty(true);
		growth_assembler_ = nullptr;
		growthelastic_assembler_ = nullptr;
		growth_mass_assembler_ = nullptr;
		growth_rhs_assembler_ = nullptr;
		growth_rhs_density_ = std::make_shared<assembler::NoDensity>();
		growth_mass_.resize(0, 0);
		stacked_lumped_mass_.resize(0, 0);
		growth_rhs_.resize(0, 0);
		growth_coupling_form_ = nullptr;
		stacked_form_ = nullptr;
		converged_solution_.resize(0, 0);
		prescribed_growth_.resize(0, 0);
		displacement_space_id_ = -1;
		growth_space_id_ = -1;
		elastic_formulation_ = "MaterialSum";
	}

	void GrowthElasticVarForm::init(
		const std::string &formulation,
		const Units &units,
		const json &args,
		const std::string &out_path)
	{
		VarForm::init(formulation, units, args, out_path);
		read_material_space_ids(args);

		// Stage 1 of the growth build is static with theta prescribed; growth
		// evolution in pseudo-time is Stage 2 and follows its own design.
		if (args.contains("time") && !args["time"].is_null())
			log_and_throw_error("GrowthElasticity supports static solves only (growth evolution is a later stage).");

		// Contact is inherited u-block machinery and structurally functional here
		// (collision mesh, barrier stiffness update), but the growth+contact
		// combination is untested; decline loudly until it has its own validation.
		if (args["contact"]["enabled"].get<bool>())
			log_and_throw_error("GrowthElasticity does not yet support contact (untested combination).");

		primary_assembler_ = assembler::AssemblerUtils::make_assembler(elastic_formulation_);
		if (args["solver"]["advanced"]["check_inversion"] == "Conservative")
		{
			if (auto elastic_assembler = std::dynamic_pointer_cast<assembler::ElasticityAssembler>(primary_assembler_))
				elastic_assembler->set_use_robust_jacobian();
		}

		// The growth space carries NO PDE of its own in Stage 1: this
		// Laplacian is used purely structurally (FE-space construction and
		// Dirichlet-value plumbing through the RhsAssembler); it is never
		// registered as a form. Its material parameter is therefore optional
		// and unused; the name reserves the slot a Stage-2 gradient
		// regularizer would occupy.
		growth_assembler_ = std::make_shared<assembler::Laplacian>("growth_diffusivity");
		growthelastic_assembler_ = assembler::AssemblerUtils::make_mixed_nl_assembler("GrowthElasticity");
		mass_assembler_ = std::make_shared<assembler::Mass>();
		pure_mass_assembler_ = std::make_shared<assembler::HRZMass>();
		// Unit-density mass on the growth space: a well-conditioned Gram
		// matrix for the augmented-Lagrangian BC form (the growth material has
		// no physical density).
		growth_mass_assembler_ = std::make_shared<assembler::Mass>(std::make_shared<assembler::NoDensity>());

		problem = std::make_shared<assembler::GenericTensorProblem>("GrowthDisplacement");
		problem->clear();
		growth_problem_ = std::make_shared<assembler::GenericScalarProblem>("GrowthField");
		growth_problem_->clear();

		json tmp;
		tmp["is_time_dependent"] = false;
		problem->set_parameters(tmp, root_path);
		growth_problem_->set_parameters(tmp, root_path);

		auto bc = args["boundary_conditions"];
		bc["root_path"] = root_path;
		problem->set_parameters(bc, root_path);
		growth_problem_->set_parameters(bc, root_path);
		problem->set_parameters(args["initial_conditions"], root_path);
		growth_problem_->set_parameters(args["initial_conditions"], root_path);
		problem->set_parameters(args["output"], root_path);
		growth_problem_->set_parameters(args["output"], root_path);

		problem->set_units(*primary_assembler_, units);
		growth_problem_->set_units(*growth_assembler_, units);

		t0 = 0.0;
		time_steps = 0;
		dt = 0.0;
		contact_dhat_was_explicit_ = args["contact"].value("_dhat_was_explicit", false);
		this->args["contact"].erase("_dhat_was_explicit");
	}

	void GrowthElasticVarForm::read_material_space_ids(const json &args)
	{
		const json material = first_material(args.at("materials"));
		displacement_space_id_ = material.at("displacement_space_id").get<int>();
		growth_space_id_ = material.at("growth_space_id").get<int>();
		if (displacement_space_id_ == growth_space_id_)
			log_and_throw_error("GrowthElasticity requires distinct displacement and growth FE spaces.");

		elastic_formulation_ = elastic_formulation_from_growth_materials(args.at("materials"));
		assert_same_space_ids(args.at("materials"), displacement_space_id_, growth_space_id_);
	}

	json GrowthElasticVarForm::elastic_material_args() const
	{
		if (args["materials"].is_array())
		{
			json materials = json::array();
			for (const json &material : args["materials"])
				materials.push_back(elastic_material_from_growth_material(material));
			return materials;
		}

		return elastic_material_from_growth_material(args["materials"]);
	}

	void GrowthElasticVarForm::load_mesh(const mesh::Mesh &mesh, const json &args)
	{
		assert(mesh_);
		std::vector<int> body_ids(mesh.n_elements());
		for (int i = 0; i < mesh.n_elements(); ++i)
			body_ids[i] = mesh.get_body_id(i);

		const json elastic_materials = elastic_material_args();

		primary_assembler_->set_size(mesh.dimension());
		primary_assembler_->set_materials(body_ids, elastic_materials, units, root_path);
		growthelastic_assembler_->set_size(mesh.dimension());
		growthelastic_assembler_->set_materials(body_ids, args["materials"], units, root_path);
		mass_assembler_->set_size(mesh.dimension());
		mass_assembler_->set_materials(body_ids, elastic_materials, units, root_path);
		pure_mass_assembler_->set_size(mass_assembler_->size());

		growth_assembler_->set_size(1);
		growth_assembler_->set_materials(body_ids, args["materials"], units, root_path);
		growth_mass_assembler_->set_size(1);
		// no set_materials: the growth mass uses NoDensity (unit density), which
		// throws on add_multimaterial by design -- it needs no material setup

		problem->init(mesh);
		growth_problem_->init(mesh);

		logger().info("Loading obstacles...");
		obstacle = mesh::read_obstacle_geometry(
			units,
			args["geometry"],
			utils::json_as_array(args["boundary_conditions"]["obstacle_displacements"]),
			utils::json_as_array(args["boundary_conditions"]["dirichlet_boundary"]),
			root_path, mesh.dimension());
	}

	void GrowthElasticVarForm::build_basis(mesh::Mesh &mesh, const bool iso_parametric, const json &args)
	{
		assert(problem);
		assert(growth_problem_);
		assert(primary_assembler_);
		assert(growth_assembler_);

		Eigen::VectorXi displacement_orders, displacement_ordersq;
		assign_discr_orders(args["space"], displacement_space_id_, mesh, displacement_orders, displacement_ordersq);

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
			displacement_ordersq,
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

		build_growth_basis(mesh, iso_parametric, args);
		build_growth_boundary(mesh);

		const auto &current_bases = space_.geometry_basis_list();
		if (args["space"]["advanced"]["count_flipped_els"])
			stats.count_flipped_elements(mesh, current_bases);

		const int n_samples = 10;
		stats.compute_mesh_size(mesh, current_bases, n_samples, args["output"]["advanced"]["curved_mesh_size"]);
		logger().info("flipped elements {}", stats.n_flipped);
		logger().info("h: {}", stats.mesh_size);

		if (std::max(space_.n_bases, growth_space_.n_bases) <= args["solver"]["advanced"]["cache_size"])
		{
			igl::Timer timer;
			timer.start();
			logger().info("Building cache...");
			ass_vals_cache_.init(mesh.is_volume(), space_.basis_list(), current_bases);
			mass_ass_vals_cache_.init(mesh.is_volume(), space_.basis_list(), current_bases, true);
			pure_mass_ass_vals_cache_.init(mesh.is_volume(), space_.basis_list(), current_bases, true);
			growth_ass_vals_cache_.init(mesh.is_volume(), growth_space_.basis_list(), growth_space_.geometry_basis_list());
			growth_mass_ass_vals_cache_.init(mesh.is_volume(), growth_space_.basis_list(), growth_space_.geometry_basis_list(), true);
			logger().info(" took {}s", timer.getElapsedTime());
		}
		else
		{
			ass_vals_cache_.init_empty();
			mass_ass_vals_cache_.init_empty(true);
			pure_mass_ass_vals_cache_.init_empty(true);
			growth_ass_vals_cache_.init_empty();
			growth_mass_ass_vals_cache_.init_empty(true);
		}
	}

	void GrowthElasticVarForm::build_displacement_boundary(mesh::Mesh &mesh)
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

	void GrowthElasticVarForm::build_growth_basis(mesh::Mesh &mesh, const bool iso_parametric, const json &args)
	{
		Eigen::VectorXi growth_orders, growth_ordersq;
		assign_discr_orders(args["space"], growth_space_id_, mesh, growth_orders, growth_ordersq);

		build_fe_space(
			mesh,
			iso_parametric,
			growth_orders,
			growth_ordersq,
			args["space"]["basis_type"],
			args["space"]["poly_basis_type"],
			*growth_assembler_,
			/*value_dim=*/1,
			args["space"]["advanced"]["quadrature_order"],
			args["space"]["advanced"]["mass_quadrature_order"],
			args["space"]["advanced"]["use_corner_quadrature"],
			args["space"]["advanced"]["n_harmonic_samples"],
			args["space"]["advanced"]["integral_constraints"],
			growth_space_,
			growth_boundary_,
			space_.geometry);

		// build_fe_space builds the input->internal node map only for spaces that
		// own their geometry; the growth space shares the displacement geometry,
		// so it must build its own (else growth_nodal_field and "growth" state
		// restarts are placed in internal order -- caught by C2b).
		build_node_mapping(mesh, args["space"]["basis_type"], growth_space_,
						   growth_space_.space_in_node_to_node,
						   growth_space_.space_in_primitive_to_primitive);

		// PolyFEM computes no input orderings for this mesh path, so the call
		// above returns an empty map. For P1 each growth node lies on one mesh
		// vertex, and the loader keeps mesh vertices in file order, so the
		// vertex -> node table of the growth space IS the input -> node map.
		if (growth_space_.space_in_node_to_node.size() == 0 && growth_space_.mesh_nodes
			&& growth_space_.disc_orders.size() > 0
			&& growth_space_.disc_orders.minCoeff() == 1 && growth_space_.disc_orders.maxCoeff() == 1)
		{
			const auto &p2n = growth_space_.mesh_nodes->primitive_to_node();
			const int nv = mesh.n_vertices();
			Eigen::VectorXi vmap(nv);
			for (int v = 0; v < nv; ++v)
			{
				const int n = p2n[growth_space_.mesh_nodes->primitive_from_vertex(v)];
				if (n < 0 || n >= growth_space_.n_bases)
					log_and_throw_error("GrowthElasticity: mesh vertex {} has no P1 growth node ({}).", v, n);
				vmap(v) = n;
			}
			growth_space_.space_in_node_to_node = vmap;
			logger().info("GrowthElasticity: growth node map built from mesh vertices ({} nodes).", nv);
		}

		logger().info("n growth bases: {}", growth_space_.n_bases);
	}

	void GrowthElasticVarForm::build_growth_boundary(mesh::Mesh &mesh)
	{
		growth_boundary_.clear_boundary_conditions();

		growth_problem_->update_nodes(growth_space_.space_in_node_to_node);

		growth_problem_->setup_bc(
			mesh,
			assembler::BoundaryKind::Dirichlet,
			growth_space_id_,
			growth_space_.basis_list(),
			growth_boundary_.total_local_boundary,
			growth_boundary_.local_boundary,
			growth_boundary_.boundary_nodes,
			/*value_dim=*/1);
		std::vector<int> unused_neumann_boundary_nodes;
		growth_problem_->setup_bc(
			mesh,
			assembler::BoundaryKind::Neumann,
			growth_space_id_,
			growth_space_.basis_list(),
			growth_boundary_.total_local_boundary,
			growth_boundary_.local_neumann_boundary,
			unused_neumann_boundary_nodes,
			/*value_dim=*/1);

		growth_problem_->setup_nodal_bc(
			mesh,
			assembler::BoundaryKind::Dirichlet,
			growth_space_id_,
			growth_space_.n_bases,
			growth_boundary_.dirichlet_nodes);
		growth_problem_->setup_nodal_bc(
			mesh,
			assembler::BoundaryKind::Neumann,
			growth_space_id_,
			growth_space_.n_bases,
			growth_boundary_.neumann_nodes);

		for (const int n_id : growth_boundary_.dirichlet_nodes)
			growth_boundary_.boundary_nodes.push_back(n_id);

		growth_boundary_.normalize_boundary_nodes();
		rebuild_node_positions(growth_space_.basis_list(), growth_boundary_.dirichlet_nodes, growth_boundary_.dirichlet_nodes_position);
		rebuild_node_positions(growth_space_.basis_list(), growth_boundary_.neumann_nodes, growth_boundary_.neumann_nodes_position);
	}

	void GrowthElasticVarForm::build_rhs_assembler()
	{
		json rhs_solver_params = args["solver"]["linear"];
		if (!rhs_solver_params.contains("Pardiso"))
			rhs_solver_params["Pardiso"] = {};
		rhs_solver_params["Pardiso"]["mtype"] = -2;

		solve_data_.rhs_assembler = std::make_shared<assembler::RhsAssembler>(
			*primary_assembler_, *mesh_, &obstacle,
			boundary_.dirichlet_nodes, boundary_.neumann_nodes,
			boundary_.dirichlet_nodes_position, boundary_.neumann_nodes_position,
			space_.n_bases, mesh_->dimension(), space_.basis_list(), space_.geometry_basis_list(),
			mass_ass_vals_cache_, *problem,
			args["space"]["advanced"]["bc_method"],
			rhs_solver_params,
			displacement_space_id_);
		rhs_assembler_ = solve_data_.rhs_assembler;

		growth_rhs_assembler_ = std::make_shared<assembler::RhsAssembler>(
			*growth_assembler_, *mesh_, nullptr,
			growth_boundary_.dirichlet_nodes, growth_boundary_.neumann_nodes,
			growth_boundary_.dirichlet_nodes_position, growth_boundary_.neumann_nodes_position,
			growth_space_.n_bases, /*size=*/1,
			growth_space_.basis_list(), growth_space_.geometry_basis_list(),
			growth_mass_ass_vals_cache_, *growth_problem_,
			args["space"]["advanced"]["bc_method"],
			rhs_solver_params,
			growth_space_id_);
	}

	void GrowthElasticVarForm::assemble_rhs(const mesh::Mesh &mesh)
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
		growth_problem_->set_parameters(p_params, root_path);

		rhs_.resize(0, 0);
		growth_rhs_.resize(0, 0);

		timer.start();
		logger().info("Assigning rhs...");

		build_rhs_assembler();
		assert(rhs_assembler_ != nullptr);
		assert(growth_rhs_assembler_ != nullptr);
		assert(growth_rhs_density_ != nullptr);
		rhs_assembler_->assemble(mass_assembler_->density(), rhs_);
		rhs_ *= -1;
		growth_rhs_assembler_->assemble(*growth_rhs_density_, growth_rhs_);
		growth_rhs_ *= -1;

		timings.assigning_rhs_time = timer.getElapsedTime();
		logger().info(" took {}s", timings.assigning_rhs_time);
	}

	void GrowthElasticVarForm::assemble_mass_mat(const mesh::Mesh &mesh, const json &args)
	{
		mass_.resize(0, 0);
		pure_mass_.resize(0, 0);
		growth_mass_.resize(0, 0);

		igl::Timer timer;
		timer.start();
		logger().info("Assembling mass mat...");

		mass_assembler_->assemble(mesh.is_volume(), space_.n_bases, space_.basis_list(), space_.geometry_basis_list(), mass_ass_vals_cache_, 0, mass_, true);
		pure_mass_assembler_->assemble(mesh.is_volume(), space_.n_bases, space_.basis_list(), space_.geometry_basis_list(), pure_mass_ass_vals_cache_, 0, pure_mass_, true);
		growth_mass_assembler_->assemble(mesh.is_volume(), growth_space_.n_bases, growth_space_.basis_list(), growth_space_.geometry_basis_list(), growth_mass_ass_vals_cache_, 0, growth_mass_, true);

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
			growth_mass_ = utils::lump_matrix(growth_mass_);
		}

		// Static problem: the stacked lumped mass only feeds the nonlinear
		// problem's scaling. Identity on the growth block (no physical mass).
		stacked_lumped_mass_ = block_diag(
			pure_mass_.size() > 0 ? pure_mass_ : identity_mass(displacement_ndof()),
			identity_mass(growth_ndof()));

		timer.stop();
		timings.assembling_mass_mat_time = timer.getElapsedTime();
		logger().info(" took {}s", timings.assembling_mass_mat_time);

		stats.nn_zero = stacked_lumped_mass_.nonZeros();
		stats.num_dofs = stacked_lumped_mass_.rows();
		stats.mat_size = (long long)stacked_lumped_mass_.rows() * (long long)stacked_lumped_mass_.cols();
		logger().info("sparsity: {}/{}", stats.nn_zero, stats.mat_size);
	}

	const Eigen::MatrixXd &GrowthElasticVarForm::prescribed_growth_field() const
	{
		if (prescribed_growth_.size() > 0)
			return prescribed_growth_;

		const std::string path = args["boundary_conditions"]["growth_nodal_field"];
		if (path.empty())
			return prescribed_growth_; // empty: no whole-field prescription

		std::ifstream in(resolve_input_path(path));
		if (!in)
			log_and_throw_error("GrowthElasticity: cannot open growth_nodal_field '{}'.", path);

		std::vector<double> values;
		values.reserve(growth_ndof());
		double v;
		while (in >> v)
			values.push_back(v);

		if (int(values.size()) != growth_ndof())
			log_and_throw_error(
				"GrowthElasticity: growth_nodal_field '{}' has {} values, expected {} (one per growth node, mesh-vertex order).",
				path, values.size(), growth_ndof());

		Eigen::MatrixXd field = Eigen::Map<Eigen::VectorXd>(values.data(), values.size());

		// Mesh-vertex (input) -> internal node order. The internal order is NOT
		// the file order in general (C2b: 57% of the quarter-tube vertices
		// differ). No AL diagnostic can catch a wrong map here: initial and
		// target both come from this function. C2b (c2b_ordering.py) is the check.
		const auto &map = growth_space_.space_in_node_to_node;
		if (map.size() != field.rows())
			log_and_throw_error(
				"GrowthElasticity: growth-space node map has {} entries for {} values; "
				"growth_nodal_field cannot be placed on the right nodes.",
				map.size(), field.rows());
		Eigen::MatrixXd remapped(field.rows(), 1);
		for (int i = 0; i < map.size(); ++i)
			remapped(map(i)) = field(i);
		field = remapped;

		if ((field.array() <= 0).any())
			log_and_throw_error(
				"GrowthElasticity: growth_nodal_field '{}' contains non-positive theta (min {}).",
				path, field.minCoeff());

		prescribed_growth_ = field;
		return prescribed_growth_;
	}

	void GrowthElasticVarForm::initial_growth_solution(Eigen::MatrixXd &solution) const
	{
		// A whole-field prescription is also the initial solution: initial ==
		// target makes the AL start error ~0 and the first iterate consistent.
		const Eigen::MatrixXd &prescribed = prescribed_growth_field();
		if (prescribed.size() > 0)
		{
			solution = prescribed;
			return;
		}

		assert(growth_rhs_assembler_ != nullptr);

		const bool was_solution_loaded = read_initial_x_from_file(
			resolve_input_path(args["input"]["data"]["state"]), "growth",
			args["input"]["data"]["reorder"], growth_space_.space_in_node_to_node,
			/*dim=*/1, solution);

		if (!was_solution_loaded)
			growth_rhs_assembler_->initial_solution(solution);

		// theta = 0 is singular (Fg not invertible); the harmless zero default
		// of the temperature precedent is fatal here, so an all-zero initial
		// growth field -- i.e. no file and no initial_conditions entry for the
		// growth space -- is replaced by the identity, theta = 1, making the
		// first iterate pure elasticity per the formulation's initialization
		// convention. A user-prescribed nonzero field is left untouched.
		if (solution.size() > 0 && (solution.array() == 0.0).all())
		{
			logger().info("Growth field initialized to theta = 1 (identity growth).");
			solution.setOnes();
		}
	}

	Eigen::MatrixXd GrowthElasticVarForm::stacked_solution(
		const Eigen::MatrixXd &displacement,
		const Eigen::MatrixXd &growth) const
	{
		assert(displacement.rows() == displacement_ndof());
		assert(growth.rows() == growth_ndof());
		assert(displacement.cols() == growth.cols());

		Eigen::MatrixXd solution(displacement.rows() + growth.rows(), displacement.cols());
		solution << displacement, growth;
		return solution;
	}

	void GrowthElasticVarForm::split_solution(
		const Eigen::MatrixXd &solution,
		Eigen::MatrixXd &displacement,
		Eigen::MatrixXd &growth) const
	{
		assert(solution.rows() == total_ndof());
		displacement = solution.topRows(displacement_ndof());
		growth = solution.bottomRows(growth_ndof());
	}

	void GrowthElasticVarForm::build_forms(Eigen::MatrixXd &solution, const double t)
	{
		assert(solution.cols() == 1);
		assert(solution.rows() == total_ndof());

		stacked_form_ = std::make_shared<solver::StackedForm>();
		const auto displacement_block = stacked_form_->add_block(displacement_ndof());
		const auto growth_block = stacked_form_->add_block(growth_ndof());

		Eigen::MatrixXd displacement, growth;
		split_solution(solution, displacement, growth);

		solve_data_.time_integrator = nullptr;

		init_forms(args, mesh_->dimension(), displacement, t);
		for (const auto &form : forms)
		{
			assert(form);
			stacked_form_->add(displacement_block, form);
		}
		solve_data_.al_form.clear();

		// No growth-block form: Stage 1 has no growth PDE. The growth block's
		// stiffness is the K_gg quadrant of the coupling form below plus the
		// augmented-Lagrangian boundary form; with every growth DOF prescribed
		// the reduced problem contains only displacements.

		assert(growthelastic_assembler_);
		growth_coupling_form_ = std::make_shared<solver::MixedAssemblerForm>(
			space_.n_bases, growth_space_.n_bases,
			space_.basis_list(), growth_space_.basis_list(), space_.geometry_basis_list(),
			*growthelastic_assembler_, ass_vals_cache_, growth_ass_vals_cache_,
			t, /*dt=*/0.0, mesh_->is_volume());
		stacked_form_->add(displacement_block, growth_block, growth_coupling_form_);

		forms.clear();
		forms.push_back(stacked_form_);
		for (const auto &form : forms)
			form->set_output_dir(output_path);

		solve_data_.al_form.clear();
		if (!boundary_.boundary_nodes.empty() || !growth_boundary_.boundary_nodes.empty()
			|| prescribed_growth_field().size() > 0)
		{
			auto stacked_al = std::make_shared<solver::StackedAugmentedLagrangianForm>();
			const auto displacement_al_block = stacked_al->add_block(displacement_block.size());
			const auto growth_al_block = stacked_al->add_block(growth_block.size());

			if (!boundary_.boundary_nodes.empty())
			{
				stacked_al->add(
					displacement_al_block,
					std::make_shared<solver::BCLagrangianForm>(
						displacement_block.size(),
						boundary_.boundary_nodes, boundary_.local_boundary, boundary_.local_neumann_boundary,
						elastic_boundary_samples(), mass_, *rhs_assembler_,
						obstacle.n_vertices() * mesh_->dimension(), /*is_time_dependent=*/false, t));
			}

			const Eigen::MatrixXd &prescribed = prescribed_growth_field();
			if (prescribed.size() > 0)
			{
				// Whole-field prescription: every growth DOF is Dirichlet with
				// tabulated targets (BCLagrangianForm's target constructor; no
				// RhsAssembler involved). Supersedes any growth-space surface
				// Dirichlet entries.
				if (!growth_boundary_.boundary_nodes.empty())
					logger().warn("growth_nodal_field supersedes {} growth-space dirichlet_boundary node(s).", growth_boundary_.boundary_nodes.size());

				std::vector<int> all_growth_nodes(growth_ndof());
				for (int i = 0; i < growth_ndof(); ++i)
					all_growth_nodes[i] = i;

				stacked_al->add(
					growth_al_block,
					std::make_shared<solver::BCLagrangianForm>(
						growth_block.size(), all_growth_nodes, growth_mass_,
						/*obstacle_ndof=*/0, prescribed));
			}
			else if (!growth_boundary_.boundary_nodes.empty())
			{
				assert(growth_space_.disc_orders.size() > 0 && "Growth boundary quadrature requires initialized FE orders");
				const int gdiscr_order = mesh_->orders().size() <= 0 ? 1 : mesh_->orders().maxCoeff();
				const QuadratureOrders growth_boundary_samples =
					n_boundary_samples(growth_space_.disc_orders.maxCoeff(), growth_space_.disc_ordersq.maxCoeff(), gdiscr_order);

				stacked_al->add(
					growth_al_block,
					std::make_shared<solver::BCLagrangianForm>(
						growth_block.size(),
						growth_boundary_.boundary_nodes, growth_boundary_.local_boundary,
						growth_boundary_.local_neumann_boundary, growth_boundary_samples,
						growth_mass_, *growth_rhs_assembler_,
						/*obstacle_ndof=*/0, /*is_time_dependent=*/false, t));
			}

			solve_data_.al_form.push_back(stacked_al);
		}
	}

	void GrowthElasticVarForm::solve_nonlinear_step(const int step, Eigen::MatrixXd &solution)
	{
		assert(solve_data_.nl_problem != nullptr && "Growth forms must initialize the nonlinear problem before solving");
		solver::NLProblem &nl_problem = *solve_data_.nl_problem;

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
			solve_data_.update_barrier_stiffness(displacement);
		};

		if (!solve_data_.al_form.empty())
		{
			solver::ALSolver al_solver(
				solve_data_.al_form,
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

	void GrowthElasticVarForm::solve_problem(
		Eigen::MatrixXd &sol,
		const InitialConditionOverride *initial_condition_override,
		const ForwardStepCallback &post_step)
	{
		assert(!initial_condition_override && "GrowthElasticity does not support initial-condition overrides");
		assert(!post_step && "GrowthElasticity does not support post-step callbacks");

		stats.spectrum.setZero();

		igl::Timer timer;
		timer.start();
		logger().info("Solving GrowthElasticity");

		{
			POLYFEM_SCOPED_TIMER("Setup RHS");

			if (sol.size() <= 0)
			{
				Eigen::MatrixXd displacement, growth;
				initial_solution(displacement);
				initial_growth_solution(growth);
				const int cols = std::max(displacement.cols(), growth.cols());
				if (displacement.cols() != cols)
					displacement.conservativeResize(Eigen::NoChange, cols);
				if (growth.cols() != cols)
					growth.conservativeResize(Eigen::NoChange, cols);
				sol = stacked_solution(displacement, growth);
			}

			if (sol.cols() > 1)
				sol.conservativeResize(Eigen::NoChange, 1);
		}

		build_forms(sol, /*t=*/1.0);

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

		solve_data_.nl_problem = std::make_shared<solver::NLProblem>(
			total_ndof(), /*t=*/1.0,
			forms, solve_data_.al_form,
			polysolve::linear::Solver::create(args["solver"]["linear"], logger()),
			characteristic_length, characteristic_force_density,
			stacked_lumped_mass_.size() > 0 ? stacked_lumped_mass_ : identity_mass(total_ndof()),
			mesh_->dimension(),
			/*is_time_dependent=*/false);
		solve_data_.nl_problem->init(sol);
		solve_data_.nl_problem->update_quantities(/*t=*/1.0, sol);
		stats.solver_info = json::array();

		solve_nonlinear_step(0, sol);
		converged_solution_ = sol;

		timer.stop();
		timings.solving_time = timer.getElapsedTime();
		logger().info(" took {}s", timings.solving_time);
	}

	void GrowthElasticVarForm::export_data(const Eigen::MatrixXd &solution) const
	{
		Eigen::MatrixXd displacement, growth;
		split_solution(solution, displacement, growth);

		// The inherited exporter expects a pure-displacement solution; handing
		// it the displacement block keeps state files written by the SAME code
		// path as plain elastic runs (directly comparable, e.g. for the V1
		// bit-identity). The growth block goes to a full-precision sidecar.
		NonlinearElasticVarForm::export_data(displacement);

		// The elastic state writer lives in the plain static solve path (not in
		// export_data), so this VarForm writes both blocks itself as
		// full-precision text: %.17g round-trips IEEE doubles exactly, so
		// these files support bit-level comparison against plain-elastic runs.
		const std::string state = args["output"]["data"]["state"];
		if (!state.empty())
		{
			std::ofstream out_u(state + ".disp.txt");
			out_u << std::setprecision(17) << displacement << "\n";
			std::ofstream out_g(state + ".growth.txt");
			out_g << std::setprecision(17) << growth << "\n";
		}
	}

	io::OutStatsData GrowthElasticVarForm::compute_errors(const Eigen::MatrixXd &solution)
	{
		if (!args["output"]["advanced"]["compute_error"])
			return stats;

		Eigen::MatrixXd displacement, growth;
		split_solution(solution, displacement, growth);

		stats.compute_errors(space_.n_bases, space_.basis_list(), space_.geometry_basis_list(), *mesh_, *problem, /*tend=*/0, displacement);
		return stats;
	}

	std::vector<io::OutputField> GrowthElasticVarForm::output_fields(
		const io::OutputSample &sample,
		const Eigen::MatrixXd &solution,
		const io::OutputFieldOptions &options) const
	{
		// See converged_solution_: prefer the cached stacked solution when the
		// passed one is not stacked-sized (export paths differ in what they
		// hand this function; slicing a wrong-sized vector fed garbage theta
		// into the reaction evaluation and tripped the positivity guard).
		const bool passed_is_stacked = solution.rows() == total_ndof();
		const Eigen::MatrixXd &stacked =
			passed_is_stacked ? solution : converged_solution_;
		if (stacked.rows() != total_ndof())
			return NonlinearElasticVarForm::output_fields(sample, solution, options);

		Eigen::MatrixXd displacement, growth;
		split_solution(stacked, displacement, growth);

		std::vector<io::OutputField> fields =
			NonlinearElasticVarForm::output_fields(sample, displacement, options);
		// The base fields carry the displacement block under the name
		// "solution", which would be misleading beside the growth fields;
		// rename rather than drop, so the vtu is self-contained (points and
		// displacement in one consistent order -- what verification tooling
		// pairs against).
		for (io::OutputField &field : fields)
		{
			if (field.name == "solution")
				field.name = "displacement";
			else if (field.name == "solution_gradient")
				field.name = "displacement_gradient";
		}

		if (!mesh_ || growth.size() <= 0)
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

		// Sample a nodal scalar field of the growth space at the requested
		// output locations (same pattern as the temperature sampling in the
		// thermo precedent).
		const auto sample_growth_scalar = [&](const Eigen::MatrixXd &nodal, Eigen::MatrixXd &values, Eigen::MatrixXd *gradients = nullptr) -> bool {
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
						*mesh_, 1, growth_space_.basis_list(), growth_space_.geometry_basis_list(),
						element_id, sample.local_points.row(i), nodal, local_sol, local_grad);
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
					if (node_id < 0 || node_id >= nodal.rows())
						return false;
					values(i) = nodal(node_id);
				}
				return sample.points.rows() == 0 || sample.points.rows() == values.rows();
			}

			return false;
		};

		const bool export_growth_gradient =
			!options.fields.empty() && options.export_field("growth_gradient");
		if (options.export_field("growth") || export_growth_gradient)
		{
			Eigen::MatrixXd values, gradients;
			if (sample_growth_scalar(growth, values, export_growth_gradient ? &gradients : nullptr))
			{
				if (options.export_field("growth"))
					fields.push_back({"growth", values, io::OutputField::Association::Point});
				if (export_growth_gradient)
					fields.push_back({"growth_gradient", gradients, io::OutputField::Association::Point});
			}
		}

		// Conjugate reaction on the growth block: the growth rows of the
		// stacked energy gradient at the converged full solution. With theta
		// prescribed everywhere this is the driving-force/reaction field r of
		// the formulation (the V3 deliverable); the augmented-Lagrangian
		// boundary forms are NOT part of stacked_form_, so this is the pure
		// energy gradient.
		if (stacked_form_ && options.export_field("growth_reaction"))
		{
			Eigen::VectorXd full_grad;
			stacked_form_->first_derivative(stacked.col(0), full_grad);
			const Eigen::MatrixXd reaction_nodal = full_grad.tail(growth_ndof());

			Eigen::MatrixXd values;
			if (sample_growth_scalar(reaction_nodal, values))
				fields.push_back({"growth_reaction", values, io::OutputField::Association::Point});
		}

		if (growthelastic_assembler_)
			append_material_fields(*growthelastic_assembler_);

		return fields;
	}
} // namespace polyfem::varform
