#include "GrowthElasticVarForm.hpp"

#include <polyfem/assembler/AssemblerUtils.hpp>
#include <polyfem/assembler/ElementAssemblyValues.hpp>
#include <polyfem/assembler/GenericProblem.hpp>
#include <polyfem/assembler/GrowthElasticity.hpp>
#include <polyfem/assembler/Laplacian.hpp>
#include <polyfem/assembler/MatParams.hpp>

#include <polyfem/io/Evaluator.hpp>
#include <polyfem/io/MatrixIO.hpp>

#include <polyfem/mesh/GeometryReader.hpp>

#include <polyfem/refinement/APriori.hpp>

#include <polyfem/solver/ALSolver.hpp>
#include <polyfem/solver/NLProblem.hpp>
#include <polyfem/solver/forms/GrowthDissipationForm.hpp>
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
#include <limits>
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
		growth_law_ = GrowthLawSettings{};
		threshold_per_element_.resize(0);
		threshold_parallel_.resize(0);
		shear_supply_.resize(0);
		shear_supply_logged_once_ = false;
		threshold_clamped_.resize(0);
		lag_residual_.resize(0);
		attainment_.resize(0);
		dissipation_.resize(0);
		shear_work_.resize(0);
		element_volume_.resize(0);
		normal_growth_per_element_.resize(0);
		growth_node_volume_.resize(0);
		growth_node_weights_.resize(0);
		theta_n_.resize(0);
		pinned_growth_nodes_.clear();
		growth_pinned_mask_.clear();
		growth_activation_step_.resize(0);
		dissipation_form_ = nullptr;
	}

	void GrowthElasticVarForm::read_growth_law(const json &args)
	{
		growth_law_ = GrowthLawSettings{};
		const json material = first_material(args["materials"]);
		if (!material.contains("growth_law") || material["growth_law"].is_null())
			return;

		// jse always injects the block with its defaults, so "off" is the
		// threshold sentinel "none" (the default), not an absent block. A block
		// whose other settings were changed but whose threshold was left out is
		// refused rather than silently ignored.
		const json &gl = material["growth_law"];
		const bool has_threshold = gl.contains("threshold") && (gl["threshold"].is_number() || gl["threshold"].is_object());
		if (!has_threshold)
		{
			const GrowthLawSettings d;
			if (gl.value("n_steps", d.n_steps) != d.n_steps || gl.value("t_end", d.t_end) != d.t_end
				|| gl.value("max_active_set_iterations", d.max_active_set_iterations) != d.max_active_set_iterations
				|| gl.value("active_set_tol", d.active_set_tol) != d.active_set_tol
				|| gl.value("feasibility_tol", d.feasibility_tol) != d.feasibility_tol
				|| gl.value("theta_floor_fraction", d.theta_floor_fraction) != d.theta_floor_fraction
				|| gl.value("save_increments", d.save_increments) != d.save_increments
				|| gl.value("shear_supply", d.shear_supply) != d.shear_supply
				|| gl.value("theta_crit", d.theta_crit) != d.theta_crit
				|| gl.value("parallel_field", d.parallel_field) != d.parallel_field
				|| gl.value("shear_update", d.shear_update) != d.shear_update
				|| gl.value("shear_update_tol", d.shear_update_tol) != d.shear_update_tol
				|| gl.value("parallel_threshold", d.parallel_threshold) != d.parallel_threshold
				|| gl.value("threshold_state", d.threshold_state) != d.threshold_state
				|| gl.value("lag_report", d.lag_report) != d.lag_report
				|| gl.value("final_tol_factor", d.final_tol_factor) != d.final_tol_factor)
				log_and_throw_error("GrowthElasticity growth_law settings are given but no threshold "
									"(a number in Pa, or a per_element_file); set one to enable the law.");
			return;
		}

		growth_law_.enabled = true;
		growth_law_.n_steps = gl.value("n_steps", growth_law_.n_steps);
		growth_law_.t_end = gl.value("t_end", growth_law_.t_end);
		growth_law_.max_active_set_iterations = gl.value("max_active_set_iterations", growth_law_.max_active_set_iterations);
		growth_law_.active_set_tol = gl.value("active_set_tol", growth_law_.active_set_tol);
		growth_law_.feasibility_tol = gl.value("feasibility_tol", growth_law_.feasibility_tol);
		growth_law_.theta_floor_fraction = gl.value("theta_floor_fraction", growth_law_.theta_floor_fraction);
		growth_law_.save_increments = gl.value("save_increments", growth_law_.save_increments);
		growth_law_.shear_supply = gl.value("shear_supply", growth_law_.shear_supply);
		growth_law_.theta_crit = gl.value("theta_crit", growth_law_.theta_crit);
		growth_law_.parallel_field = gl.value("parallel_field", growth_law_.parallel_field);
		growth_law_.shear_update = gl.value("shear_update", growth_law_.shear_update);
		growth_law_.shear_update_tol = gl.value("shear_update_tol", growth_law_.shear_update_tol);
		growth_law_.parallel_threshold = gl.value("parallel_threshold", growth_law_.parallel_threshold);
		growth_law_.threshold_state = gl.value("threshold_state", growth_law_.threshold_state);
		growth_law_.lag_report = gl.value("lag_report", growth_law_.lag_report);
		growth_law_.final_tol_factor = gl.value("final_tol_factor", growth_law_.final_tol_factor);
		if (!(growth_law_.final_tol_factor > 0) || growth_law_.final_tol_factor > 1)
			log_and_throw_error("growth_law final_tol_factor must be in (0, 1], got {}.", growth_law_.final_tol_factor);
		{
			const std::string &pt = growth_law_.parallel_threshold;
			if (pt != "file" && pt != "state" && pt != "lagged")
				log_and_throw_error("growth_law parallel_threshold must be \"file\", \"state\" or \"lagged\", got \"{}\".", pt);
			if (pt != "file" && !(growth_law_.theta_crit > 0))
				log_and_throw_error("growth_law parallel_threshold \"{}\" needs theta_crit > 0 (the C_check construction).", pt);
			if (pt == "lagged" && growth_law_.shear_supply != "lagged")
				log_and_throw_error("growth_law parallel_threshold \"lagged\" needs shear_supply \"lagged\" (both parts follow the state).");
		}
		if (growth_law_.shear_update != "increment" && growth_law_.shear_update != "predictor" && growth_law_.shear_update != "pass")
			log_and_throw_error("growth_law shear_update must be \"increment\", \"predictor\" or \"pass\", got \"{}\".", growth_law_.shear_update);
		if (!(growth_law_.shear_update_tol > 0))
			log_and_throw_error("growth_law shear_update_tol must be positive.");
		if (growth_law_.shear_supply != "frozen" && growth_law_.shear_supply != "lagged")
			log_and_throw_error("growth_law shear_supply must be \"frozen\" or \"lagged\", got \"{}\".", growth_law_.shear_supply);
		if (growth_law_.shear_supply == "lagged")
		{
			if (!(growth_law_.theta_crit > 0))
				log_and_throw_error("growth_law shear_supply \"lagged\" needs theta_crit > 0 (the C_check construction).");
			if (!gl["threshold"].is_object() && gl.value("parallel_threshold", std::string("file")) == "file")
				log_and_throw_error("growth_law shear_supply \"lagged\" needs a per_element_file threshold carrying {} "
									"(or parallel_threshold \"state\" / \"lagged\").", growth_law_.parallel_field);
		}

		if (growth_law_.n_steps < 1)
			log_and_throw_error("growth_law n_steps must be >= 1, got {}.", growth_law_.n_steps);
		if (!(growth_law_.t_end > 0))
			log_and_throw_error("growth_law t_end must be positive, got {}.", growth_law_.t_end);
		if (growth_law_.max_active_set_iterations < 1)
			log_and_throw_error("growth_law max_active_set_iterations must be >= 1.");

		// Solved and prescribed growth are different modes of the same field;
		// the formulation allows partial prescription, but not through this
		// switch, so refuse the ambiguous combination rather than pick one.
		const std::string nodal = args["boundary_conditions"].value("growth_nodal_field", "");
		if (!nodal.empty())
			log_and_throw_error("growth_law (solved growth) cannot be combined with growth_nodal_field (prescribed growth).");

		for (const json &m : utils::json_as_array(args["materials"]))
			if (m.contains("growth_law") && m["growth_law"] != gl)
				log_and_throw_error("All GrowthElasticity materials must share one growth_law block "
									"(per-element thresholds go in the threshold file).");
	}

	void GrowthElasticVarForm::init(
		const std::string &formulation,
		const Units &units,
		const json &args,
		const std::string &out_path)
	{
		VarForm::init(formulation, units, args, out_path);
		read_material_space_ids(args);
		read_growth_law(args);


		// No dynamics: prescribed growth is a single static solve, and solved
		// growth (growth_law) steps its own quasi-static pseudo-time through
		// growth_law.n_steps / t_end, evaluating the displacement Dirichlet
		// expressions at each increment's t. PolyFEM's "time" block (inertia,
		// integrators) is not used.
		if (args.contains("time") && !args["time"].is_null())
			log_and_throw_error("GrowthElasticity has no dynamics: use growth_law.n_steps / t_end for pseudo-time, not \"time\".");

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
		// Solved growth evaluates the displacement Dirichlet expressions at each
		// pseudo-time increment (the production ramp), so they are not constant
		// in time. The growth problem stays static: its only Dirichlet data are
		// the active-set pins, which carry tabulated targets.
		if (growth_law_.enabled)
		{
			json td;
			td["is_time_dependent"] = true;
			problem->set_parameters(td, root_path);
		}

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

	void GrowthElasticVarForm::build_basis(mesh::Mesh &mesh, const bool iso_parametric, const json &args_in)
	{
		// Solved growth needs more than the centroid rule. At P1 the default
		// quadrature is one point per tet, so the coupling energy and the
		// dissipation see theta only through element averages, and nodal
		// patterns with zero average on every element are exact null modes
		// (hourglass modes of the growth field; on a Kuhn-split cube every tet
		// holds one node of each class (i+j+k) mod 4, and V2 converged to such a
		// pattern). With order >= 2 (four points on a tet) the nodal values map
		// invertibly onto the point values, and the energy, convex in theta
		// pointwise, fixes them. Prescribed growth is unaffected (theta is
		// data), so only the solved mode raises it. Both spaces share the order
		// (MixedAssemblerForm requires matching quadrature). Done here, on a
		// local copy, because State re-injects its args after init().
		json args = args_in;
		if (growth_law_.enabled && args["space"]["advanced"]["quadrature_order"].get<int>() < 2)
		{
			logger().info("growth_law: quadrature_order raised to 2 "
						  "(the centroid rule leaves the P1 growth field with null modes).");
			args["space"]["advanced"]["quadrature_order"] = 2;
		}

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

		// Solved growth: the rate-independent dissipation on the growth block.
		// It sits inside stacked_form_, so the stacked gradient's growth rows
		// are dPi_inc/dtheta (the KKT residual of the active-set loop).
		dissipation_form_ = nullptr;
		if (growth_law_.enabled)
		{
			assert(growth_node_weights_.size() == growth_ndof());
			assert(theta_n_.size() == growth_ndof());
			dissipation_form_ = std::make_shared<solver::GrowthDissipationForm>(
				growth_node_weights_, theta_n_, growth_law_.theta_floor_fraction);
			stacked_form_->add(growth_block, dissipation_form_);
		}

		forms.clear();
		forms.push_back(stacked_form_);
		for (const auto &form : forms)
			form->set_output_dir(output_path);

		solve_data_.al_form.clear();
		const bool has_growth_pins = growth_law_.enabled && !pinned_growth_nodes_.empty();
		if (!boundary_.boundary_nodes.empty() || !growth_boundary_.boundary_nodes.empty()
			|| prescribed_growth_field().size() > 0 || has_growth_pins)
		{
			auto stacked_al = std::make_shared<solver::StackedAugmentedLagrangianForm>();
			const auto displacement_al_block = stacked_al->add_block(displacement_block.size());
			const auto growth_al_block = stacked_al->add_block(growth_block.size());

			if (!boundary_.boundary_nodes.empty())
			{
				// Prescribed growth is one static solve; solved growth steps the
				// displacement Dirichlet data through pseudo-time.
				stacked_al->add(
					displacement_al_block,
					std::make_shared<solver::BCLagrangianForm>(
						displacement_block.size(),
						boundary_.boundary_nodes, boundary_.local_boundary, boundary_.local_neumann_boundary,
						elastic_boundary_samples(), mass_, *rhs_assembler_,
						obstacle.n_vertices() * mesh_->dimension(),
						/*is_time_dependent=*/growth_law_.enabled, t));
			}

			const Eigen::MatrixXd &prescribed = prescribed_growth_field();
			if (growth_law_.enabled)
			{
				// Active-set pins: every pinned growth DOF is Dirichlet at
				// theta_n (tabulated targets). The node list is a member because
				// BCLagrangianForm keeps a reference to it.
				if (!growth_boundary_.boundary_nodes.empty())
					log_and_throw_error("growth_law: growth-space dirichlet_boundary entries are not supported "
										"(the growth law owns the growth DOFs; tag displacement BCs with fe_space).");
				if (has_growth_pins)
					stacked_al->add(
						growth_al_block,
						std::make_shared<solver::BCLagrangianForm>(
							growth_block.size(), pinned_growth_nodes_, growth_mass_,
							/*obstacle_ndof=*/0, Eigen::MatrixXd(theta_n_)));
			}
			else if (prescribed.size() > 0)
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

	void GrowthElasticVarForm::solve_nonlinear_step(const int step, Eigen::MatrixXd &solution, const double grad_tol_factor)
	{
		assert(solve_data_.nl_problem != nullptr && "Growth forms must initialize the nonlinear problem before solving");
		solver::NLProblem &nl_problem = *solve_data_.nl_problem;

		json nonlinear_params = solver_params_for_residual_mode(args["solver"]["nonlinear"], nl_problem.is_residual());
		if (grad_tol_factor != 1.0 && nonlinear_params.contains("grad_norm_tol"))
			nonlinear_params["grad_norm_tol"] = nonlinear_params["grad_norm_tol"].get<double>() * grad_tol_factor;
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

		stats.solver_info = json::array();

		if (growth_law_.enabled)
		{
			solve_growth_law(sol);
		}
		else
		{
			build_forms(sol, /*t=*/1.0);
			make_nl_problem(sol, /*t=*/1.0);
			solve_nonlinear_step(0, sol);
		}
		converged_solution_ = sol;

		timer.stop();
		timings.solving_time = timer.getElapsedTime();
		logger().info(" took {}s", timings.solving_time);
	}

	void GrowthElasticVarForm::make_nl_problem(const Eigen::MatrixXd &sol, const double t)
	{
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
			total_ndof(), t,
			forms, solve_data_.al_form,
			polysolve::linear::Solver::create(args["solver"]["linear"], logger()),
			characteristic_length, characteristic_force_density,
			stacked_lumped_mass_.size() > 0 ? stacked_lumped_mass_ : identity_mass(total_ndof()),
			mesh_->dimension(),
			/*is_time_dependent=*/false);
		solve_data_.nl_problem->init(sol);
		solve_data_.nl_problem->update_quantities(t, sol);
	}

	// ======================================================================
	// Stage 2: solved growth
	// ======================================================================

	void GrowthElasticVarForm::compute_growth_node_weights()
	{
		const int n_el = mesh_->n_elements();
		const json material = first_material(args["materials"]);
		const json &gl = material["growth_law"];

		const auto read_scalar_field = [&](const json &spec, const std::string &what,
										   const std::string &default_field, Eigen::VectorXd &out) {
			if (spec.is_number())
			{
				out.setConstant(n_el, spec.get<double>());
				return;
			}
			if (!spec.is_object() || spec.value("type", "") != "per_element_file")
				log_and_throw_error("growth_law: {} must be a number or {{type: per_element_file, path, field}}.", what);
			const Eigen::MatrixXd values = assembler::read_growth_cell_array(
				resolve_input_path(spec.at("path").get<std::string>()), spec.value("field", default_field), 1);
			if (values.rows() != n_el)
				log_and_throw_error("growth_law: {} file has {} cells, mesh has {} elements.", what, values.rows(), n_el);
			out = values.col(0);
		};

		read_scalar_field(gl.at("threshold"), "threshold", "PHI", threshold_per_element_);
		if (!threshold_per_element_.allFinite())
			log_and_throw_error("growth_law: threshold contains non-finite values.");

		normal_growth_per_element_.setOnes(n_el);
		if (material.contains("normal_growth"))
			read_scalar_field(material["normal_growth"], "normal_growth", "VN", normal_growth_per_element_);
		if (!(normal_growth_per_element_.array() > 0).all())
			log_and_throw_error("growth_law: normal_growth must be positive.");

		const std::string &par = growth_law_.parallel_threshold;
		if (par == "file" && growth_law_.shear_supply == "lagged")
		{
			// Only the dissipative part is frozen; the shear supply starts at zero
			// and is set from the state during every increment.
			json spec = gl.at("threshold");
			spec["field"] = growth_law_.parallel_field;
			read_scalar_field(spec, "threshold (" + growth_law_.parallel_field + ")", growth_law_.parallel_field, threshold_parallel_);
			if (!threshold_parallel_.allFinite())
				log_and_throw_error("growth_law: {} contains non-finite values.", growth_law_.parallel_field);
			if ((threshold_parallel_.array() < 0).any())
				log_and_throw_error("growth_law: {} must be >= 0 (max(f_par(C_check), 0)); min {}.",
									growth_law_.parallel_field, threshold_parallel_.minCoeff());
			shear_supply_.setZero(n_el);
			threshold_per_element_ = threshold_parallel_;
		}
		else if (par == "state")
		{
			// Route B at C_check of a given displacement state, with the solver's own
			// energy and fibers (replaces the V0b threshold file): PHI_PAR frozen from
			// it, and f_gamma too unless the shear supply is lagged.
			std::string path = growth_law_.threshold_state;
			if (path.empty())
				path = args["input"]["data"]["state"].get<std::string>();
			if (path.empty())
				log_and_throw_error("growth_law parallel_threshold \"state\" needs threshold_state (or input.data.state).");
			Eigen::MatrixXd us;
			if (!read_initial_x_from_file(resolve_input_path(path), "u", args["input"]["data"]["reorder"],
										  space_.space_in_node_to_node, mesh_->dimension(), us)
				|| us.rows() != displacement_ndof())
				log_and_throw_error("growth_law: cannot read the threshold-state displacement \"u\" from {}.", path);
			Eigen::VectorXd full, ns, tt;
			evaluate_threshold_state(us.col(0), growth_law_.t_end, full, ns, tt);
			threshold_parallel_ = ns.cwiseMax(0.0);
			threshold_clamped_ = (ns.array() < 0).cast<double>().matrix();
			if (growth_law_.shear_supply == "lagged")
				shear_supply_.setZero(n_el);
			else
				shear_supply_ = full - ns;
			threshold_per_element_ = threshold_parallel_ + shear_supply_;
			logger().info("growth_law: C_check threshold from the state in {}: PHI_PAR median {:.4g} Pa, clamped on {:.1f}% "
						  "of elements; f_gamma median {:.4g} Pa ({})",
						  path, median_of(threshold_parallel_), 100.0 * threshold_clamped_.mean(), median_of(full - ns),
						  growth_law_.shear_supply == "lagged" ? "replaced per increment" : "frozen");
		}
		else if (par == "lagged")
		{
			// set from the state before any decision is made (increment start or
			// after the elastic predictor); the file value is only a placeholder
			threshold_parallel_ = threshold_per_element_.cwiseMax(0.0);
			threshold_clamped_.setZero(n_el);
			shear_supply_.setZero(n_el);
			threshold_per_element_ = threshold_parallel_;
		}

		assemble_growth_node_weights();
	}

	double GrowthElasticVarForm::median_of(const Eigen::VectorXd &v)
	{
		if (v.size() == 0)
			return std::nan("");
		std::vector<double> s(v.data(), v.data() + v.size());
		std::nth_element(s.begin(), s.begin() + s.size() / 2, s.end());
		return s[s.size() / 2];
	}

	void GrowthElasticVarForm::assemble_growth_node_weights()
	{
		const int n_el = mesh_->n_elements();

		// w_a = sum_e vn_e phi_e int_e N_a dV (growth basis, the element's own
		// quadrature), and the plain nodal volume int N_a dV for tolerances.
		growth_node_volume_.setZero(growth_ndof());
		growth_node_vn_volume_.setZero(growth_ndof());
		growth_node_weights_.setZero(growth_ndof());
		assembler::ElementAssemblyValues vals;
		for (int e = 0; e < n_el; ++e)
		{
			growth_ass_vals_cache_.compute(
				e, mesh_->is_volume(), growth_space_.basis_list()[e], growth_space_.geometry_basis_list()[e], vals);
			const Eigen::ArrayXd da = vals.det.array() * vals.quadrature.weights.array();
			const double coeff = normal_growth_per_element_(e) * threshold_per_element_(e);
			for (const auto &bv : vals.basis_values)
			{
				const double integral = (bv.val.array() * da).sum();
				for (const auto &g : bv.global)
				{
					growth_node_volume_(g.index) += g.val * integral;
					growth_node_vn_volume_(g.index) += g.val * integral * normal_growth_per_element_(e);
					growth_node_weights_(g.index) += g.val * integral * coeff;
				}
			}
		}

		if (!shear_supply_logged_once_)
		{
			logger().info("growth_law: threshold phi in [{:g}, {:g}] Pa over {} elements; {} increments to t = {:g}; "
						  "{} growth quadrature points on element 0; shear supply {}.",
						  threshold_per_element_.minCoeff(), threshold_per_element_.maxCoeff(), n_el,
						  growth_law_.n_steps, growth_law_.t_end, vals.quadrature.weights.size(), growth_law_.shear_supply);
			shear_supply_logged_once_ = true;
		}
	}

	void GrowthElasticVarForm::evaluate_threshold_state(
		const Eigen::VectorXd &u, const double t,
		Eigen::VectorXd &phi_full, Eigen::VectorXd &phi_no_shear, Eigen::VectorXd &theta_tot_out)
	{
		const auto ge = std::dynamic_pointer_cast<assembler::GrowthElasticity>(growthelastic_assembler_);
		if (!ge)
			log_and_throw_error("growth_law: the C_check threshold needs the GrowthElasticity assembler.");
		const int n_el = mesh_->n_elements();
		const int dim = mesh_->dimension();
		if (dim != 3)
			log_and_throw_error("growth_law: the C_check threshold is 3D only.");
		if (!(growth_law_.theta_crit > 0))
			log_and_throw_error("growth_law: the C_check threshold needs theta_crit > 0.");
		const double crit = growth_law_.theta_crit;
		phi_full.resize(n_el);
		phi_no_shear.resize(n_el);
		theta_tot_out.resize(n_el);

		assembler::ElementAssemblyValues vals;
		for (int e = 0; e < n_el; ++e)
		{
			ass_vals_cache_.compute(e, mesh_->is_volume(), space_.basis_list()[e], space_.geometry_basis_list()[e], vals);
			const Eigen::ArrayXd da = vals.det.array() * vals.quadrature.weights.array();
			Eigen::Matrix3d F = Eigen::Matrix3d::Zero();
			RowVectorNd pos = RowVectorNd::Zero(dim);
			for (int p = 0; p < da.size(); ++p)
			{
				Eigen::Matrix3d Fp = Eigen::Matrix3d::Identity();
				for (const auto &bv : vals.basis_values)
					for (const auto &g : bv.global)
						for (int d = 0; d < dim; ++d)
							Fp.row(d) += g.val * u(g.index * dim + d) * bv.grad_t_m.row(p);
				F += da(p) * Fp;
				pos += da(p) * vals.val.row(p);
			}
			F /= da.sum();
			pos /= da.sum();
			const Eigen::Matrix3d C = F.transpose() * F;

			// wall frame: n0 and any orthonormal in-plane pair
			const Eigen::Vector3d n = ge->wall_normal(e).normalized();
			const Eigen::Vector3d trial = std::abs(n(0)) < 0.9 ? Eigen::Vector3d::UnitX() : Eigen::Vector3d::UnitY();
			const Eigen::Vector3d e1 = (trial - trial.dot(n) * n).normalized();
			const Eigen::Vector3d e2 = n.cross(e1);
			const double c11 = e1.dot(C * e1), c22 = e2.dot(C * e2), c12 = e1.dot(C * e2);
			const double theta_tot = std::sqrt(std::max(c11 * c22 - c12 * c12, 0.0));
			if (!(theta_tot > 0))
				log_and_throw_error("growth_law: degenerate in-plane stretch on element {}.", e);
			theta_tot_out(e) = theta_tot;

			// C_check: Fg at theta = theta_tot / theta_crit (so vartheta^e = theta_crit), vn_e
			const double th = theta_tot / crit;
			const double vn = normal_growth_per_element_(e);
			const Eigen::Matrix3d nn = n * n.transpose();
			const Eigen::Matrix3d P = Eigen::Matrix3d::Identity() - nn;
			const Eigen::Matrix3d Fgi = P / std::sqrt(th) + nn / vn;
			const Eigen::Matrix3d Ce = Fgi.transpose() * C * Fgi;

			// shear-free state: transverse shears removed, Ce_nn at the layer value
			const double layer = 1.0 / n.dot(Ce.inverse() * n);
			const Eigen::Matrix3d Ce_ns = P * Ce * P + layer * nn;

			phi_full(e) = ge->driving_force(e, pos, t, Ce);
			phi_no_shear(e) = ge->driving_force(e, pos, t, Ce_ns);
		}
		if (!phi_full.allFinite() || !phi_no_shear.allFinite())
			log_and_throw_error("growth_law: non-finite C_check threshold.");
	}

	void GrowthElasticVarForm::update_lagged_threshold(const Eigen::MatrixXd &solution, const double t)
	{
		const int n_el = mesh_->n_elements();
		Eigen::VectorXd full, ns, tt;
		evaluate_threshold_state(solution.col(0).head(displacement_ndof()), t, full, ns, tt);
		shear_supply_ = full - ns;
		const bool parallel_lagged = growth_law_.parallel_threshold == "lagged";
		if (parallel_lagged)
		{
			threshold_parallel_ = ns.cwiseMax(0.0);
			threshold_clamped_ = (ns.array() < 0).cast<double>().matrix();
		}

		threshold_per_element_ = threshold_parallel_ + shear_supply_;
		assemble_growth_node_weights();

		// round-off of f_gamma on zero-floored PHI_PAR is not a negative threshold
		const double neg_tol = 1e-6 * std::max(1.0, threshold_parallel_.cwiseAbs().maxCoeff());
		const int n_neg = int((threshold_per_element_.array() < -neg_tol).count());
		const std::string par_info = parallel_lagged
										 ? fmt::format("; PHI_PAR (lagged) median {:.4g} Pa, clamped on {:.1f}%",
													   median_of(threshold_parallel_), 100.0 * threshold_clamped_.mean())
										 : std::string();
		logger().info("growth_law: lagged shear supply at t = {:g}: f_gamma median {:.4g} Pa, in [{:.4g}, {:.4g}]; "
					  "threshold in [{:.4g}, {:.4g}] Pa, negative on {:.1f}% of elements{}",
					  t, median_of(shear_supply_), shear_supply_.minCoeff(), shear_supply_.maxCoeff(),
					  threshold_per_element_.minCoeff(), threshold_per_element_.maxCoeff(), 100.0 * n_neg / n_el, par_info);
	}

	Eigen::VectorXd GrowthElasticVarForm::element_mean_growth(const Eigen::VectorXd &theta)
	{
		const int n_el = mesh_->n_elements();
		Eigen::VectorXd out(n_el);
		assembler::ElementAssemblyValues vals;
		for (int e = 0; e < n_el; ++e)
		{
			growth_ass_vals_cache_.compute(
				e, mesh_->is_volume(), growth_space_.basis_list()[e], growth_space_.geometry_basis_list()[e], vals);
			const Eigen::ArrayXd da = vals.det.array() * vals.quadrature.weights.array();
			Eigen::ArrayXd th = Eigen::ArrayXd::Zero(da.size());
			for (const auto &bv : vals.basis_values)
				for (const auto &g : bv.global)
					th += g.val * theta(g.index) * bv.val.array();
			out(e) = (th * da).sum() / da.sum();
		}
		return out;
	}

	void GrowthElasticVarForm::log_kkt_residual(const int step, const double t, const Eigen::MatrixXd &solution,
												 const Eigen::VectorXd &mult, const std::vector<char> &pinned)
	{
		// dPi/dtheta per v_n-weighted nodal volume is phi - f_I in Pa (nodal average):
		// ~0 on free DOFs (the law holds), >= 0 on pinned ones (the slack).
		std::vector<double> free_res;
		double min_slack = std::numeric_limits<double>::infinity();
		int n_pinned = 0;
		for (int a = 0; a < int(pinned.size()); ++a)
		{
			const double v = growth_node_vn_volume_(a) > 0 ? growth_node_vn_volume_(a) : 1.0;
			const double r = mult(a) / v;
			if (pinned[a])
			{
				min_slack = std::min(min_slack, r);
				++n_pinned;
			}
			else
				free_res.push_back(std::abs(r));
		}
		double med = 0, mx = 0;
		if (!free_res.empty())
		{
			mx = *std::max_element(free_res.begin(), free_res.end());
			std::nth_element(free_res.begin(), free_res.begin() + free_res.size() / 2, free_res.end());
			med = free_res[free_res.size() / 2];
		}
		logger().info("growth step {} (t = {:g}) KKT: free DOFs |phi - f_I| median {:.3g} Pa, max {:.3g} Pa ({}); "
					  "pinned min slack {:.3g} Pa ({})",
					  step, t, med, mx, free_res.size(), n_pinned > 0 ? min_slack : 0.0, n_pinned);
		stats.solver_info.push_back(
			{{"type", "growth_kkt"}, {"t", t}, {"step", step}, {"free_median_pa", med}, {"free_max_pa", mx},
			 {"pinned_min_slack_pa", n_pinned > 0 ? min_slack : 0.0}});
	}

	void GrowthElasticVarForm::accumulate_dissipation(const int step, const double t, const Eigen::VectorXd &dtheta)
	{
		// Route-B split of the increment's dissipation term v_n phi dtheta (per
		// reference volume, Pa): D = v_n max(f_par, 0) dtheta >= 0, the dissipation,
		// and S = v_n f_gamma dtheta, the shear supply (Q12 obligation 3). Needs the
		// split, i.e. a solver-side threshold or a lagged shear supply.
		const int n_el = mesh_->n_elements();
		if (threshold_parallel_.size() != n_el || shear_supply_.size() != n_el)
			return;
		if (element_volume_.size() != n_el)
		{
			element_volume_.resize(n_el);
			assembler::ElementAssemblyValues vals;
			for (int e = 0; e < n_el; ++e)
			{
				growth_ass_vals_cache_.compute(
					e, mesh_->is_volume(), growth_space_.basis_list()[e], growth_space_.geometry_basis_list()[e], vals);
				element_volume_(e) = (vals.det.array() * vals.quadrature.weights.array()).sum();
			}
			dissipation_.setZero(n_el);
			shear_work_.setZero(n_el);
		}
		const Eigen::VectorXd dth = element_mean_growth(dtheta.cwiseMax(0.0));
		const Eigen::VectorXd D = normal_growth_per_element_.cwiseProduct(threshold_parallel_).cwiseProduct(dth);
		const Eigen::VectorXd S = normal_growth_per_element_.cwiseProduct(shear_supply_).cwiseProduct(dth);
		dissipation_ += D;
		shear_work_ += S;
		const double iD = D.dot(element_volume_), iS = S.dot(element_volume_);
		const double cD = dissipation_.dot(element_volume_), cS = shear_work_.dot(element_volume_);
		int n_grow = 0, n_neg = 0, n_dneg = 0;
		for (int e = 0; e < n_el; ++e)
			if (dth(e) > 0)
			{
				++n_grow;
				n_neg += (threshold_per_element_(e) < 0);
				n_dneg += (D(e) < 0);
			}
		logger().info("growth step {} (t = {:g}): dissipation int D dV {:.4g} (cumulative {:.4g}), shear supply int S dV "
					  "{:.4g} (cumulative {:.4g}) [Pa x mesh volume]; S/D {:.3f}; growing elements {}, with phi < 0: {}, with D < 0: {}",
					  step, t, iD, cD, iS, cS, iD != 0 ? iS / iD : 0.0, n_grow, n_neg, n_dneg);
		stats.solver_info.push_back(
			{{"type", "growth_dissipation"}, {"t", t}, {"step", step}, {"D", iD}, {"S", iS},
			 {"D_cumulative", cD}, {"S_cumulative", cS}, {"n_phi_negative", n_neg}, {"n_D_negative", n_dneg}});
	}

	void GrowthElasticVarForm::report_lag_residual(const int step, const double t, const Eigen::MatrixXd &solution)
	{
		// The threshold the converged state itself implies (route B at its own
		// C_check) against the one the increment used: the variant-(b) lag.
		Eigen::VectorXd full, ns, tt;
		evaluate_threshold_state(solution.col(0).head(displacement_ndof()), t, full, ns, tt);
		const Eigen::VectorXd phi_state = ns.cwiseMax(0.0) + (full - ns);
		lag_residual_ = phi_state - threshold_per_element_;
		const Eigen::VectorXd th_e = element_mean_growth(solution.col(0).tail(growth_ndof()));
		attainment_ = tt.cwiseQuotient(th_e) / growth_law_.theta_crit;

		std::vector<double> res, att;
		for (int e = 0; e < th_e.size(); ++e)
			if (th_e(e) > 1.0 + 1e-9 && ns(e) >= 0) // grown, not clamped at this state's C_check
			{
				res.push_back(std::abs(lag_residual_(e)));
				att.push_back(attainment_(e));
			}
		if (res.empty())
		{
			logger().info("growth step {} (t = {:g}): lag residual: no grown unclamped elements yet", step, t);
			return;
		}
		const auto q = [](std::vector<double> v, const double f) {
			const size_t k = std::min(v.size() - 1, size_t(f * double(v.size() - 1) + 0.5));
			std::nth_element(v.begin(), v.begin() + k, v.end());
			return v[k];
		};
		logger().info("growth step {} (t = {:g}): lag residual |phi(state) - phi used| median {:.4g} Pa (p95 {:.4g}); "
					  "attainment median {:.4f} (p5 {:.4f}, p95 {:.4f}) over {} grown unclamped elements",
					  step, t, q(res, 0.5), q(res, 0.95), q(att, 0.5), q(att, 0.05), q(att, 0.95), res.size());
		stats.solver_info.push_back(
			{{"type", "growth_lag_residual"}, {"t", t}, {"step", step}, {"abs_median", q(res, 0.5)},
			 {"abs_p95", q(res, 0.95)}, {"attainment_median", q(att, 0.5)}});
	}

	Eigen::VectorXd GrowthElasticVarForm::growth_multiplier(const Eigen::MatrixXd &solution) const
	{
		assert(stacked_form_);
		Eigen::VectorXd full_grad;
		stacked_form_->first_derivative(solution.col(0), full_grad);
		return full_grad.tail(growth_ndof());
	}

	void GrowthElasticVarForm::solve_with_current_pins(const int step, const double t, Eigen::MatrixXd &solution, const double grad_tol_factor)
	{
		build_forms(solution, t);
		make_nl_problem(solution, t);
		solve_nonlinear_step(step, solution, grad_tol_factor);
	}

	int GrowthElasticVarForm::solve_growth_increment(const int step, const double t, Eigen::MatrixXd &solution)
	{
		const int nu = displacement_ndof();
		const int ng = growth_ndof();

		// Release tolerance per DOF: the multiplier is an energy per unit growth
		// (Pa x volume), so it is scaled by the nodal volume and the threshold
		// magnitude (at least 1 Pa, so a zero threshold still has a scale).

		// Pass 1 is the elastic predictor: every growth DOF pinned at theta_n.
		// Once the set settles, one confirmation pass re-solves with the gradient
		// tolerance scaled by final_tol_factor (when < 1), so the multipliers the
		// set was decided on, and the reported state, are converged tighter than
		// the release tolerance they are compared with.
		std::vector<char> pinned(ng, 1);
		bool tight = false;
		for (int pass = 1; pass <= growth_law_.max_active_set_iterations; ++pass)
		{
			pinned_growth_nodes_.clear();
			for (int a = 0; a < ng; ++a)
			{
				if (pinned[a])
				{
					pinned_growth_nodes_.push_back(a);
					solution(nu + a, 0) = theta_n_(a); // pinned values exact: AL start error 0
				}
			}

			solve_with_current_pins(step, t, solution, tight ? growth_law_.final_tol_factor : 1.0);

			Eigen::VectorXd mult = growth_multiplier(solution);

			// Lagged shear supply re-evaluated on this pass's state. The dissipation
			// is linear in theta with gradient w, so the multiplier for the updated
			// threshold is mult + (w_new - w_old), exactly, without another solve.
			bool threshold_moved = false;
			if (growth_law_.shear_supply == "lagged"
				&& (growth_law_.shear_update == "pass" || (growth_law_.shear_update == "predictor" && pass == 1)))
			{
				const Eigen::VectorXd w_old = growth_node_weights_;
				const Eigen::VectorXd phi_old = threshold_per_element_;
				update_lagged_threshold(solution, t);
				mult += growth_node_weights_ - w_old;
				const double dphi = (threshold_per_element_ - phi_old).cwiseAbs().maxCoeff();
				// a changed threshold only moves the solution where theta is free
				const bool any_free = std::find(pinned.begin(), pinned.end(), 0) != pinned.end();
				threshold_moved = growth_law_.shear_update == "pass" && any_free && dphi > growth_law_.shear_update_tol;
				logger().info("growth step {} (t = {:g}) pass {}: shear supply updated, max |d phi| {:.4g} Pa{}",
							  step, t, pass, dphi, threshold_moved ? " (re-solving)" : "");
			}

			// scale from the threshold in use (it may have just been updated)
			const double force_scale = std::max(1.0, threshold_per_element_.cwiseAbs().maxCoeff());
			const Eigen::VectorXd release_tol = growth_law_.active_set_tol * force_scale * growth_node_volume_;
			int n_released = 0, n_pinned = 0, n_pinned_total = 0;
			std::vector<char> next = pinned;
			for (int a = 0; a < ng; ++a)
			{
				const double theta = solution(nu + a, 0);
				if (pinned[a])
				{
					// at the bound: the law wants growth when dPi/dtheta < 0
					if (mult(a) < -release_tol(a))
					{
						next[a] = 0;
						++n_released;
					}
				}
				else if (theta < theta_n_(a) - growth_law_.feasibility_tol)
				{
					next[a] = 1; // free DOF crossed the bound: irreversibility violated
					++n_pinned;
				}
				n_pinned_total += next[a];
			}

			growth_pinned_mask_ = pinned;
			logger().info("growth step {} (t = {:g}) active-set pass {}: {} released, {} pinned, {}/{} at theta_n",
						  step, t, pass, n_released, n_pinned, n_pinned_total, ng);
			stats.solver_info.push_back(
				{{"type", "growth_active_set"}, {"t", t}, {"step", step}, {"pass", pass},
				 {"released", n_released}, {"pinned", n_pinned}, {"n_at_bound", n_pinned_total}});

			if (n_released == 0 && n_pinned == 0 && !threshold_moved)
			{
				if (growth_law_.final_tol_factor < 1.0 && !tight)
				{
					tight = true;
					logger().info("growth step {} (t = {:g}): active set settled; confirmation pass with the gradient "
								  "tolerance x {:g}", step, t, growth_law_.final_tol_factor);
					continue;
				}
				log_kkt_residual(step, t, solution, mult, pinned);
				return pass;
			}
			pinned = next;
		}

		log_and_throw_error("growth step {} (t = {:g}): active set did not settle in {} passes "
							"(raise growth_law.max_active_set_iterations, or reduce the increment).",
							step, t, growth_law_.max_active_set_iterations);
		return -1;
	}

	void GrowthElasticVarForm::solve_growth_law(Eigen::MatrixXd &solution)
	{
		compute_growth_node_weights();

		const int nu = displacement_ndof();
		const int ng = growth_ndof();
		theta_n_ = solution.col(0).tail(ng);
		if (!(theta_n_.array() > 0).all())
			log_and_throw_error("growth_law: initial growth must be positive (min {}).", theta_n_.minCoeff());
		growth_activation_step_ = Eigen::VectorXi::Zero(ng);

		for (int step = 1; step <= growth_law_.n_steps; ++step)
		{
			const double t = growth_law_.t_end * double(step) / double(growth_law_.n_steps);
			if (growth_law_.shear_supply == "lagged" && growth_law_.shear_update == "increment")
				update_lagged_threshold(solution, growth_law_.t_end * double(step - 1) / double(growth_law_.n_steps));
			const int passes = solve_growth_increment(step, t, solution);

			const Eigen::VectorXd theta = solution.col(0).tail(ng);
			const Eigen::VectorXd dtheta = theta - theta_n_;
			const int n_active = int(std::count(growth_pinned_mask_.begin(), growth_pinned_mask_.end(), 0));
			logger().info("growth step {}/{} (t = {:g}): {} pass(es), {}/{} DOFs growing, "
						  "theta in [{:.6g}, {:.6g}], max increment {:.3e}, min increment {:.3e}",
						  step, growth_law_.n_steps, t, passes, n_active, ng,
						  theta.minCoeff(), theta.maxCoeff(), dtheta.maxCoeff(), dtheta.minCoeff());
			stats.solver_info.push_back(
				{{"type", "growth_step"}, {"t", t}, {"step", step}, {"passes", passes},
				 {"n_growing", n_active}, {"theta_min", theta.minCoeff()}, {"theta_max", theta.maxCoeff()}});

			accumulate_dissipation(step, t, dtheta);

			// Irreversibility holds by construction: pinned DOFs equal theta_n and
			// free DOFs passed the feasibility test. Clamp round-off only.
			theta_n_ = theta.cwiseMax(theta_n_);
			solution.col(0).segment(nu, ng) = theta_n_;

			// Activation increment (Q13(a)): the first increment in which a DOF
			// was free in the settled active set AND grew. Free-but-unmoved
			// DOFs (exactly at threshold) do not count.
			for (int a = 0; a < ng; ++a)
				if (growth_activation_step_(a) == 0 && !growth_pinned_mask_[a] && dtheta(a) > 0)
					growth_activation_step_(a) = step;

			if (growth_law_.lag_report && growth_law_.theta_crit > 0)
				report_lag_residual(step, t, solution);

			if (growth_law_.save_increments)
				save_growth_increment(step, t, solution);
		}
	}

	void GrowthElasticVarForm::save_growth_increment(const int step, const double t, const Eigen::MatrixXd &solution)
	{
		const io::OutputSpace space = output_space();
		if (!space.mesh)
			return;

		// output_fields falls back to converged_solution_ for non-stacked
		// inputs; keep it current so every export path sees this increment.
		converged_solution_ = solution;

		ensure_output_sampler();
		const auto opts = export_options(space);
		const double dt = growth_law_.t_end / double(growth_law_.n_steps);
		const std::string name = fmt::format("growth_step_{:d}.vtu", step);
		output_geometry_.save_vtu(resolve_output_path(name), space,
								  output_field_function(solution, opts), t, dt, opts);

		// The PVD is rewritten each increment, so an aborted ramp still leaves
		// a readable series of the increments that converged.
		std::ofstream pvd(resolve_output_path("growth_steps.pvd"));
		pvd << "<?xml version=\"1.0\"?>\n"
			<< "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">\n"
			<< "  <Collection>\n";
		for (int k = 1; k <= step; ++k)
			pvd << fmt::format("    <DataSet timestep=\"{:.17g}\" group=\"\" part=\"0\" file=\"growth_step_{:d}.vtu\"/>\n",
							   growth_law_.t_end * double(k) / double(growth_law_.n_steps), k);
		pvd << "  </Collection>\n</VTKFile>\n";
		logger().info("growth step {}: wrote {}", step, resolve_output_path(name));
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
			Eigen::MatrixXd reaction_nodal = full_grad.tail(growth_ndof());
			// Solved growth adds the dissipation to stacked_form_; growth_reaction
			// keeps its V3 meaning, the pure energy gradient (assembled -vn f_I).
			if (dissipation_form_)
				reaction_nodal -= dissipation_form_->weights();

			Eigen::MatrixXd values;
			if (sample_growth_scalar(reaction_nodal, values))
				fields.push_back({"growth_reaction", values, io::OutputField::Association::Point});

			// The KKT residual of the active set, dPi_inc/dtheta: ~0 where theta
			// grows, >= 0 where pinned at theta_n (the multiplier of the bound).
			if (dissipation_form_ && options.export_field("growth_multiplier"))
			{
				const Eigen::MatrixXd mult = full_grad.tail(growth_ndof());
				if (sample_growth_scalar(mult, values))
					fields.push_back({"growth_multiplier", values, io::OutputField::Association::Point});
			}
		}

		if (growth_law_.enabled && int(growth_pinned_mask_.size()) == growth_ndof()
			&& options.export_field("growth_pinned"))
		{
			Eigen::MatrixXd mask(growth_ndof(), 1);
			for (int a = 0; a < growth_ndof(); ++a)
				mask(a) = growth_pinned_mask_[a] ? 1.0 : 0.0;
			Eigen::MatrixXd values;
			if (sample_growth_scalar(mask, values))
				fields.push_back({"growth_pinned", values, io::OutputField::Association::Point});
		}

		if (growth_law_.enabled && has_element_samples && threshold_per_element_.size() == mesh_->n_elements())
		{
			const auto per_element = [&](const Eigen::VectorXd &data, const std::string &name) {
				if (!options.export_field(name))
					return;
				Eigen::MatrixXd values = Eigen::MatrixXd::Zero(output_rows, 1);
				for (int i = 0; i < sample.element_ids.size() && i < output_rows; ++i)
					if (sample.element_ids(i) >= 0)
						values(i) = data(sample.element_ids(i));
				fields.push_back({name, values, io::OutputField::Association::Point});
			};
			per_element(threshold_per_element_, "growth_threshold");
			if (shear_supply_.size() == mesh_->n_elements())
				per_element(shear_supply_, "growth_shear_supply");
			if (threshold_parallel_.size() == mesh_->n_elements())
				per_element(threshold_parallel_, "growth_threshold_parallel");
			if (threshold_clamped_.size() == mesh_->n_elements())
				per_element(threshold_clamped_, "growth_clamped");
			if (lag_residual_.size() == mesh_->n_elements())
				per_element(lag_residual_, "growth_lag_residual");
			if (attainment_.size() == mesh_->n_elements())
				per_element(attainment_, "growth_attainment");
			if (dissipation_.size() == mesh_->n_elements())
				per_element(dissipation_, "growth_dissipation");
			if (shear_work_.size() == mesh_->n_elements())
				per_element(shear_work_, "growth_shear_supply_work");
		}

		if (growth_law_.enabled && growth_activation_step_.size() == growth_ndof()
			&& options.export_field("growth_activation_step"))
		{
			const Eigen::MatrixXd steps = growth_activation_step_.cast<double>();
			Eigen::MatrixXd values;
			if (sample_growth_scalar(steps, values))
				fields.push_back({"growth_activation_step", values, io::OutputField::Association::Point});
		}

		if (growthelastic_assembler_)
			append_material_fields(*growthelastic_assembler_);

		return fields;
	}
} // namespace polyfem::varform
