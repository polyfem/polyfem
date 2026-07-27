#include <polyfem/varforms/diff/DifferentiableNonlinearElasticVarForm.hpp>

#include <polyfem/assembler/MacroStrain.hpp>
#include <polyfem/io/MatrixIO.hpp>
#include <polyfem/io/SolverCSVWriter.hpp>
#include <polyfem/solver/ALSolver.hpp>
#include <polyfem/solver/NLProblem.hpp>
#include <polyfem/time_integrator/ImplicitTimeIntegrator.hpp>
#include <polyfem/utils/Jacobian.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/Timer.hpp>

#include <igl/Timer.h>

#include <polysolve/linear/Solver.hpp>
#include <polysolve/nonlinear/Solver.hpp>

namespace polyfem::varform
{
	void DifferentiableNonlinearElasticVarForm::solve(
		Eigen::MatrixXd &solution,
		const InitialConditionOverride *initial_condition_override,
		const ForwardStepCallback &post_step,
		const bool differentiable)
	{
		differentiable_mode_ = differentiable;
		NonlinearElasticVarForm::solve(solution, initial_condition_override, post_step);
	}

	void DifferentiableNonlinearElasticVarForm::prepare()
	{
		NonlinearElasticVarForm::prepare();
	}

	void DifferentiableNonlinearElasticVarForm::save_vtu(
		const std::string &path,
		const Eigen::MatrixXd &solution,
		const double time,
		const double dt) const
	{
		const io::OutputSpace space = NonlinearElasticVarForm::output_space();
		NonlinearElasticVarForm::ensure_output_sampler();
		const auto opts = NonlinearElasticVarForm::export_options(space);
		output_geometry_.save_vtu(
			path, space, NonlinearElasticVarForm::output_field_function(solution, opts), time, dt, opts);
	}

	json &DifferentiableNonlinearElasticVarForm::get_args() { return args; }
	const json &DifferentiableNonlinearElasticVarForm::get_args() const { return args; }

	const mesh::Mesh &DifferentiableNonlinearElasticVarForm::get_mesh() const
	{
		assert(mesh_ && "The mesh must be loaded before it is accessed");
		return *mesh_;
	}

	assembler::Problem &DifferentiableNonlinearElasticVarForm::get_problem()
	{
		assert(problem && "The problem must be initialized before it is accessed");
		return *problem;
	}

	const assembler::Problem &DifferentiableNonlinearElasticVarForm::get_problem() const
	{
		assert(problem && "The problem must be initialized before it is accessed");
		return *problem;
	}

	const std::string &DifferentiableNonlinearElasticVarForm::get_root_path() const { return root_path; }

	std::string DifferentiableNonlinearElasticVarForm::input_path(const std::string &path, const bool only_if_exists) const
	{
		return NonlinearElasticVarForm::resolve_input_path(path, only_if_exists);
	}

	std::string DifferentiableNonlinearElasticVarForm::output_file_path(const std::string &path) const
	{
		return NonlinearElasticVarForm::resolve_output_path(path);
	}

	const Units &DifferentiableNonlinearElasticVarForm::get_units() const { return units; }
	bool DifferentiableNonlinearElasticVarForm::is_contact_enabled() const { return NonlinearElasticVarForm::is_contact_enabled(); }

	const FESpace &DifferentiableNonlinearElasticVarForm::primary_space() const { return space_; }
	const VarFormBoundaryState &DifferentiableNonlinearElasticVarForm::boundary_state() const { return boundary_; }

	const assembler::Assembler &DifferentiableNonlinearElasticVarForm::primary_assembler() const
	{
		assert(primary_assembler_ && "The primary assembler must be initialized before it is accessed");
		return *primary_assembler_;
	}

	const assembler::Mass &DifferentiableNonlinearElasticVarForm::mass_assembler() const
	{
		assert(mass_assembler_ && "The mass assembler must be initialized before it is accessed");
		return *mass_assembler_;
	}

	const assembler::AssemblyValsCache &DifferentiableNonlinearElasticVarForm::assembly_cache() const { return ass_vals_cache_; }
	const assembler::AssemblyValsCache &DifferentiableNonlinearElasticVarForm::mass_assembly_cache() const { return mass_ass_vals_cache_; }
	const StiffnessMatrix &DifferentiableNonlinearElasticVarForm::mass_matrix() const { return mass_; }
	solver::SolveData *DifferentiableNonlinearElasticVarForm::solve_data() { return &solve_data_; }
	const solver::SolveData *DifferentiableNonlinearElasticVarForm::solve_data() const { return &solve_data_; }
	const ipc::CollisionMesh &DifferentiableNonlinearElasticVarForm::collision_mesh() const { return collision_mesh_; }
	const mesh::Obstacle &DifferentiableNonlinearElasticVarForm::get_obstacle() const { return obstacle; }

	const assembler::ViscousDamping *DifferentiableNonlinearElasticVarForm::damping_assembler() const
	{
		return damping_assembler_.get();
	}

	const assembler::ViscousDampingPrev *DifferentiableNonlinearElasticVarForm::damping_prev_assembler() const
	{
		return damping_prev_assembler_.get();
	}

	void DifferentiableNonlinearElasticVarForm::initial_solution(
		Eigen::MatrixXd &solution,
		const InitialConditionOverride *override) const
	{
		NonlinearElasticVarForm::initial_solution(solution, override);
	}

	void DifferentiableNonlinearElasticVarForm::initial_velocity(
		Eigen::MatrixXd &velocity,
		const InitialConditionOverride *override) const
	{
		NonlinearElasticVarForm::initial_velocity(velocity, override);
	}

	void DifferentiableNonlinearElasticVarForm::initial_acceleration(
		Eigen::MatrixXd &acceleration,
		const InitialConditionOverride *override) const
	{
		NonlinearElasticVarForm::initial_acceleration(acceleration, override);
	}

	mesh::Mesh &DifferentiableNonlinearElasticVarForm::mutable_mesh()
	{
		assert(mesh_ && "Vertex positions can only be updated after loading a mesh");
		return *mesh_;
	}

	void DifferentiableNonlinearElasticVarForm::invalidate_after_geometry_update()
	{
		forms.clear();
		solve_data_ = solver::SolveData();
		elasticity_pressure_assembler = nullptr;
		damping_assembler_ = nullptr;
		damping_prev_assembler_ = nullptr;
		rhs_assembler_ = nullptr;
		prepared_ = false;
		output_sampler_initialized_ = false;
	}

	void DifferentiableNonlinearElasticVarForm::invalidate_after_parameter_update()
	{
		forms.clear();
		solve_data_ = solver::SolveData();
		elasticity_pressure_assembler = nullptr;
		damping_assembler_ = nullptr;
		damping_prev_assembler_ = nullptr;
		rhs_assembler_ = nullptr;
		prepared_ = false;
	}

	QuadratureOrders DifferentiableNonlinearElasticVarForm::boundary_samples(
		const int discr_order,
		const int geometry_discr_order) const
	{
		return NonlinearElasticVarForm::n_boundary_samples(discr_order, geometry_discr_order);
	}

	// The differentiable path matches NonlinearElasticVarForm::init_forms except
	// that it enables IPC shape derivatives when constructing the contact forms.
	void DifferentiableNonlinearElasticVarForm::init_forms(
		const json &args,
		const int dim,
		Eigen::MatrixXd &solution,
		const double time)
	{
		if (!differentiable_mode_)
		{
			NonlinearElasticVarForm::init_forms(args, dim, solution, time);
			return;
		}

		damping_assembler_ = std::make_shared<assembler::ViscousDamping>();
		set_materials(*damping_assembler_, mesh_->dimension());

		elasticity_pressure_assembler = build_pressure_assembler();

		damping_prev_assembler_ = std::make_shared<assembler::ViscousDampingPrev>();
		set_materials(*damping_prev_assembler_, mesh_->dimension());

		const solver::ElementInversionCheck check_inversion = args["solver"]["advanced"]["check_inversion"];

		forms = solve_data_.init_forms(
			units,
			dim, time, space_.space_in_node_to_node,
			space_.n_bases, *space_.bases, space_.geometry_basis_list(), *primary_assembler_, ass_vals_cache_, mass_ass_vals_cache_, args["solver"]["advanced"]["jacobian_threshold"], check_inversion,
			args["solver"]["advanced"]["conservative_max_iter"],
			0, boundary_.boundary_nodes, boundary_.local_boundary,
			boundary_.local_neumann_boundary,
			elastic_boundary_samples(), rhs_, solution, mass_assembler_->density(),
			boundary_.local_pressure_boundary, boundary_.local_pressure_cavity, elasticity_pressure_assembler,
			args.value("/time/quasistatic"_json_pointer, true), mass_,
			damping_assembler_->is_valid() ? damping_assembler_ : nullptr,
			args["solver"]["advanced"]["lagged_regularization_weight"],
			args["solver"]["advanced"]["lagged_regularization_iterations"],
			obstacle.ndof(), args["constraints"]["hard"], args["constraints"]["soft"],
			args["contact"]["enabled"], collision_mesh_, args["contact"]["dhat"],
			avg_mass_, args["contact"]["use_convergent_formulation"] ? bool(args["contact"]["use_area_weighting"]) : false,
			args["contact"]["use_convergent_formulation"] ? bool(args["contact"]["use_improved_max_operator"]) : false,
			args["contact"]["use_convergent_formulation"] ? bool(args["contact"]["use_physical_barrier"]) : false,
			args["solver"]["contact"]["barrier_stiffness"],
			args["solver"]["contact"]["initial_barrier_stiffness"],
			args["solver"]["contact"]["CCD"]["broad_phase"],
			args["solver"]["contact"]["CCD"]["tolerance"],
			args["solver"]["contact"]["CCD"]["max_iterations"],
			true,
			args["contact"]["use_gcp_formulation"],
			args["contact"]["alpha_t"],
			args["contact"]["alpha_n"],
			args["contact"]["use_adaptive_dhat"],
			args["contact"]["min_distance_ratio"],
			args["contact"]["adhesion"]["adhesion_enabled"],
			args["contact"]["adhesion"]["dhat_p"],
			args["contact"]["adhesion"]["dhat_a"],
			args["contact"]["adhesion"]["adhesion_strength"],
			args["contact"]["adhesion"]["tangential_adhesion_coefficient"],
			args["contact"]["adhesion"]["epsa"],
			args["solver"]["contact"]["tangential_adhesion_iterations"],
			assembler::MacroStrainValue(),
			false, Eigen::VectorXi(), nullptr,
			args["contact"]["friction_coefficient"],
			args["contact"]["epsv"],
			args["solver"]["contact"]["friction_iterations"],
			args["solver"]["rayleigh_damping"]);

		for (const auto &form : forms)
			form->set_output_dir(output_path);

		if (solve_data_.contact_form != nullptr)
			solve_data_.contact_form->save_ccd_debug_meshes = args["output"]["advanced"]["save_ccd_debug_meshes"];
	}

	// The differentiable path performs the initial nonlinear solve but skips the
	// subsequent lagging iterations. The adjoint uses the forms frozen at that
	// solution instead of differentiating through the lagging update loop.
	void DifferentiableNonlinearElasticVarForm::solve_tensor_nonlinear(
		const int step,
		Eigen::MatrixXd &solution,
		const bool init_lagging)
	{
		if (!differentiable_mode_)
		{
			NonlinearElasticVarForm::solve_tensor_nonlinear(step, solution, init_lagging);
			return;
		}

		assert(solve_data_.nl_problem != nullptr && "Nonlinear forms must initialize the nonlinear problem before solving");
		solver::NLProblem &nl_problem = *solve_data_.nl_problem;
		assert(solution.size() == rhs_.size());

		if (nl_problem.uses_lagging())
		{
			if (init_lagging)
			{
				POLYFEM_SCOPED_TIMER("Initializing lagging");
				nl_problem.init_lagging(solution);
			}
			logger().info("Lagging iteration 1:");
		}

		save_subsolve(0, step, solution);

		std::shared_ptr<polysolve::nonlinear::Solver> nl_solver =
			polysolve::nonlinear::Solver::create(
				args["solver"]["augmented_lagrangian"]["nonlinear"],
				args["solver"]["linear"], units.characteristic_length(), logger());

		solver::ALSolver al_solver(
			solve_data_.al_form,
			args["solver"]["augmented_lagrangian"]["initial_weight"],
			args["solver"]["augmented_lagrangian"]["scaling"],
			args["solver"]["augmented_lagrangian"]["max_weight"],
			args["solver"]["augmented_lagrangian"]["eta"],
			[&](const Eigen::VectorXd &) {
				solve_data_.update_barrier_stiffness(solution);
			});

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
			nl_problem, solution,
			args["solver"]["augmented_lagrangian"]["nonlinear"],
			args["solver"]["linear"], units.characteristic_length());

		al_solver.solve_reduced(
			nl_problem, solution,
			args["solver"]["nonlinear"],
			args["solver"]["linear"], units.characteristic_length());

		if (args["space"]["advanced"]["count_flipped_els_continuous"])
		{
			const auto invalid = utils::count_invalid(
				mesh_->dimension(), space_.basis_list(), space_.geometry_basis_list(), solution);
			logger().debug("Flipped elements (cnt {}) : {}", invalid.size(), invalid);
		}
	}

	// Same as NonlinearElasticStaticVarForm
	void DifferentiableNonlinearElasticStaticVarForm::solve_problem(
		Eigen::MatrixXd &solution,
		const InitialConditionOverride *initial_condition_override,
		const ForwardStepCallback &post_step)
	{
		assert(
			(!initial_condition_override
			 || (initial_condition_override->velocity.size() == 0
				 && initial_condition_override->acceleration.size() == 0))
			&& "Static elasticity does not accept initial velocity or acceleration overrides");

		stats.spectrum.setZero();

		igl::Timer timer;
		timer.start();
		logger().info("Solving {}", primary_assembler_->name());

		{
			POLYFEM_SCOPED_TIMER("Setup RHS");

			if (initial_condition_override && initial_condition_override->solution.size() != 0)
				initial_elastic_solution(solution, initial_condition_override);
			else if (solution.size() <= 0)
				initial_elastic_solution(solution, initial_condition_override);

			if (initial_condition_override && initial_condition_override->solution.size() != 0)
				assert(solution.cols() == 1 && "Static initial solution override must have exactly one column");
			else if (solution.cols() != 1)
				log_and_throw_error("Static elasticity requires exactly one initial solution column.");
		}

		init_solve(solution, 1.0, initial_condition_override);
		solve_tensor_nonlinear(0, solution, true);
		if (post_step)
			post_step(0, solution);

		const std::string state_path = resolve_output_path(args["output"]["data"]["state"]);
		if (!state_path.empty())
			io::write_matrix(state_path, "u", solution);

		timer.stop();
		timings.solving_time = timer.getElapsedTime();
		logger().info(" took {}s", timings.solving_time);
	}

	// Same as NonlinearElasticTransientVarForm.
	void DifferentiableNonlinearElasticTransientVarForm::solve_problem(
		Eigen::MatrixXd &solution,
		const InitialConditionOverride *initial_condition_override,
		const ForwardStepCallback &post_step)
	{
		const bool save_stats = args["output"]["stats"];
		stats.spectrum.setZero();

		igl::Timer timer;
		timer.start();
		logger().info("Solving {}", primary_assembler_->name());

		{
			POLYFEM_SCOPED_TIMER("Setup RHS");

			if (initial_condition_override && initial_condition_override->solution.size() != 0)
				initial_elastic_solution(solution, initial_condition_override);
			else if (solution.size() <= 0)
				initial_elastic_solution(solution, initial_condition_override);

			if (solution.cols() > 1)
				solution.conservativeResize(Eigen::NoChange, 1);
		}

		init_solve(solution, t0 + dt, initial_condition_override);
		if (post_step)
			post_step(0, solution);

		int save_i = 0;
		std::unique_ptr<io::EnergyCSVWriter> energy_csv;
		std::unique_ptr<io::RuntimeStatsCSVWriter> stats_csv;

		if (save_stats)
		{
			logger().debug(
				"Saving nl stats to {} and {}",
				resolve_output_path("energy.csv"), resolve_output_path("stats.csv"));
			energy_csv = std::make_unique<io::EnergyCSVWriter>(resolve_output_path("energy.csv"), solve_data_);
			const io::OutputSpace space = output_space();
			stats_csv = std::make_unique<io::RuntimeStatsCSVWriter>(
				resolve_output_path("stats.csv"),
				space_.n_bases,
				space.mesh ? space.mesh->n_elements() : 0,
				t0, dt);
		}

		if (energy_csv)
			energy_csv->write(save_i, solution);
		save_timestep(t0, 0, t0, dt, solution);
		++save_i;

		for (int t = 1; t <= time_steps; ++t)
		{
			double forward_solve_time = 0;
			const double remeshing_time = 0;
			const double global_relaxation_time = 0;

			{
				POLYFEM_SCOPED_TIMER(forward_solve_time);
				solve_tensor_nonlinear(t, solution, true);
			}
			if (post_step)
				post_step(t, solution);

			if (energy_csv)
				energy_csv->write(save_i, solution);
			save_timestep(t0 + dt * t, t, t0, dt, solution);
			++save_i;

			{
				POLYFEM_SCOPED_TIMER("Update quantities");
				solve_data_.time_integrator->update_quantities(solution);
				solve_data_.nl_problem->update_quantities(t0 + (t + 1) * dt, solution);
				solve_data_.update_dt();
				solve_data_.update_barrier_stiffness(solution);
			}

			logger().info("{}/{}  t={}", t, time_steps, t0 + dt * t);
			notify_time_step(t, time_steps, t0, dt);
			save_elastic_step_state(t0, dt, t, solve_data_.time_integrator.get());
			if (stats_csv)
				stats_csv->write(t, forward_solve_time, remeshing_time, global_relaxation_time);
		}

		timer.stop();
		timings.solving_time = timer.getElapsedTime();
		logger().info(" took {}s", timings.solving_time);
	}
} // namespace polyfem::varform
