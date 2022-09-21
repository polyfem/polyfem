#include <polyfem/State.hpp>

#include <polyfem/solver/forms/BodyForm.hpp>
#include <polyfem/solver/forms/ContactForm.hpp>
#include <polyfem/solver/forms/ElasticForm.hpp>
#include <polyfem/solver/forms/FrictionForm.hpp>
#include <polyfem/solver/forms/DampingForm.hpp>
#include <polyfem/solver/forms/InertiaForm.hpp>
#include <polyfem/solver/forms/LaggedRegForm.hpp>
#include <polyfem/solver/forms/ALForm.hpp>

#include <polyfem/solver/NonlinearSolver.hpp>
#include <polyfem/solver/LBFGSSolver.hpp>
#include <polyfem/solver/SparseNewtonDescentSolver.hpp>
#include <polyfem/solver/NLProblem.hpp>
#include <polyfem/solver/ALSolver.hpp>
#include <polyfem/io/OBJWriter.hpp>
#include <polyfem/utils/MatrixUtils.hpp>
#include <polyfem/utils/Timer.hpp>
#include <polyfem/utils/JSONUtils.hpp>

#include <ipc/ipc.hpp>

// map BroadPhaseMethod values to JSON as strings
namespace ipc
{
	NLOHMANN_JSON_SERIALIZE_ENUM(
		ipc::BroadPhaseMethod,
		{{ipc::BroadPhaseMethod::HASH_GRID, "hash_grid"}, // also default
		 {ipc::BroadPhaseMethod::HASH_GRID, "HG"},
		 {ipc::BroadPhaseMethod::BRUTE_FORCE, "brute_force"},
		 {ipc::BroadPhaseMethod::BRUTE_FORCE, "BF"},
		 {ipc::BroadPhaseMethod::SPATIAL_HASH, "spatial_hash"},
		 {ipc::BroadPhaseMethod::SPATIAL_HASH, "SH"},
		 {ipc::BroadPhaseMethod::SWEEP_AND_TINIEST_QUEUE, "sweep_and_tiniest_queue"},
		 {ipc::BroadPhaseMethod::SWEEP_AND_TINIEST_QUEUE, "STQ"},
		 {ipc::BroadPhaseMethod::SWEEP_AND_TINIEST_QUEUE_GPU, "sweep_and_tiniest_queue_gpu"},
		 {ipc::BroadPhaseMethod::SWEEP_AND_TINIEST_QUEUE_GPU, "STQ_GPU"}})
} // namespace ipc

namespace polyfem
{
	using namespace solver;
	using namespace io;
	using namespace utils;

	void SolveData::updated_barrier_stiffness(const Eigen::VectorXd &x)
	{
		// TODO: missing use_adaptive_barrier_stiffness_ if (use_adaptive_barrier_stiffness_ && is_time_dependent_)
		// if (inertia_form == nullptr)
		// 	return;
		if (contact_form == nullptr)
			return;

		if (!contact_form->use_adaptive_barrier_stiffness())
			return;

		Eigen::VectorXd grad_energy(x.size(), 1);
		grad_energy.setZero();

		elastic_form->first_derivative(x, grad_energy);

		if (inertia_form)
		{
			Eigen::VectorXd grad_inertia(x.size());
			inertia_form->first_derivative(x, grad_inertia);
			grad_energy += grad_inertia;
		}

		Eigen::VectorXd body_energy(x.size());
		body_form->first_derivative(x, body_energy);
		grad_energy += body_energy;

		contact_form->update_barrier_stiffness(x, grad_energy);
	}

	void SolveData::update_dt()
	{
		if (inertia_form)
		{
			elastic_form->set_weight(time_integrator->acceleration_scaling());
			body_form->set_weight(time_integrator->acceleration_scaling());
			damping_form->set_weight(time_integrator->acceleration_scaling());

			// TODO: Determine if friction should be scaled by h²
			// if (friction_form)
			// 	friction_form->set_weight(time_integrator->acceleration_scaling());
		}
	}

	template <typename ProblemType>
	std::shared_ptr<cppoptlib::NonlinearSolver<ProblemType>> State::make_nl_solver() const
	{
		const std::string name = args["solver"]["nonlinear"]["solver"];
		if (name == "newton" || name == "Newton")
		{
			return std::make_shared<cppoptlib::SparseNewtonDescentSolver<ProblemType>>(
				args["solver"]["nonlinear"], args["solver"]["linear"]);
		}
		else if (name == "lbfgs" || name == "LBFGS" || name == "L-BFGS")
		{
			return std::make_shared<cppoptlib::LBFGSSolver<ProblemType>>(args["solver"]["nonlinear"]);
		}
		else
		{
			throw std::invalid_argument(fmt::format("invalid nonlinear solver type: {}", name));
		}
	}

	void State::solve_transient_tensor_nonlinear(const int time_steps, const double t0, const double dt)
	{
		init_nonlinear_tensor_solve(t0 + dt);

		save_timestep(t0, 0, t0, dt);

		if (args["differentiable"])
			cache_transient_adjoint_quantities(0);

		for (int t = 1; t <= time_steps; ++t)
		{
			solve_tensor_nonlinear(t);

			if (args["differentiable"])
				cache_transient_adjoint_quantities(t);

			{
				POLYFEM_SCOPED_TIMER("Update quantities");

				solve_data.time_integrator->update_quantities(sol);

				solve_data.nl_problem->update_quantities(t0 + (t + 1) * dt, sol);

				solve_data.update_dt();
				solve_data.updated_barrier_stiffness(sol);
			}

			save_timestep(t0 + dt * t, t, t0, dt);

			logger().info("{}/{}  t={}", t, time_steps, t0 + dt * t);
		}

		solve_data.time_integrator->save_raw(
			resolve_output_path(args["output"]["data"]["u_path"]),
			resolve_output_path(args["output"]["data"]["v_path"]),
			resolve_output_path(args["output"]["data"]["a_path"]));
	}

	void State::init_nonlinear_tensor_solve(const double t)
	{
		assert(!assembler.is_linear(formulation()) || is_contact_enabled()); // non-linear
		assert(!problem->is_scalar());                                       // tensor
		assert(!assembler.is_mixed(formulation()));

		const auto &cur_sol = (pre_sol.size() == sol.size()) ? pre_sol : sol;
		///////////////////////////////////////////////////////////////////////
		// Check for initial intersections
		if (is_contact_enabled())
		{
			POLYFEM_SCOPED_TIMER("Check for initial intersections");

			Eigen::MatrixXd displaced = boundary_nodes_pos + unflatten(cur_sol, mesh->dimension());

			if (ipc::has_intersections(collision_mesh, collision_mesh.vertices(displaced)))
			{
				OBJWriter::write(
					resolve_output_path("intersection.obj"), collision_mesh.vertices(displaced),
					collision_mesh.edges(), collision_mesh.faces());
				log_and_throw_error("Unable to solve, initial solution has intersections!");
			}
		}

		assert(solve_data.rhs_assembler != nullptr);

		std::vector<std::shared_ptr<Form>> forms;
		solve_data.elastic_form = std::make_shared<ElasticForm>(*this);
		forms.push_back(solve_data.elastic_form);
		solve_data.body_form = std::make_shared<BodyForm>(*this, *solve_data.rhs_assembler, /*apply_DBC=*/true);
		solve_data.body_form->update_quantities(t, cur_sol);
		forms.push_back(solve_data.body_form);

		solve_data.inertia_form = nullptr;
		solve_data.damping_form = nullptr;
		if (problem->is_time_dependent())
		{
			solve_data.time_integrator = time_integrator::ImplicitTimeIntegrator::construct_time_integrator(args["time"]["integrator"]);
			solve_data.time_integrator->set_parameters(args["time"]);
			solve_data.inertia_form = std::make_shared<InertiaForm>(mass, *solve_data.time_integrator);
			forms.push_back(solve_data.inertia_form);
			solve_data.damping_form = std::make_shared<DampingForm>(*this, args["time"]["dt"]);
			forms.push_back(solve_data.damping_form);
		}
		else
		{
			// FIXME:
			// const double lagged_regularization_weight = args["solver"]["advanced"]["lagged_regularization_weight"];
			// if (lagged_regularization_weight > 0)
			// {
			// 	forms.push_back(std::make_shared<LaggedRegForm>(lagged_damping_weight));
			// 	forms.back()->set_weight(lagged_regularization_weight);
			// }
		}

		solve_data.al_form = std::make_shared<ALForm>(*this, *solve_data.rhs_assembler, t);
		forms.push_back(solve_data.al_form);

		solve_data.contact_form = nullptr;
		solve_data.friction_form = nullptr;
		if (args["contact"]["enabled"])
		{

			const bool use_adaptive_barrier_stiffness = !args["solver"]["contact"]["barrier_stiffness"].is_number();

			solve_data.contact_form = std::make_shared<ContactForm>(
				*this,
				args["contact"]["dhat"],
				use_adaptive_barrier_stiffness,
				/*is_time_dependent=*/solve_data.time_integrator != nullptr,
				args["solver"]["contact"]["CCD"]["broad_phase"],
				args["solver"]["contact"]["CCD"]["tolerance"],
				args["solver"]["contact"]["CCD"]["max_iterations"]);

			if (use_adaptive_barrier_stiffness)
			{
				solve_data.contact_form->set_weight(1);
				logger().debug("Using adaptive barrier stiffness");
			}
			else
			{
				solve_data.contact_form->set_weight(args["solver"]["contact"]["barrier_stiffness"]);
				logger().debug("Using fixed barrier stiffness of {}", solve_data.contact_form->barrier_stiffness());
			}

			forms.push_back(solve_data.contact_form);

			// ----------------------------------------------------------------

			if (args["contact"]["friction_coefficient"].get<double>() != 0)
			{
				solve_data.friction_form = std::make_shared<FrictionForm>(
					*this,
					args["contact"]["epsv"],
					args["contact"]["friction_coefficient"],
					args["contact"]["dhat"],
					args["solver"]["contact"]["CCD"]["broad_phase"],
					args.value("/time/dt"_json_pointer, 1.0), // dt=1.0 if static
					*solve_data.contact_form);
				forms.push_back(solve_data.friction_form);
			}
		}

		///////////////////////////////////////////////////////////////////////
		// Initialize nonlinear problems
		solve_data.nl_problem = std::make_shared<NLProblem>(*this, *solve_data.rhs_assembler, t, forms);

		///////////////////////////////////////////////////////////////////////
		// Initialize time integrator
		if (problem->is_time_dependent())
		{
			POLYFEM_SCOPED_TIMER("Initialize time integrator");

			Eigen::MatrixXd velocity, acceleration;
			initial_velocity(velocity);
			assert(velocity.size() == sol.size());
			initial_velocity(acceleration);
			assert(acceleration.size() == sol.size());

			if (args["differentiable"])
			{
				if (initial_sol_update.size() > 0)
					sol = initial_sol_update;
				else
					initial_sol_update = sol;
				if (initial_vel_update.size() > 0)
					velocity = initial_vel_update;
				else
					initial_vel_update = velocity;
			}

			initial_velocity_cache = velocity;

			const double dt = args["time"]["dt"];
			solve_data.time_integrator->init(sol, velocity, acceleration, dt);
		}
		solve_data.update_dt();

		///////////////////////////////////////////////////////////////////////

		solver_info = json::array();
	}

	void State::solve_tensor_nonlinear(const int t)
	{

		assert(solve_data.nl_problem != nullptr);
		NLProblem &nl_problem = *(solve_data.nl_problem);

		assert(sol.size() == rhs.size());

		{
			POLYFEM_SCOPED_TIMER("Initializing lagging");
			nl_problem.init_lagging(sol);
		}

		if (pre_sol.rows() == sol.rows())
		{
			logger().debug("Use better initial guess...");
			sol = pre_sol;
		}

		const int friction_iterations = args["solver"]["contact"]["friction_iterations"];
		assert(friction_iterations >= 0);
		if (friction_iterations > 0)
			logger().debug("Lagging iteration 1");

		const double lagging_tol = args["solver"]["contact"].value("friction_convergence_tol", 1e-2);

		// Disable damping for the final lagged iteration
		// TODO: fix me lagged regularization
		// if (friction_iterations <= 1)
		// {
		// 	nl_problem.lagged_regularization_weight() = 0;
		// 	alnl_problem.lagged_regularization_weight() = 0;
		// }

		// ---------------------------------------------------------------------

		// Save the subsolve sequence for debugging
		int subsolve_count = 0;
		save_subsolve(subsolve_count, t);

		// ---------------------------------------------------------------------

		std::shared_ptr<cppoptlib::NonlinearSolver<NLProblem>> nl_solver = make_nl_solver<NLProblem>();

		ALSolver al_solver(
			nl_solver, solve_data.al_form,
			args["solver"]["augmented_lagrangian"]["initial_weight"],
			args["solver"]["augmented_lagrangian"]["max_weight"],
			[&](const Eigen::VectorXd &x) {
				this->solve_data.updated_barrier_stiffness(sol);
			});

		al_solver.post_subsolve = [&](const double al_weight) {
			json info;
			nl_solver->get_info(info);
			solver_info.push_back(
				{{"type", al_weight > 0 ? "al" : "rc"},
				 {"t", t}, // TODO: null if static?
				 {"info", info}});
			if (al_weight > 0)
				solver_info.back()["weight"] = al_weight;
			
			n_linear_solves += info["iterations"].get<int>();
			n_nonlinear_solves += 1;

			this->save_subsolve(++subsolve_count, t);
		};

		al_solver.solve(nl_problem, sol, args["solver"]["augmented_lagrangian"]["force"]);

		// ---------------------------------------------------------------------

		Eigen::VectorXd tmp_sol = nl_problem.full_to_reduced(sol);

		// TODO: fix this lagged damping
		// nl_problem.lagged_regularization_weight() = 0;

		if (!args["differentiable"])
		{
			// Lagging loop (start at 1 because we already did an iteration above)
			int lag_i;
			nl_problem.update_lagging(tmp_sol);
			Eigen::VectorXd tmp_grad;
			nl_problem.gradient(tmp_sol, tmp_grad);
			bool lagging_converged = tmp_grad.norm() <= lagging_tol;
			for (lag_i = 1; !lagging_converged && lag_i < friction_iterations; lag_i++)
			{
				logger().debug("Lagging iteration {:d}", lag_i + 1);
				nl_problem.init(sol);
				solve_data.updated_barrier_stiffness(sol);
				// Disable damping for the final lagged iteration
				// TODO: fix this lagged damping
				// if (lag_i == friction_iterations - 1)
				// 	nl_problem.lagged_regularization_weight() = 0;
				nl_solver->minimize(nl_problem, tmp_sol);

				json info;
				nl_solver->get_info(info);
				solver_info.push_back(
					{{"type", "rc"},
					{"t", t}, // TODO: null if static?
					{"lag_i", lag_i},
					{"info", info}});

				sol = nl_problem.reduced_to_full(tmp_sol);
				nl_problem.update_lagging(tmp_sol);
				nl_problem.gradient(tmp_sol, tmp_grad);
				lagging_converged = tmp_grad.norm() <= lagging_tol;
				logger().debug("Lagging convergece grad_norm={:g} tol={:g}", tmp_grad.norm(), lagging_tol);

				save_subsolve(++subsolve_count, t);
			}

			// ---------------------------------------------------------------------

			if (friction_iterations > 0)
			{
				nl_problem.gradient(tmp_sol, tmp_grad);
				logger().log(
					lagging_converged ? spdlog::level::info : spdlog::level::warn,
					"{} {:d} lagging iteration(s) (err={:g} tol={:g})",
					lagging_converged ? "Friction lagging converged using" : "Friction lagging maxed out at", lag_i,
					tmp_grad.norm(),
					args["solver"]["contact"]["friction_convergence_tol"].get<double>());
			}
		}
	}

	////////////////////////////////////////////////////////////////////////
	// Template instantiations
	template std::shared_ptr<cppoptlib::NonlinearSolver<NLProblem>> State::make_nl_solver() const;
} // namespace polyfem
