#include <polyfem/State.hpp>

#include <polyfem/solver/forms/BodyForm.hpp>
#include <polyfem/solver/forms/ContactForm.hpp>
#include <polyfem/solver/forms/ElasticForm.hpp>
#include <polyfem/solver/forms/FrictionForm.hpp>
#include <polyfem/solver/forms/InertiaForm.hpp>
#include <polyfem/solver/forms/LaggedRegForm.hpp>
#include <polyfem/solver/forms/ALForm.hpp>

#include <polyfem/solver/NonlinearSolver.hpp>
#include <polyfem/solver/LBFGSSolver.hpp>
#include <polyfem/solver/SparseNewtonDescentSolver.hpp>
#include <polyfem/solver/NLProblem.hpp>
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

	void SolveData::set_al_weight(const double weight)
	{
		if (al_form == nullptr)
			return;
		if (weight >= 0)
		{
			al_form->set_enabled(true);
			al_form->set_weight(weight);
			body_form->set_apply_DBC(false);
			nl_problem->set_full_size(true);
		}
		else
		{
			al_form->set_enabled(false);
			body_form->set_apply_DBC(true);
			nl_problem->set_full_size(false);
		}
	}

	void SolveData::updated_barrier_stiffness(const Eigen::VectorXd &x)
	{
		// TODO: missing use_adaptive_barrier_stiffness_ if (use_adaptive_barrier_stiffness_ && is_time_dependent_)
		if (inertia_form == nullptr)
			return;
		if (contact_form == nullptr)
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

		contact_form->initialize_barrier_stiffness(x, grad_energy);
	}

	void SolveData::update_dt()
	{
		if (inertia_form)
		{
			elastic_form->set_weight(inertia_form->acceleration_scaling());
			body_form->set_weight(inertia_form->acceleration_scaling());

			// if (friction_form)
			// 	friction_form->set_weight(inertia_form->acceleration_scaling());
		}
	}

	template <typename ProblemType>
	std::shared_ptr<cppoptlib::NonlinearSolver<ProblemType>> State::make_nl_solver() const
	{
		const std::string name = args["solver"]["nonlinear"]["solver"];
		if (name == "newton" || name == "Newton")
		{
			return std::make_shared<cppoptlib::SparseNewtonDescentSolver<ProblemType>>(
				args["solver"]["nonlinear"], args["solver"]["linear"]["solver"], args["solver"]["linear"]["precond"]);
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

		for (int t = 1; t <= time_steps; ++t)
		{
			solve_tensor_nonlinear(t);

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

		///////////////////////////////////////////////////////////////////////
		// Check for initial intersections
		if (is_contact_enabled())
		{
			POLYFEM_SCOPED_TIMER("Check for initial intersections");

			Eigen::MatrixXd displaced = boundary_nodes_pos + unflatten(sol, mesh->dimension());

			if (ipc::has_intersections(collision_mesh, collision_mesh.vertices(displaced)))
			{
				OBJWriter::write(
					resolve_output_path("intersection.obj"), collision_mesh.vertices(displaced),
					collision_mesh.edges(), collision_mesh.faces());
				logger().error("Unable to solve, initial solution has intersections!");
				throw std::runtime_error("Unable to solve, initial solution has intersections!");
			}
		}

		assert(solve_data.rhs_assembler != nullptr);

		std::vector<std::shared_ptr<Form>> forms;
		solve_data.elastic_form = std::make_shared<ElasticForm>(*this);
		forms.push_back(solve_data.elastic_form);
		solve_data.body_form = std::make_shared<BodyForm>(*this, *solve_data.rhs_assembler, /*apply_DBC=*/true);
		forms.push_back(solve_data.body_form);

		solve_data.inertia_form = nullptr;
		if (problem->is_time_dependent())
		{
			solve_data.time_integrator = time_integrator::ImplicitTimeIntegrator::construct_time_integrator(args["time"]["integrator"]);
			solve_data.time_integrator->set_parameters(args["time"]);
			solve_data.time_integrator->set_parameters(args["time"]["BDF"]);
			solve_data.time_integrator->set_parameters(args["time"]["newmark"]);
			solve_data.inertia_form = std::make_shared<InertiaForm>(mass, *solve_data.time_integrator);
			forms.push_back(solve_data.inertia_form);
		}
		else
		{
			// TODO: fix me
			//  const double lagged_damping_weight = args["solver"]["contact"]["lagged_damping_weight"].get<double>();
			//  if (lagged_damping_weight > 0)
			//  {
			//  	forms.push_back(std::make_shared<LaggedRegForm>(lagged_damping_weight));
			//  }
		}

		solve_data.al_form = std::make_shared<ALForm>(*this, *solve_data.rhs_assembler, t);
		forms.push_back(solve_data.al_form);

		solve_data.contact_form = nullptr;
		solve_data.friction_form = nullptr;
		if (args["contact"]["enabled"])
		{
			const double dhat = args["contact"]["dhat"];
			assert(dhat > 0);
			const double epsv = args["contact"]["epsv"];
			assert(epsv > 0);
			const double mu = args["contact"]["friction_coefficient"];
			const bool use_adaptive_barrier_stiffness = !args["solver"]["contact"]["barrier_stiffness"].is_number();
			double barrier_stiffness;
			if (use_adaptive_barrier_stiffness)
			{
				barrier_stiffness = 1;
				logger().debug("Using adaptive barrier stiffness");
			}
			else
			{
				assert(args["solver"]["contact"]["barrier_stiffness"].is_number());
				barrier_stiffness = args["solver"]["contact"]["barrier_stiffness"];
				logger().debug("Using fixed barrier stiffness of {}", barrier_stiffness);
			}

			const ipc::BroadPhaseMethod broad_phase_method = args["solver"]["contact"]["CCD"]["broad_phase"];
			const double ccd_tolerance = args["solver"]["contact"]["CCD"]["tolerance"];
			const int ccd_max_iterations = args["solver"]["contact"]["CCD"]["max_iterations"];

			solve_data.contact_form = std::make_shared<ContactForm>(*this, args["contact"]["dhat"], use_adaptive_barrier_stiffness,
																	solve_data.time_integrator != nullptr,
																	broad_phase_method, ccd_tolerance, ccd_max_iterations);
			forms.push_back(solve_data.contact_form);
			if (mu != 0)
			{
				const double dt = args["time"]["dt"];
				solve_data.friction_form = std::make_shared<FrictionForm>(*this, epsv, mu, dhat, broad_phase_method, dt, *solve_data.contact_form);
				forms.push_back(solve_data.friction_form);
			}
		}

		///////////////////////////////////////////////////////////////////////
		// Initialize nonlinear problems
		solve_data.nl_problem = std::make_shared<NLProblem>(*this, forms);

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

			const double dt = args["time"]["dt"];
			solve_data.time_integrator->init(sol, velocity, acceleration, dt);
		}
		solve_data.update_dt();

		///////////////////////////////////////////////////////////////////////

		solver_info = json::array();
	}

	void State::solve_tensor_nonlinear(const int t)
	{
		Eigen::VectorXd tmp_sol;

		assert(solve_data.nl_problem != nullptr);
		NLProblem &nl_problem = *(solve_data.nl_problem);

		assert(sol.size() == rhs.size());

		double al_weight = args["solver"]["augmented_lagrangian"]["initial_weight"];
		const double max_al_weight = args["solver"]["augmented_lagrangian"]["max_weight"];

		nl_problem.full_to_reduced(sol, tmp_sol);
		assert(sol.size() == rhs.size());
		assert(tmp_sol.size() <= rhs.size());

		{
			POLYFEM_SCOPED_TIMER("Initializing lagging");
			nl_problem.init_lagging(sol);
		}

		const int friction_iterations = args["solver"]["contact"]["friction_iterations"];
		assert(friction_iterations >= 0);
		if (friction_iterations > 0)
			logger().debug("Lagging iteration 1");

		const double lagging_tol = args["solver"]["contact"].value("friction_convergence_tol", 1e-2);

		// Disable damping for the final lagged iteration
		// TODO: fix me lagged damping
		// if (friction_iterations <= 1)
		// {
		// 	nl_problem.lagged_regularization_weight() = 0;
		// 	alnl_problem.lagged_regularization_weight() = 0;
		// }

		// Save the subsolve sequence for debugging
		int subsolve_count = 0;
		save_subsolve(subsolve_count, t);

		///////////////////////////////////////////////////////////////////////

		nl_problem.line_search_begin(sol, tmp_sol);
		bool force_al = args["solver"]["augmented_lagrangian"]["force"];
		while (
			force_al || !std::isfinite(nl_problem.value(tmp_sol)) || !nl_problem.is_step_valid(sol, tmp_sol)
			|| (solve_data.contact_form != nullptr && !solve_data.contact_form->is_step_collision_free(sol, tmp_sol)))
		{
			force_al = false;
			nl_problem.line_search_end();
			solve_data.set_al_weight(al_weight);
			logger().debug("Solving AL Problem with weight {}", al_weight);

			std::shared_ptr<cppoptlib::NonlinearSolver<NLProblem>> alnl_solver = make_nl_solver<NLProblem>();
			alnl_solver->setLineSearch(args["solver"]["nonlinear"]["line_search"]["method"]);
			nl_problem.init(sol);
			solve_data.updated_barrier_stiffness(sol);
			tmp_sol = sol;
			alnl_solver->minimize(nl_problem, tmp_sol);
			json alnl_solver_info;
			alnl_solver->getInfo(alnl_solver_info);

			solver_info.push_back(
				{{"type", "al"},
				 {"t", t}, // TODO: null if static?
				 {"weight", al_weight},
				 {"info", alnl_solver_info}});

			sol = tmp_sol;
			nl_problem.full_to_reduced(sol, tmp_sol);
			nl_problem.line_search_begin(sol, tmp_sol);

			al_weight *= 2;

			if (al_weight >= max_al_weight)
			{
				const std::string msg =
					fmt::format("Unable to solve AL problem, weight {} >= {}, stopping", al_weight, max_al_weight);
				logger().error(msg);
				throw std::runtime_error(msg);
				break;
			}

			save_subsolve(++subsolve_count, t);
		}
		solve_data.set_al_weight(-1);
		nl_problem.line_search_end();

		///////////////////////////////////////////////////////////////////////

		std::shared_ptr<cppoptlib::NonlinearSolver<NLProblem>> nl_solver = make_nl_solver<NLProblem>();
		nl_solver->setLineSearch(args["solver"]["nonlinear"]["line_search"]["method"]);
		nl_problem.init(sol);
		solve_data.updated_barrier_stiffness(sol);
		nl_solver->minimize(nl_problem, tmp_sol);
		json nl_solver_info;
		nl_solver->getInfo(nl_solver_info);
		solver_info.push_back(
			{{"type", "rc"},
			 {"t", t}, // TODO: null if static?
			 {"info", nl_solver_info}});
		nl_problem.reduced_to_full(tmp_sol, sol);

		save_subsolve(++subsolve_count, t);

		///////////////////////////////////////////////////////////////////////

		// TODO: fix this lagged damping
		// nl_problem.lagged_regularization_weight() = 0;

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

			nl_solver->getInfo(nl_solver_info);
			solver_info.push_back(
				{{"type", "rc"},
				 {"t", t}, // TODO: null if static?
				 {"lag_i", lag_i},
				 {"info", nl_solver_info}});

			nl_problem.reduced_to_full(tmp_sol, sol);
			nl_problem.update_lagging(tmp_sol);
			nl_problem.gradient(tmp_sol, tmp_grad);
			lagging_converged = tmp_grad.norm() <= lagging_tol;
			logger().debug("Lagging convergece grad_norm={:g} tol={:g}", tmp_grad.norm(), lagging_tol);

			save_subsolve(++subsolve_count, t);
		}

		///////////////////////////////////////////////////////////////////////

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

	////////////////////////////////////////////////////////////////////////
	// Template instantiations
	template std::shared_ptr<cppoptlib::NonlinearSolver<NLProblem>> State::make_nl_solver() const;
} // namespace polyfem
