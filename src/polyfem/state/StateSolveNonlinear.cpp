#include <polyfem/State.hpp>

#include <polyfem/solver/NonlinearSolver.hpp>
#include <polyfem/solver/LBFGSSolver.hpp>
#include <polyfem/solver/SparseNewtonDescentSolver.hpp>
#include <polyfem/solver/NLProblem.hpp>
#include <polyfem/solver/ALNLProblem.hpp>
#include <polyfem/mesh/RemeshAdaptive.hpp>
#include <polyfem/utils/OBJ_IO.hpp>
#include <polyfem/utils/L2Projection.hpp>
#include <polyfem/utils/MatrixUtils.hpp>
#include <polyfem/utils/Timer.hpp>

#include <ipc/ipc.hpp>

#include <igl/PI.h>

namespace polyfem
{
	using namespace assembler;
	using namespace mesh;
	using namespace solver;
	using namespace utils;

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
				solve_data.nl_problem->update_quantities(t0 + (t + 1) * dt, sol);
				solve_data.alnl_problem->update_quantities(t0 + (t + 1) * dt, sol);
			}

			if (true)
			{
				remesh(t0, dt, t);
			}

			save_timestep(t0 + dt * t, t, t0, dt);

			logger().info("{}/{}  t={}", t, time_steps, t0 + dt * t);
		}

		solve_data.nl_problem->save_raw(
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
				OBJWriter::save(
					resolve_output_path("intersection.obj"), collision_mesh.vertices(displaced),
					collision_mesh.edges(), collision_mesh.faces());
				log_and_throw_error("Unable to solve, initial solution has intersections!");
			}
		}

		///////////////////////////////////////////////////////////////////////
		// Initialize nonlinear problems
		assert(solve_data.rhs_assembler != nullptr);
		solve_data.nl_problem =
			std::make_shared<NLProblem>(*this, *solve_data.rhs_assembler, t, args["contact"]["dhat"]);

		const double al_weight = args["solver"]["augmented_lagrangian"]["initial_weight"];
		solve_data.alnl_problem =
			std::make_shared<ALNLProblem>(*this, *solve_data.rhs_assembler, t, args["contact"]["dhat"], al_weight);

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
			solve_data.nl_problem->init_time_integrator(sol, velocity, acceleration, dt);
			solve_data.alnl_problem->init_time_integrator(sol, velocity, acceleration, dt);
		}

		///////////////////////////////////////////////////////////////////////

		solver_info = json::array();
	}

	void State::solve_tensor_nonlinear(const int t)
	{
		Eigen::VectorXd tmp_sol;

		assert(solve_data.nl_problem != nullptr);
		NLProblem &nl_problem = *(solve_data.nl_problem);
		assert(solve_data.alnl_problem != nullptr);
		ALNLProblem &alnl_problem = *(solve_data.alnl_problem);

		assert(sol.size() == rhs.size());

		double al_weight = args["solver"]["augmented_lagrangian"]["initial_weight"];
		const double max_al_weight = args["solver"]["augmented_lagrangian"]["max_weight"];

		nl_problem.full_to_reduced(sol, tmp_sol);
		assert(sol.size() == rhs.size());
		assert(tmp_sol.size() <= rhs.size());

		{
			POLYFEM_SCOPED_TIMER("Initializing lagging");
			nl_problem.init_lagging(sol);
			alnl_problem.init_lagging(sol);
		}

		const int friction_iterations = args["solver"]["contact"]["friction_iterations"];
		assert(friction_iterations >= 0);
		if (friction_iterations > 0)
			logger().debug("Lagging iteration 1");

		// Disable damping for the final lagged iteration
		if (friction_iterations <= 1)
		{
			nl_problem.lagged_regularization_weight() = 0;
			alnl_problem.lagged_regularization_weight() = 0;
		}

		// Save the subsolve sequence for debugging
		int subsolve_count = 0;
		save_subsolve(subsolve_count, t);

		///////////////////////////////////////////////////////////////////////

		nl_problem.line_search_begin(sol, tmp_sol);
		bool force_al = args["solver"]["augmented_lagrangian"]["force"];
		while (force_al || !std::isfinite(nl_problem.value(tmp_sol)) || !nl_problem.is_step_valid(sol, tmp_sol)
			   || !nl_problem.is_step_collision_free(sol, tmp_sol))
		{
			force_al = false;
			nl_problem.line_search_end();
			alnl_problem.set_weight(al_weight);
			logger().debug("Solving AL Problem with weight {}", al_weight);

			std::shared_ptr<cppoptlib::NonlinearSolver<ALNLProblem>> alnl_solver = make_nl_solver<ALNLProblem>();
			alnl_solver->setLineSearch(args["solver"]["nonlinear"]["line_search"]["method"]);
			alnl_problem.init(sol);
			tmp_sol = sol;
			alnl_solver->minimize(alnl_problem, tmp_sol);
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
				log_and_throw_error(fmt::format("Unable to solve AL problem, weight {} >= {}, stopping", al_weight, max_al_weight));

			save_subsolve(++subsolve_count, t);
		}
		nl_problem.line_search_end();

		///////////////////////////////////////////////////////////////////////

		std::shared_ptr<cppoptlib::NonlinearSolver<NLProblem>> nl_solver = make_nl_solver<NLProblem>();
		nl_solver->setLineSearch(args["solver"]["nonlinear"]["line_search"]["method"]);
		nl_problem.init(sol);
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

		// TODO: fix this
		nl_problem.lagged_regularization_weight() = 0;

		// Lagging loop (start at 1 because we already did an iteration above)
		int lag_i;
		nl_problem.update_lagging(tmp_sol);
		bool lagging_converged = nl_problem.lagging_converged(tmp_sol);
		for (lag_i = 1; !lagging_converged && lag_i < friction_iterations; lag_i++)
		{
			logger().debug("Lagging iteration {:d}", lag_i + 1);
			nl_problem.init(sol);
			// Disable damping for the final lagged iteration
			if (lag_i == friction_iterations - 1)
				nl_problem.lagged_regularization_weight() = 0;
			nl_solver->minimize(nl_problem, tmp_sol);

			nl_solver->getInfo(nl_solver_info);
			solver_info.push_back(
				{{"type", "rc"},
				 {"t", t}, // TODO: null if static?
				 {"lag_i", lag_i},
				 {"info", nl_solver_info}});

			nl_problem.reduced_to_full(tmp_sol, sol);
			nl_problem.update_lagging(tmp_sol);
			lagging_converged = nl_problem.lagging_converged(tmp_sol);

			save_subsolve(++subsolve_count, t);
		}

		///////////////////////////////////////////////////////////////////////

		if (friction_iterations > 0)
		{
			logger().log(
				lagging_converged ? spdlog::level::info : spdlog::level::warn,
				"{} {:d} lagging iteration(s) (err={:g} tol={:g})",
				lagging_converged ? "Friction lagging converged using" : "Friction lagging maxed out at", lag_i,
				nl_problem.compute_lagging_error(tmp_sol),
				args["solver"]["contact"]["friction_convergence_tol"].get<double>());
		}
	}

	void State::remesh(const double t0, const double dt, const int t)
	{
		Eigen::MatrixXd V(mesh->n_vertices(), mesh->dimension());
		for (int i = 0; i < mesh->n_vertices(); ++i)
			V.row(i) = mesh->point(i);
		Eigen::MatrixXi F(mesh->n_faces(), mesh->dimension() + 1);
		for (int i = 0; i < F.rows(); ++i)
			for (int j = 0; j < F.cols(); ++j)
				F(i, j) = mesh->face_vertex(i, j);
		OBJWriter::save(resolve_output_path("rest.obj"), V, F);

		// TODO: compute stress at the nodes
		// Eigen::MatrixXd SF;
		// compute_scalar_value(mesh->n_vertices(), sol, SF, false, false);
		Eigen::MatrixXd SV, TV;
		// average_grad_based_function(mesh->n_vertices(), sol, SV, TV, false, false);

		// TODO: What measure to use for remeshing?
		// SV.normalize();
		SV.setOnes(mesh->n_vertices(), 1);
		SV *= 0.1 / t;

		MmgOptions mmg_options;
		mmg_options.hmin = 1e-4;

		Eigen::MatrixXd V_new;
		Eigen::MatrixXi F_new;
		if (!mesh->is_volume())
		{
			mesh::remesh_adaptive_2d(V, F, SV, V_new, F_new, mmg_options);

			// Rotate 90 degrees each step
			// Matrix2d R;
			// const double theta = 90 * (igl::PI / 180);
			// R << cos(theta), sin(theta),
			// 	-sin(theta), cos(theta);
			// V_new = V * R.transpose();

			// V_new = V;
			// F_new = F;

			OBJWriter::save(resolve_output_path("remeshed.obj"), V_new, F_new);
		}
		else
		{
			Eigen::MatrixXi _;
			mesh::remesh_adaptive_3d(V, F, SV, V_new, _, F_new);
		}

		// --------------------------------------------------------------------

		// Save old values
		const int old_n_bases = n_bases;
		const std::vector<ElementBases> old_bases = bases;
		const std::vector<ElementBases> old_geom_bases = iso_parametric() ? bases : geom_bases;
		const StiffnessMatrix old_mass = mass;
		Eigen::MatrixXd y(sol.size(), 3); // Old values of independent variables
		y.col(0) = sol;
		y.col(1) = solve_data.nl_problem->time_integrator()->v_prev();
		y.col(2) = solve_data.nl_problem->time_integrator()->a_prev();

		// --------------------------------------------------------------------

		this->load_mesh(V_new, F_new);
		// FIXME:
		mesh->compute_boundary_ids(1e-6);
		mesh->set_body_ids(std::vector<int>(mesh->n_elements(), 1));
		this->set_materials(); // TODO: Explain why I need this?
		this->build_basis();
		this->assemble_rhs();
		this->assemble_stiffness_mat();

		// --------------------------------------------------------------------

		// L2 Projection
		ass_vals_cache.clear(); // Clear this because the mass matrix needs to be recomputed
		Eigen::MatrixXd x;
		L2_projection(
			*this, *solve_data.rhs_assembler,
			mesh->is_volume(), mesh->is_volume() ? 3 : 2,
			old_n_bases, old_bases, old_geom_bases,                // from
			n_bases, bases, iso_parametric() ? bases : geom_bases, // to
			ass_vals_cache, y, x, /*lump_mass_matrix=*/false);

		sol = x.col(0);
		Eigen::VectorXd vel = x.col(1);
		Eigen::VectorXd acc = x.col(2);

		if (x.rows() < 30)
		{
			logger().critical("yᵀ:\n{}", y.transpose());
			logger().critical("xᵀ:\n{}", x.transpose());
		}

		// Compute Projection error
		if (false)
		{
			Eigen::MatrixXd y2;
			L2_projection(
				*this, *solve_data.rhs_assembler,
				mesh->is_volume(), mesh->is_volume() ? 3 : 2,
				n_bases, bases, iso_parametric() ? bases : geom_bases, // from
				old_n_bases, old_bases, old_geom_bases,                // to
				ass_vals_cache, x, y2);

			auto error = [&old_mass](const Eigen::VectorXd &old_y, const Eigen::VectorXd &new_y) -> double {
				const auto diff = new_y - old_y;
				return diff.transpose() * old_mass * diff;
				// return diff.norm() / diff.rows();
			};

			std::cout << fmt::format(
				"L2_Projection_Error, {}, {}, {}, {}",
				error(y.col(0), y2.col(0)),
				error(y.col(1), y2.col(1)),
				error(y.col(2), y2.col(2)),
				V_new.rows())
					  << std::endl;
		}

		// --------------------------------------------------------------------

		json rhs_solver_params = args["solver"]["linear"];
		if (!rhs_solver_params.contains("Pardiso"))
			rhs_solver_params["Pardiso"] = {};
		rhs_solver_params["Pardiso"]["mtype"] = -2; // matrix type for Pardiso (2 = SPD)
		const auto &gbases = iso_parametric() ? bases : geom_bases;
		solve_data.rhs_assembler = std::make_shared<RhsAssembler>(
			assembler, *mesh, obstacle, input_dirichlet,
			n_bases, problem->is_scalar() ? 1 : mesh->dimension(),
			bases, gbases, ass_vals_cache,
			formulation(), *problem,
			args["space"]["advanced"]["bc_method"],
			args["solver"]["linear"]["solver"],
			args["solver"]["linear"]["precond"],
			rhs_solver_params);

		const int full_size = n_bases * mesh->dimension();
		const int reduced_size = n_bases * mesh->dimension() - boundary_nodes.size();

		solve_data.nl_problem = std::make_shared<NLProblem>(*this, *solve_data.rhs_assembler, t0 + t * dt, args["contact"]["dhat"]);
		solve_data.nl_problem->init_time_integrator(sol, vel, acc, dt);

		double al_weight = args["solver"]["augmented_lagrangian"]["initial_weight"];
		solve_data.alnl_problem = std::make_shared<ALNLProblem>(*this, *solve_data.rhs_assembler, t0 + t * dt, args["contact"]["dhat"], al_weight);
		solve_data.alnl_problem->init_time_integrator(sol, vel, acc, dt);

		// TODO: Check for inversions and intersections due to remeshing
	}

	////////////////////////////////////////////////////////////////////////
	// Template instantiations
	template std::shared_ptr<cppoptlib::NonlinearSolver<NLProblem>> State::make_nl_solver() const;
	template std::shared_ptr<cppoptlib::NonlinearSolver<ALNLProblem>> State::make_nl_solver() const;
} // namespace polyfem
