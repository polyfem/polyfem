#include <polyfem/State.hpp>

#include <polyfem/utils/Timer.hpp>

namespace polyfem
{
	using namespace assembler;
	using namespace utils;

	void State::init_solve()
	{
		POLYFEM_SCOPED_TIMER("Setup RHS");

		json rhs_solver_params = args["solver"]["linear"];
		if (!rhs_solver_params.contains("Pardiso"))
			rhs_solver_params["Pardiso"] = {};
		rhs_solver_params["Pardiso"]["mtype"] = -2; // matrix type for Pardiso (2 = SPD)

		const int size = problem->is_scalar() ? 1 : mesh->dimension();
		const auto &gbases = iso_parametric() ? bases : geom_bases;

		solve_data.rhs_assembler = std::make_shared<RhsAssembler>(
			assembler, *mesh, obstacle, input_dirichlet, n_bases, size, bases, gbases, ass_vals_cache, formulation(),
			*problem, args["space"]["advanced"]["bc_method"], args["solver"]["linear"]["solver"],
			args["solver"]["linear"]["precond"], rhs_solver_params);

		initial_solution(sol);

		if (assembler.is_mixed(formulation()))
		{
			pressure.resize(0, 0);
			const int prev_size = sol.size();
			sol.conservativeResize(rhs.size(), sol.cols());
			// Zero initial pressure
			sol.middleRows(prev_size, n_pressure_bases).setZero();
			sol(sol.size() - 1) = 0;

			sol_to_pressure();
		}

		if (problem->is_time_dependent())
			save_timestep(0, 0, 0, 0);
	}

	void State::initial_solution(Eigen::MatrixXd &solution) const
	{
		assert(solve_data.rhs_assembler != nullptr);
		const std::string in_path = resolve_input_path(args["input"]["data"]["u_path"]);
		if (!in_path.empty())
			import_matrix(in_path, args["import"], solution);
		else
		{
			if (problem->is_time_dependent())
				solve_data.rhs_assembler->initial_solution(solution);
			else
			{
				solution.resize(rhs.size(), 1);
				solution.setZero();
			}
		}
	}

	void State::initial_velocity(Eigen::MatrixXd &velocity) const
	{
		assert(solve_data.rhs_assembler != nullptr);
		const std::string in_path = resolve_input_path(args["input"]["data"]["v_path"]);
		if (!in_path.empty())
			import_matrix(in_path, args["import"], velocity);
		else
			solve_data.rhs_assembler->initial_velocity(velocity);
	}

	void State::initial_acceleration(Eigen::MatrixXd &acceleration) const
	{
		assert(solve_data.rhs_assembler != nullptr);
		const std::string in_path = resolve_input_path(args["input"]["data"]["a_path"]);
		if (!in_path.empty())
			import_matrix(in_path, args["import"], acceleration);
		else
			solve_data.rhs_assembler->initial_acceleration(acceleration);
	}
} // namespace polyfem
