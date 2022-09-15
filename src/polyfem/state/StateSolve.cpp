#include <polyfem/State.hpp>

#include <polyfem/io/MatrixIO.hpp>
#include <polyfem/utils/Timer.hpp>

namespace polyfem
{
	using namespace assembler;
	using namespace io;
	using namespace utils;

	void State::init_solve()
	{
		POLYFEM_SCOPED_TIMER("Setup RHS");

		logger().info("Solve using {} linear solver", args["solver"]["linear"]["solver"].get<std::string>());

		solve_data.rhs_assembler = build_rhs_assembler();

		initial_solution(sol);

		if (assembler.is_mixed(formulation()))
		{
			pressure.resize(0, 0);
			sol.conservativeResize(rhs.size(), sol.cols());
			// Zero initial pressure
			sol.middleRows(n_bases * mesh->dimension(), n_pressure_bases).setZero();
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
