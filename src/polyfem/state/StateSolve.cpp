#include <polyfem/State.hpp>

#include <polyfem/io/MatrixIO.hpp>
#include <polyfem/utils/Timer.hpp>

namespace polyfem
{
	using namespace assembler;
	using namespace io;
	using namespace utils;

	void State::init_solve(Eigen::MatrixXd &sol, Eigen::MatrixXd &pressure)
	{
		POLYFEM_SCOPED_TIMER("Setup RHS");

		logger().info("Solve using {} linear solver", args["solver"]["linear"]["solver"].get<std::string>());

		solve_data.rhs_assembler = build_rhs_assembler();

		initial_solution(sol);
		if (sol.cols() > 1) // ignore previous solutions
			sol.conservativeResize(Eigen::NoChange, 1);

		if (mixed_assembler != nullptr)
		{
			pressure.resize(0, 0);
			sol.conservativeResize(rhs.size(), sol.cols());
			// Zero initial pressure
			sol.middleRows(n_bases * mesh->dimension(), n_pressure_bases).setZero();
			sol(sol.size() - 1) = 0;

			sol_to_pressure(sol, pressure);
		}

		if (problem->is_time_dependent())
			save_timestep(0, 0, 0, 0, sol, pressure);
	}

	namespace
	{
		bool read_initial_x_from_file(
			const std::string &state_path,
			const std::string &x_name,
			const bool reorder,
			const Eigen::VectorXi &in_node_to_node,
			const int dim,
			Eigen::MatrixXd &x)
		{
			if (state_path.empty())
				return false;

			if (!read_matrix(state_path, x_name, x))
				log_and_throw_error("Unable to read initial {} from file ({})!", x_name, state_path);

			if (reorder)
			{
				const int ndof = in_node_to_node.size() * dim;
				// only reorder the first ndof rows
				x.topRows(ndof) = utils::reorder_matrix(x.topRows(ndof), in_node_to_node, -1, dim);
			}

			return true;
		}
	} // namespace

	void State::initial_solution(Eigen::MatrixXd &solution) const
	{
		assert(solve_data.rhs_assembler != nullptr);

		const bool was_solution_loaded = read_initial_x_from_file(
			resolve_input_path(args["input"]["data"]["state"]), "u",
			args["input"]["data"]["reorder"], in_node_to_node,
			mesh->dimension(), solution);

		if (!was_solution_loaded)
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

		const bool was_velocity_loaded = read_initial_x_from_file(
			resolve_input_path(args["input"]["data"]["state"]), "v",
			args["input"]["data"]["reorder"], in_node_to_node,
			mesh->dimension(), velocity);

		if (!was_velocity_loaded)
			solve_data.rhs_assembler->initial_velocity(velocity);
	}

	void State::initial_acceleration(Eigen::MatrixXd &acceleration) const
	{
		assert(solve_data.rhs_assembler != nullptr);

		const bool was_acceleration_loaded = read_initial_x_from_file(
			resolve_input_path(args["input"]["data"]["state"]), "a",
			args["input"]["data"]["reorder"], in_node_to_node,
			mesh->dimension(), acceleration);

		if (!was_acceleration_loaded)
			solve_data.rhs_assembler->initial_acceleration(acceleration);
	}
} // namespace polyfem
