#include <polyfem/State.hpp>

#include <polyfem/io/MatrixIO.hpp>
#include <polyfem/utils/Timer.hpp>

namespace polyfem
{
	using namespace assembler;
	using namespace io;
	using namespace utils;

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
			{
				logger().debug("Unable to read initial {} from file ({})", x_name, state_path);
				return false;
			}

			if (reorder)
			{
				const int ndof = in_node_to_node.size() * dim;
				// only reorder the first ndof rows
				x.topRows(ndof) = utils::reorder_matrix(x.topRows(ndof), in_node_to_node, -1, dim);
			}

			return true;
		}

		bool check_override_shape(const Eigen::MatrixXd &override, const int ndof)
		{
			if (override.rows() != ndof)
			{
				return false;
			}
			if (override.cols() < 1)
			{
				return false;
			}
			return true;
		}
	} // namespace

	void State::init_solve(Eigen::MatrixXd &sol, Eigen::MatrixXd &pressure, const InitialConditionOverride *ic_override)
	{
		POLYFEM_SCOPED_TIMER("Setup RHS");

		solve_data.rhs_assembler = build_rhs_assembler();

		initial_solution(sol, ic_override);
		if (sol.cols() > 1) // ignore previous solutions
			sol.conservativeResize(Eigen::NoChange, 1);

		if (mixed_assembler != nullptr)
		{
			const int actual_dim = problem->is_scalar() ? 1 : mesh->dimension();

			pressure.resize(0, 0);
			sol.conservativeResize(rhs.size(), sol.cols());
			// Zero initial pressure
			sol.middleRows(n_bases * actual_dim, n_pressure_bases).setZero();
			sol(sol.size() - 1) = 0;

			sol_to_pressure(sol, pressure);
		}

		if (problem->is_time_dependent())
			save_timestep(0, 0, 0, 0, sol, pressure);
	}

	void State::initial_solution(Eigen::MatrixXd &solution, const InitialConditionOverride *ic_override) const
	{
		assert(solve_data.rhs_assembler != nullptr);

		// Runtime override has the highest priority.
		if (ic_override && ic_override->solution.size() != 0)
		{
			if (!check_override_shape(ic_override->solution, ndof()))
			{
				log_and_throw_adjoint_error("Invalid initial solution shape ({}, {}). Expect ({}, >=1).",
											ic_override->solution.rows(),
											ic_override->solution.cols(),
											ndof());
			}
			logger().info("Using runtime override for initial solution.");
			solution = ic_override->solution;
			return;
		}

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

	void State::initial_velocity(Eigen::MatrixXd &velocity, const InitialConditionOverride *ic_override) const
	{
		assert(solve_data.rhs_assembler != nullptr);

		// Runtime override has the highest priority.
		if (ic_override && ic_override->velocity.size() != 0)
		{
			if (!check_override_shape(ic_override->velocity, ndof()))
			{
				log_and_throw_adjoint_error("Invalid initial velocity shape ({}, {}). Expect ({}, >=1).",
											ic_override->velocity.rows(),
											ic_override->velocity.cols(),
											ndof());
			}
			logger().info("Using runtime override for initial velocity.");
			velocity = ic_override->velocity;
			return;
		}

		const bool was_velocity_loaded = read_initial_x_from_file(
			resolve_input_path(args["input"]["data"]["state"]), "v",
			args["input"]["data"]["reorder"], in_node_to_node,
			mesh->dimension(), velocity);

		if (!was_velocity_loaded)
			solve_data.rhs_assembler->initial_velocity(velocity);
	}

	void State::initial_acceleration(Eigen::MatrixXd &acceleration, const InitialConditionOverride *ic_override) const
	{
		assert(solve_data.rhs_assembler != nullptr);

		if (ic_override != nullptr && ic_override->acceleration.size() != 0)
		{
			if (!check_override_shape(ic_override->acceleration, ndof()))
			{
				log_and_throw_adjoint_error("Invalid initial acceleration shape ({}, {}). Expect ({}, >=1).",
											ic_override->acceleration.rows(),
											ic_override->acceleration.cols(),
											ndof());
			}
			logger().info("Using runtime override for initial acceleration.");
			acceleration = ic_override->acceleration;
			return;
		}

		const bool was_acceleration_loaded = read_initial_x_from_file(
			resolve_input_path(args["input"]["data"]["state"]), "a",
			args["input"]["data"]["reorder"], in_node_to_node,
			mesh->dimension(), acceleration);

		if (!was_acceleration_loaded)
			solve_data.rhs_assembler->initial_acceleration(acceleration);
	}
} // namespace polyfem
