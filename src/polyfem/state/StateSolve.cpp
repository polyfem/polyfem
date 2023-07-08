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

	void State::initial_solution(Eigen::MatrixXd &solution) const
	{
		assert(solve_data.rhs_assembler != nullptr);
		const std::string in_path = resolve_input_path(args["input"]["data"]["u_path"]);
		if (!in_path.empty())
		{
			if (!read_matrix(in_path, solution))
				log_and_throw_error("Unable to read initial solution from file ({})!", in_path);
			assert(solution.cols() == 1);
			if (args["input"]["data"]["reorder"].get<bool>())
			{
				const int ndof = in_node_to_node.size() * mesh->dimension();
				assert(ndof + obstacle.ndof() == solution.size());
				// only reorder the first ndof rows
				solution.topRows(ndof) = utils::reorder_matrix(solution.topRows(ndof), in_node_to_node, -1, mesh->dimension());
			}
		}
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
		{
			if (!read_matrix(in_path, velocity))
				log_and_throw_error("Unable to read initial velocity from file ({})!", in_path);
			assert(velocity.cols() == 1);
			if (args["input"]["data"]["reorder"].get<bool>())
			{
				const int ndof = in_node_to_node.size() * mesh->dimension();
				assert(ndof + obstacle.ndof() == velocity.size());
				// only reorder the first ndof rows
				velocity.topRows(ndof) = utils::reorder_matrix(velocity.topRows(ndof), in_node_to_node, -1, mesh->dimension());
			}
		}
		else
			solve_data.rhs_assembler->initial_velocity(velocity);
	}

	void State::initial_acceleration(Eigen::MatrixXd &acceleration) const
	{
		assert(solve_data.rhs_assembler != nullptr);
		const std::string in_path = resolve_input_path(args["input"]["data"]["a_path"]);
		if (!in_path.empty())
		{
			if (!read_matrix(in_path, acceleration))
				log_and_throw_error("Unable to read initial acceleration from file ({})!", in_path);
			if (args["input"]["data"]["reorder"].get<bool>())
			{
				assert(acceleration.cols() == 1);
				const int ndof = in_node_to_node.size() * mesh->dimension();
				assert(ndof + obstacle.ndof() == acceleration.size());
				// only reorder the first ndof rows
				acceleration.topRows(ndof) = utils::reorder_matrix(acceleration.topRows(ndof), in_node_to_node, -1, mesh->dimension());
			}
		}
		else
			solve_data.rhs_assembler->initial_acceleration(acceleration);
	}

	int State::full_to_periodic(StiffnessMatrix &A) const
	{
		const int problem_dim = problem->is_scalar() ? 1 : mesh->dimension();
		const int independent_dof = bases_to_periodic_map.maxCoeff() + 1;
		
		// account for potential pressure block
		auto index_map = [&](int id){
			if (id < bases_to_periodic_map.size())
				return bases_to_periodic_map(id);
			else
				return (int)(id + independent_dof - n_bases * problem_dim);
		};

		StiffnessMatrix A_periodic(index_map(A.rows()), index_map(A.cols()));
		std::vector<Eigen::Triplet<double>> entries;
		entries.reserve(A.nonZeros());
		for (int k = 0; k < A.outerSize(); k++)
		{
			for (StiffnessMatrix::InnerIterator it(A,k); it; ++it)
			{
				entries.emplace_back(index_map(it.row()), index_map(it.col()), it.value());
			}
		}
		A_periodic.setFromTriplets(entries.begin(),entries.end());

		std::swap(A_periodic, A);

		return independent_dof;
	}

	int State::full_to_periodic(Eigen::MatrixXd &b, bool accumulate) const
	{
		const int problem_dim = problem->is_scalar() ? 1 : mesh->dimension();
		const int independent_dof = bases_to_periodic_map.maxCoeff() + 1;
		
		// account for potential pressure block
		auto index_map = [&](int id){
			if (id < bases_to_periodic_map.size())
				return bases_to_periodic_map(id);
			else
				return (int)(id + independent_dof - bases_to_periodic_map.size());
		};

		// rhs under periodic basis
		Eigen::MatrixXd b_periodic;
		b_periodic.setZero(index_map(b.rows()), b.cols());
		if (accumulate)
			for (int k = 0; k < b.rows(); k++)
				b_periodic.row(index_map(k)) += b.row(k);
		else
			for (int k = 0; k < b.rows(); k++)
				b_periodic.row(index_map(k)) = b.row(k);

		b = b_periodic;

		return independent_dof;
	}

	Eigen::MatrixXd State::periodic_to_full(const int ndofs, const Eigen::MatrixXd &x_periodic) const
	{
		const int problem_dim = problem->is_scalar() ? 1 : mesh->dimension();
		const int independent_dof = bases_to_periodic_map.maxCoeff() + 1;
		
		auto index_map = [&](int id){
			if (id < bases_to_periodic_map.size())
				return bases_to_periodic_map(id);
			else
				return (int)(id + independent_dof - bases_to_periodic_map.size());
		};

		Eigen::MatrixXd x_full;
		x_full.resize(ndofs, x_periodic.cols());
		for (int i = 0; i < x_full.rows(); i++)
			x_full.row(i) = x_periodic.row(index_map(i));

		return x_full;
	}
} // namespace polyfem
