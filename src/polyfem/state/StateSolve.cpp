#include <polyfem/State.hpp>

#include <polyfem/utils/Timer.hpp>

namespace polyfem
{
	void State::init_solve()
	{
		POLYFEM_SCOPED_TIMER("Setup RHS");

		json rhs_solver_params = args["solver"]["linear"];
		if (!rhs_solver_params.contains("Pardiso"))
			rhs_solver_params["Pardiso"] = {};
		rhs_solver_params["Pardiso"]["mtype"] = -2; // matrix type for Pardiso (2 = SPD)

		const int size = problem->is_scalar() ? 1 : mesh->dimension();
		const auto &gbases = iso_parametric() ? bases : geom_bases;

		solve_data.rhs_assembler = std::make_shared<assembler::RhsAssembler>(
			assembler, *mesh, obstacle, input_dirichelt, n_bases, size, bases, gbases, ass_vals_cache, formulation(),
			*problem, args["space"]["advanced"]["bc_method"], args["solver"]["linear"]["solver"],
			args["solver"]["linear"]["precond"], rhs_solver_params);

		const std::string u_path = resolve_input_path(args["input"]["data"]["u_path"]);
		if (!u_path.empty())
			utils::read_matrix(u_path, sol);
		else
			solve_data.rhs_assembler->initial_solution(sol);

		if (assembler.is_mixed(formulation()))
		{
			pressure.resize(0, 0);
			const int prev_size = sol.size();
			sol.conservativeResize(rhs.size(), sol.cols());
			// Zero initial pressure
			sol.block(prev_size, 0, n_pressure_bases, sol.cols()).setZero();
			sol(sol.size() - 1) = 0;
		}

		if (assembler.is_mixed(formulation()))
			sol_to_pressure();
	}
} // namespace polyfem
