#include <polyfem/NavierStokesSolver.hpp>

#include <polyfem/MatrixUtils.hpp>
#include <polysolve/FEMSolver.hpp>
#include <polysolve/LinearSolver.hpp>

#include <polyfem/AssemblerUtils.hpp>

#include <polyfem/Logger.hpp>

#include <igl/Timer.h>

#include <unsupported/Eigen/SparseExtra>

#include <cmath>

namespace polyfem
{
	using namespace polysolve;

NavierStokesSolver::NavierStokesSolver(double viscosity, const json &solver_param, const json &problem_params, const std::string &solver_type, const std::string &precond_type)
	: viscosity(viscosity), solver_param(solver_param), problem_params(problem_params), solver_type(solver_type), precond_type(precond_type)
{
	gradNorm = solver_param.count("gradNorm") ? double(solver_param["gradNorm"]) : 1e-8;
	iterations = solver_param.count("nl_iterations") ? int(solver_param["nl_iterations"]) : 100;
}

void NavierStokesSolver::minimize(const State &state, const Eigen::MatrixXd &rhs, Eigen::VectorXd &x)
{
	auto &assembler = AssemblerUtils::instance();
	assembler.clear_cache();

	// problem_params["viscosity"] = 1;
	// assembler.set_parameters(problem_params);

	auto solver = LinearSolver::create(solver_type, precond_type);
	solver->setParameters(solver_param);
	polyfem::logger().debug("\tinternal solver {}", solver->name());

	const auto &gbases = state.iso_parametric() ? state.bases : state.geom_bases;
	const int problem_dim = state.problem->is_scalar() ? 1 : state.mesh->dimension();
	const int precond_num = problem_dim * state.n_bases;

	igl::Timer time;

	time.start();
	StiffnessMatrix stoke_stiffness;
	StiffnessMatrix velocity_stiffness, mixed_stiffness, pressure_stiffness;
	assembler.assemble_problem(state.formulation(), state.mesh->is_volume(), state.n_bases, state.bases, gbases, velocity_stiffness);
	assembler.assemble_mixed_problem(state.formulation(), state.mesh->is_volume(), state.n_pressure_bases, state.n_bases, state.pressure_bases, state.bases, gbases, mixed_stiffness);
	assembler.assemble_pressure_problem(state.formulation(), state.mesh->is_volume(), state.n_pressure_bases, state.pressure_bases, gbases, pressure_stiffness);

	AssemblerUtils::merge_mixed_matrices(state.n_bases, state.n_pressure_bases, problem_dim, state.use_avg_pressure,
										 velocity_stiffness, mixed_stiffness, pressure_stiffness,
										 stoke_stiffness);
	time.stop();
	stokes_matrix_time = time.getElapsedTimeInSec();
	logger().debug("\tStokes matrix assembly time {}s", time.getElapsedTimeInSec());

	time.start();

	logger().info("{}...", solver->name());

	Eigen::VectorXd b = rhs;
	dirichlet_solve(*solver, stoke_stiffness, b, state.boundary_nodes, x, precond_num);
	// solver->getInfo(solver_info);
	time.stop();
	stokes_solve_time = time.getElapsedTimeInSec();
	logger().debug("\tStokes solve time {}s", time.getElapsedTimeInSec());
	logger().debug("\tStokes solver error: {}", (stoke_stiffness * x - b).norm());

	assembly_time = 0;
	inverting_time = 0;

	// velocity_stiffness *= viscosity;
	int it = 0;
	double nlres_norm = 0;
	b = rhs;
	it += minimize_aux(state.formulation() + "Picard", state, velocity_stiffness, mixed_stiffness, pressure_stiffness, b,     1e-3, solver, nlres_norm, x);
	it += minimize_aux(state.formulation()           , state, velocity_stiffness, mixed_stiffness, pressure_stiffness, b, gradNorm, solver, nlres_norm, x);

	solver_info["iterations"] = it;
	solver_info["gradNorm"] = nlres_norm;

	assembly_time /= it;
	inverting_time /= it;

	solver_info["time_assembly"] = assembly_time;
	solver_info["time_inverting"] = inverting_time;
	solver_info["time_stokes_assembly"] = stokes_matrix_time;
	solver_info["time_stokes_solve"] = stokes_solve_time;
}

int NavierStokesSolver::minimize_aux(
	const std::string &formulation, const State &state,
	const StiffnessMatrix &velocity_stiffness, const StiffnessMatrix &mixed_stiffness, const StiffnessMatrix &pressure_stiffness,
	const Eigen::VectorXd &rhs, const double grad_norm,
	std::unique_ptr<LinearSolver> &solver, double &nlres_norm,
	Eigen::VectorXd &x)
{
	igl::Timer time;
	const auto &assembler = AssemblerUtils::instance();
	const auto &gbases = state.iso_parametric() ? state.bases : state.geom_bases;
	const int problem_dim = state.problem->is_scalar() ? 1 : state.mesh->dimension();
	const int precond_num = problem_dim * state.n_bases;

	StiffnessMatrix nl_matrix;
	StiffnessMatrix total_matrix;


	time.start();
	assembler.assemble_energy_hessian(state.formulation() + "Picard", state.mesh->is_volume(), state.n_bases, state.bases, gbases, x, nl_matrix);
	AssemblerUtils::merge_mixed_matrices(state.n_bases, state.n_pressure_bases, problem_dim, state.use_avg_pressure,
										 velocity_stiffness + nl_matrix, mixed_stiffness, pressure_stiffness,
										 total_matrix);
	time.stop();
	assembly_time = time.getElapsedTimeInSec();
	logger().debug("\tNavier Stokes assembly time {}s", time.getElapsedTimeInSec());

	Eigen::VectorXd nlres = -(total_matrix * x) + rhs;
	for (int i : state.boundary_nodes)
		nlres[i] = 0;
	Eigen::VectorXd dx;
	nlres_norm = nlres.norm();
	logger().debug("\tInitial residula norm {}", nlres_norm);

	int it = 0;

	while (nlres_norm > grad_norm && it < iterations)
	{
		++it;

		time.start();
		if (formulation != state.formulation() + "Picard"){
			assembler.assemble_energy_hessian(formulation, state.mesh->is_volume(), state.n_bases, state.bases, gbases, x, nl_matrix);
			AssemblerUtils::merge_mixed_matrices(state.n_bases, state.n_pressure_bases, problem_dim, state.use_avg_pressure,
												 velocity_stiffness + nl_matrix, mixed_stiffness, pressure_stiffness,
												 total_matrix);
		}
		dirichlet_solve(*solver, total_matrix, nlres, state.boundary_nodes, dx, precond_num);
		// for (int i : state.boundary_nodes)
		// 	dx[i] = 0;
		time.stop();
		inverting_time += time.getElapsedTimeInSec();
		logger().debug("\tinverting time {}s", time.getElapsedTimeInSec());
		logger().debug("\tinverting error: {}", (total_matrix * dx - nlres).norm());

		x += dx;
		//TODO check for nans

		time.start();
		assembler.assemble_energy_hessian(state.formulation() + "Picard", state.mesh->is_volume(), state.n_bases, state.bases, gbases, x, nl_matrix);
		AssemblerUtils::merge_mixed_matrices(state.n_bases, state.n_pressure_bases, problem_dim, state.use_avg_pressure,
											 velocity_stiffness + nl_matrix, mixed_stiffness, pressure_stiffness,
											 total_matrix);
		time.stop();
		logger().debug("\tassembly time {}s", time.getElapsedTimeInSec());
		assembly_time += time.getElapsedTimeInSec();

		nlres = -(total_matrix * x) + rhs;
		for (int i : state.boundary_nodes)
			nlres[i] = 0;
		nlres_norm = nlres.norm();

		polyfem::logger().debug("\titer: {},  ||g||_2 = {}, ||step|| = {}\n",
								it, nlres_norm, dx.norm());
	}

	// solver_info["internal_solver"] = internal_solver;
	// solver_info["internal_solver_first"] = internal_solver.front();
	// solver_info["status"] = this->status();

	return it;
}

bool NavierStokesSolver::has_nans(const polyfem::StiffnessMatrix &hessian)
{
	for (int k = 0; k < hessian.outerSize(); ++k)
	{
		for (polyfem::StiffnessMatrix::InnerIterator it(hessian, k); it; ++it)
		{
			if (std::isnan(it.value()))
				return true;
		}
	}

	return false;
}
} // namespace polyfem
