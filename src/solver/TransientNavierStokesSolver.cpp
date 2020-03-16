#include <polyfem/TransientNavierStokesSolver.hpp>

#include <polyfem/MatrixUtils.hpp>
#include <polyfem/FEMSolver.hpp>
#include <polyfem/AssemblerUtils.hpp>

#include <polyfem/Logger.hpp>

#include <igl/Timer.h>

#include <unsupported/Eigen/SparseExtra>

#include <cmath>

namespace polyfem
{

TransientNavierStokesSolver::TransientNavierStokesSolver(double viscosity, const json &solver_param, const json &problem_params, const std::string &solver_type, const std::string &precond_type)
	: viscosity(viscosity), solver_param(solver_param), problem_params(problem_params), solver_type(solver_type), precond_type(precond_type)
{
	gradNorm = solver_param.count("gradNorm") ? double(solver_param["gradNorm"]) : 1e-8;
	iterations = solver_param.count("nl_iterations") ? int(solver_param["nl_iterations"]) : 100;
}

void TransientNavierStokesSolver::minimize(
	const State &state, const double dt, const Eigen::VectorXd &prev_sol,
	const StiffnessMatrix &velocity_stiffness, const StiffnessMatrix &mixed_stiffness, const StiffnessMatrix &pressure_stiffness,
	const StiffnessMatrix &velocity_mass,
	const Eigen::MatrixXd &rhs, Eigen::VectorXd &x)
{
	auto &assembler = AssemblerUtils::instance();
	assembler.clear_cache();

	auto solver = LinearSolver::create(solver_type, precond_type);
	solver->setParameters(solver_param);
	polyfem::logger().debug("\tinternal solver {}", solver->name());

	const int problem_dim = state.mesh->dimension();

	igl::Timer time;

	time.start();
	StiffnessMatrix stoke_stiffness;
	Eigen::VectorXd prev_sol_mass(prev_sol.size()); //prev_sol_mass=prev_sol
	prev_sol_mass.setZero();
	prev_sol_mass.block(0, 0, velocity_mass.rows(), 1) = velocity_mass * prev_sol.block(0, 0, velocity_mass.rows(), 1);
	for (int i : state.boundary_nodes)
		prev_sol_mass[i] = 0;

	AssemblerUtils::merge_mixed_matrices(state.n_bases, state.n_pressure_bases, problem_dim, false,
										 dt * velocity_stiffness + velocity_mass, mixed_stiffness, pressure_stiffness,
										 stoke_stiffness);
	time.stop();
	stokes_matrix_time = time.getElapsedTimeInSec();
	logger().debug("\tStokes matrix assembly time {}s", time.getElapsedTimeInSec());


	time.start();
	logger().info("{}...", solver->name());

	Eigen::VectorXd b = rhs + prev_sol_mass;
	dirichlet_solve(*solver, stoke_stiffness, b, state.boundary_nodes, x);
	// solver->getInfo(solver_info);
	time.stop();
	stokes_solve_time = time.getElapsedTimeInSec();
	logger().debug("\tStokes solve time {}s", time.getElapsedTimeInSec());
	logger().debug("\tStokes solver error: {}", (stoke_stiffness * x - b).norm());

	assembly_time = 0;
	inverting_time = 0;

	int it = 0;
	double nlres_norm = 0;
	it += minimize_aux(state.formulation() + "Picard", state, dt, velocity_stiffness, mixed_stiffness, pressure_stiffness, velocity_mass, b,     1e-3, solver, nlres_norm, x);
	it += minimize_aux(state.formulation()           , state, dt, velocity_stiffness, mixed_stiffness, pressure_stiffness, velocity_mass, b, gradNorm, solver, nlres_norm, x);

	solver_info["iterations"] = it;
	solver_info["gradNorm"] = nlres_norm;

	assembly_time /= it;
	inverting_time /= it;

	solver_info["time_assembly"] = assembly_time;
	solver_info["time_inverting"] = inverting_time;
	solver_info["time_stokes_assembly"] = stokes_matrix_time;
	solver_info["time_stokes_solve"] = stokes_solve_time;
}

int TransientNavierStokesSolver::minimize_aux(
	const std::string &formulation, const State &state, const double dt,
	const StiffnessMatrix &velocity_stiffness, const StiffnessMatrix &mixed_stiffness, const StiffnessMatrix &pressure_stiffness,
	const StiffnessMatrix &velocity_mass,
	const Eigen::VectorXd &rhs, const double grad_norm,
	std::unique_ptr<LinearSolver> &solver, double &nlres_norm,
	Eigen::VectorXd &x)
{
	igl::Timer time;
	const auto &assembler = AssemblerUtils::instance();
	const auto &gbases = state.iso_parametric() ? state.bases : state.geom_bases;
	const int problem_dim = state.problem->is_scalar() ? 1 : state.mesh->dimension();

	StiffnessMatrix nl_matrix;
	StiffnessMatrix total_matrix;


	time.start();
	assembler.assemble_energy_hessian(state.formulation() + "Picard", state.mesh->is_volume(), state.n_bases, state.bases, gbases, x, nl_matrix);
	AssemblerUtils::merge_mixed_matrices(state.n_bases, state.n_pressure_bases, problem_dim, false,
										 dt * (velocity_stiffness + nl_matrix) + velocity_mass, mixed_stiffness, pressure_stiffness,
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
			AssemblerUtils::merge_mixed_matrices(state.n_bases, state.n_pressure_bases, problem_dim, false,
												 dt * (velocity_stiffness + nl_matrix) + velocity_mass, mixed_stiffness, pressure_stiffness,
												 total_matrix);
		}
		dirichlet_solve(*solver, total_matrix, nlres, state.boundary_nodes, dx);
		// for (int i : state.boundary_nodes)
		// 	dx[i] = 0;
		time.stop();
		inverting_time += time.getElapsedTimeInSec();
		logger().debug("\tinverting time {}s", time.getElapsedTimeInSec());

		x += dx;
		//TODO check for nans

		time.start();
		assembler.assemble_energy_hessian(state.formulation() + "Picard", state.mesh->is_volume(), state.n_bases, state.bases, gbases, x, nl_matrix);
		AssemblerUtils::merge_mixed_matrices(state.n_bases, state.n_pressure_bases, problem_dim, false,
											 dt * (velocity_stiffness + nl_matrix) + velocity_mass, mixed_stiffness, pressure_stiffness,
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

} // namespace polyfem
