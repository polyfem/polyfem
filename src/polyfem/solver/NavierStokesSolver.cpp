#include "NavierStokesSolver.hpp"

#include <polyfem/utils/MatrixUtils.hpp>
#include <polysolve/FEMSolver.hpp>
#include <polysolve/LinearSolver.hpp>

#include <polyfem/assembler/AssemblerUtils.hpp>

#include <polyfem/utils/Logger.hpp>

#include <igl/Timer.h>

#include <unsupported/Eigen/SparseExtra>

#include <cmath>

namespace polyfem
{
	using namespace polysolve;
	using namespace assembler;
	using namespace utils;

	namespace solver
	{
		NavierStokesSolver::NavierStokesSolver(const json &solver_param)
			: solver_param(solver_param),
			  solver_type(solver_param["linear"]["solver"]),
			  precond_type(solver_param["linear"]["precond"])
		{
			gradNorm = solver_param["nonlinear"]["grad_norm"];
			iterations = solver_param["nonlinear"]["max_iterations"];
		}

		void NavierStokesSolver::minimize(const State &state, const Eigen::MatrixXd &rhs, Eigen::VectorXd &x)
		{
			const auto &assembler = state.assembler;

			auto solver = LinearSolver::create(solver_type, precond_type);
			solver->setParameters(solver_param["linear"]);
			logger().debug("\tinternal solver {}", solver->name());

			const auto &gbases = state.iso_parametric() ? state.bases : state.geom_bases;
			const int problem_dim = state.problem->is_scalar() ? 1 : state.mesh->dimension();
			const int precond_num = problem_dim * state.n_bases;

			igl::Timer time;

			time.start();
			StiffnessMatrix stoke_stiffness;
			StiffnessMatrix velocity_stiffness, mixed_stiffness, pressure_stiffness;
			assembler.assemble_problem(state.formulation(), state.mesh->is_volume(), state.n_bases, state.bases, gbases, state.ass_vals_cache, velocity_stiffness);
			assembler.assemble_mixed_problem(state.formulation(), state.mesh->is_volume(), state.n_pressure_bases, state.n_bases, state.pressure_bases, state.bases, gbases, state.pressure_ass_vals_cache, state.ass_vals_cache, mixed_stiffness);
			assembler.assemble_pressure_problem(state.formulation(), state.mesh->is_volume(), state.n_pressure_bases, state.pressure_bases, gbases, state.pressure_ass_vals_cache, pressure_stiffness);

			AssemblerUtils::merge_mixed_matrices(state.n_bases, state.n_pressure_bases, problem_dim, state.use_avg_pressure,
												 velocity_stiffness, mixed_stiffness, pressure_stiffness,
												 stoke_stiffness);
			time.stop();
			stokes_matrix_time = time.getElapsedTimeInSec();
			logger().debug("\tStokes matrix assembly time {}s", time.getElapsedTimeInSec());

			time.start();

			logger().info("{}...", solver->name());

			Eigen::VectorXd b = rhs;
			dirichlet_solve(*solver, stoke_stiffness, b, state.boundary_nodes, x, precond_num, "", false, true, state.use_avg_pressure);
			// solver->getInfo(solver_info);
			time.stop();
			stokes_solve_time = time.getElapsedTimeInSec();
			logger().debug("\tStokes solve time {}s", time.getElapsedTimeInSec());
			logger().debug("\tStokes solver error: {}", (stoke_stiffness * x - b).norm());

			std::vector<bool> zero_col(stoke_stiffness.cols(), true);
			for (int k = 0; k < stoke_stiffness.outerSize(); ++k)
			{
				for (StiffnessMatrix::InnerIterator it(stoke_stiffness, k); it; ++it)
				{
					if (fabs(it.value()) > 1e-12)
						zero_col[it.col()] = false;
				}
			}
			std::vector<int> skipping;
			for (int i = 0; i < zero_col.size(); ++i)
			{
				if (zero_col[i])
				{
					skipping.push_back(i);
				}
			}

			assembly_time = 0;
			inverting_time = 0;

			int it = 0;
			double nlres_norm = 0;
			b = rhs;
			it += minimize_aux(state.formulation() + "Picard", skipping, state, velocity_stiffness, mixed_stiffness, pressure_stiffness, b, 1e-3, solver, nlres_norm, x);
			it += minimize_aux(state.formulation(), skipping, state, velocity_stiffness, mixed_stiffness, pressure_stiffness, b, gradNorm, solver, nlres_norm, x);

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
			const std::string &formulation, const std::vector<int> &skipping, const State &state,
			const StiffnessMatrix &velocity_stiffness, const StiffnessMatrix &mixed_stiffness, const StiffnessMatrix &pressure_stiffness,
			const Eigen::VectorXd &rhs, const double grad_norm,
			std::unique_ptr<LinearSolver> &solver, double &nlres_norm,
			Eigen::VectorXd &x)
		{
			igl::Timer time;
			const auto &assembler = state.assembler;
			const auto &gbases = state.iso_parametric() ? state.bases : state.geom_bases;
			const int problem_dim = state.problem->is_scalar() ? 1 : state.mesh->dimension();
			const int precond_num = problem_dim * state.n_bases;

			StiffnessMatrix nl_matrix;
			StiffnessMatrix total_matrix;
			SpareMatrixCache mat_cache;

			time.start();
			assembler.assemble_energy_hessian(state.formulation() + "Picard", state.mesh->is_volume(), state.n_bases, false, state.bases, gbases, state.ass_vals_cache, x, mat_cache, nl_matrix);
			AssemblerUtils::merge_mixed_matrices(state.n_bases, state.n_pressure_bases, problem_dim, state.use_avg_pressure,
												 velocity_stiffness + nl_matrix, mixed_stiffness, pressure_stiffness,
												 total_matrix);
			time.stop();
			assembly_time = time.getElapsedTimeInSec();
			logger().debug("\tNavier Stokes assembly time {}s", time.getElapsedTimeInSec());

			Eigen::VectorXd nlres = -(total_matrix * x) + rhs;
			for (int i : state.boundary_nodes)
				nlres[i] = 0;
			for (int i : skipping)
				nlres[i] = 0;
			Eigen::VectorXd dx;
			nlres_norm = nlres.norm();
			logger().debug("\tInitial residula norm {}", nlres_norm);

			int it = 0;

			while (nlres_norm > grad_norm && it < iterations)
			{
				++it;

				time.start();
				if (formulation != state.formulation() + "Picard")
				{
					assembler.assemble_energy_hessian(formulation, state.mesh->is_volume(), state.n_bases, false, state.bases, gbases, state.ass_vals_cache, x, mat_cache, nl_matrix);
					AssemblerUtils::merge_mixed_matrices(state.n_bases, state.n_pressure_bases, problem_dim, state.use_avg_pressure,
														 velocity_stiffness + nl_matrix, mixed_stiffness, pressure_stiffness,
														 total_matrix);
				}
				dirichlet_solve(*solver, total_matrix, nlres, state.boundary_nodes, dx, precond_num, "", false, true, state.use_avg_pressure);
				// for (int i : state.boundary_nodes)
				// 	dx[i] = 0;
				time.stop();
				inverting_time += time.getElapsedTimeInSec();
				logger().debug("\tinverting time {}s", time.getElapsedTimeInSec());
				logger().debug("\tinverting error: {}", (total_matrix * dx - nlres).norm());

				x += dx;
				//TODO check for nans

				time.start();
				assembler.assemble_energy_hessian(state.formulation() + "Picard", state.mesh->is_volume(), state.n_bases, false, state.bases, gbases, state.ass_vals_cache, x, mat_cache, nl_matrix);
				AssemblerUtils::merge_mixed_matrices(state.n_bases, state.n_pressure_bases, problem_dim, state.use_avg_pressure,
													 velocity_stiffness + nl_matrix, mixed_stiffness, pressure_stiffness,
													 total_matrix);
				time.stop();
				logger().debug("\tassembly time {}s", time.getElapsedTimeInSec());
				assembly_time += time.getElapsedTimeInSec();

				nlres = -(total_matrix * x) + rhs;
				for (int i : state.boundary_nodes)
					nlres[i] = 0;
				for (int i : skipping)
					nlres[i] = 0;
				nlres_norm = nlres.norm();

				logger().debug("\titer: {},  ||g||_2 = {}, ||step|| = {}\n",
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
	} // namespace solver
} // namespace polyfem
