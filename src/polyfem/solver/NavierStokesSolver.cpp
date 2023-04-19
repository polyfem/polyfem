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

		void NavierStokesSolver::minimize(
			const int n_bases,
			const int n_pressure_bases,
			const std::vector<basis::ElementBases> &bases,
			const std::vector<basis::ElementBases> &pressure_bases,
			const std::vector<basis::ElementBases> &gbases,
			const assembler::Assembler &velocity_stokes_assembler,
			assembler::NavierStokesVelocity &velocity_assembler,
			const assembler::MixedAssembler &mixed_assembler,
			const assembler::Assembler &pressure_assembler,
			const assembler::AssemblyValsCache &ass_vals_cache,
			const assembler::AssemblyValsCache &pressure_ass_vals_cache,
			const std::vector<int> &boundary_nodes,
			const bool use_avg_pressure,
			const int problem_dim,
			const bool is_volume,
			const Eigen::MatrixXd &rhs, Eigen::VectorXd &x)
		{
			assert(velocity_assembler.name() == "NavierStokes");

			auto solver = LinearSolver::create(solver_type, precond_type);
			solver->setParameters(solver_param["linear"]);
			logger().debug("\tinternal solver {}", solver->name());

			const int precond_num = problem_dim * n_bases;

			igl::Timer time;

			time.start();
			StiffnessMatrix stoke_stiffness;
			StiffnessMatrix velocity_stiffness, mixed_stiffness, pressure_stiffness;
			velocity_stokes_assembler.assemble(is_volume, n_bases, bases, gbases, ass_vals_cache, velocity_stiffness);
			mixed_assembler.assemble(is_volume, n_pressure_bases, n_bases, pressure_bases, bases, gbases, pressure_ass_vals_cache, ass_vals_cache, mixed_stiffness);
			pressure_assembler.assemble(is_volume, n_pressure_bases, pressure_bases, gbases, pressure_ass_vals_cache, pressure_stiffness);

			AssemblerUtils::merge_mixed_matrices(n_bases, n_pressure_bases, problem_dim, use_avg_pressure,
												 velocity_stiffness, mixed_stiffness, pressure_stiffness,
												 stoke_stiffness);
			time.stop();
			stokes_matrix_time = time.getElapsedTimeInSec();
			logger().debug("\tStokes matrix assembly time {}s", time.getElapsedTimeInSec());

			time.start();

			logger().info("{}...", solver->name());

			Eigen::VectorXd b = rhs;
			dirichlet_solve(*solver, stoke_stiffness, b, boundary_nodes, x, precond_num, "", false, true, use_avg_pressure);
			// solver->get_info(solver_info);
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
			it += minimize_aux(true, skipping,
							   n_bases,
							   n_pressure_bases,
							   bases,
							   gbases,
							   velocity_assembler,
							   ass_vals_cache,
							   boundary_nodes,
							   use_avg_pressure,
							   problem_dim,
							   is_volume,
							   velocity_stiffness, mixed_stiffness, pressure_stiffness, b, 1e-3, solver, nlres_norm, x);
			it += minimize_aux(false, skipping,
							   n_bases,
							   n_pressure_bases,
							   bases,
							   gbases,
							   velocity_assembler,
							   ass_vals_cache,
							   boundary_nodes,
							   use_avg_pressure,
							   problem_dim,
							   is_volume,
							   velocity_stiffness, mixed_stiffness, pressure_stiffness, b, gradNorm, solver, nlres_norm, x);

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
			bool is_picard,
			const std::vector<int> &skipping,
			const int n_bases,
			const int n_pressure_bases,
			const std::vector<basis::ElementBases> &bases,
			const std::vector<basis::ElementBases> &gbases,
			assembler::NavierStokesVelocity &velocity_assembler,
			const assembler::AssemblyValsCache &ass_vals_cache,
			const std::vector<int> &boundary_nodes,
			const bool use_avg_pressure,
			const int problem_dim,
			const bool is_volume,
			const StiffnessMatrix &velocity_stiffness, const StiffnessMatrix &mixed_stiffness, const StiffnessMatrix &pressure_stiffness,
			const Eigen::VectorXd &rhs, const double grad_norm,
			std::unique_ptr<LinearSolver> &solver, double &nlres_norm,
			Eigen::VectorXd &x)
		{
			igl::Timer time;
			// const int problem_dim = problem->is_scalar() ? 1 : mesh->dimension();
			const int precond_num = problem_dim * n_bases;

			StiffnessMatrix nl_matrix;
			StiffnessMatrix total_matrix;
			SparseMatrixCache mat_cache;

			time.start();
			velocity_assembler.set_picard(true);
			velocity_assembler.assemble_hessian(is_volume, n_bases, false, bases, gbases, ass_vals_cache, 0, x, Eigen::MatrixXd(), mat_cache, nl_matrix);
			AssemblerUtils::merge_mixed_matrices(n_bases, n_pressure_bases, problem_dim, use_avg_pressure,
												 velocity_stiffness + nl_matrix, mixed_stiffness, pressure_stiffness,
												 total_matrix);
			time.stop();
			assembly_time = time.getElapsedTimeInSec();
			logger().debug("\tNavier Stokes assembly time {}s", time.getElapsedTimeInSec());

			Eigen::VectorXd nlres = -(total_matrix * x) + rhs;
			for (int i : boundary_nodes)
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
				if (!is_picard)
				{
					velocity_assembler.set_picard(false);
					velocity_assembler.assemble_hessian(is_volume, n_bases, false, bases, gbases, ass_vals_cache, 0, x, Eigen::MatrixXd(), mat_cache, nl_matrix);
					AssemblerUtils::merge_mixed_matrices(n_bases, n_pressure_bases, problem_dim, use_avg_pressure,
														 velocity_stiffness + nl_matrix, mixed_stiffness, pressure_stiffness,
														 total_matrix);
				}
				dirichlet_solve(*solver, total_matrix, nlres, boundary_nodes, dx, precond_num, "", false, true, use_avg_pressure);
				// for (int i : boundary_nodes)
				// 	dx[i] = 0;
				time.stop();
				inverting_time += time.getElapsedTimeInSec();
				logger().debug("\tinverting time {}s", time.getElapsedTimeInSec());
				logger().debug("\tinverting error: {}", (total_matrix * dx - nlres).norm());

				x += dx;
				// TODO check for nans

				time.start();
				velocity_assembler.set_picard(true);
				velocity_assembler.assemble_hessian(is_volume, n_bases, false, bases, gbases, ass_vals_cache, 0, x, Eigen::MatrixXd(), mat_cache, nl_matrix);
				AssemblerUtils::merge_mixed_matrices(n_bases, n_pressure_bases, problem_dim, use_avg_pressure,
													 velocity_stiffness + nl_matrix, mixed_stiffness, pressure_stiffness,
													 total_matrix);
				time.stop();
				logger().debug("\tassembly time {}s", time.getElapsedTimeInSec());
				assembly_time += time.getElapsedTimeInSec();

				nlres = -(total_matrix * x) + rhs;
				for (int i : boundary_nodes)
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
