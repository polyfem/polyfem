#pragma once

#include <polyfem/Common.hpp>
#include <polyfem/basis/ElementBases.hpp>
#include <polyfem/assembler/NavierStokes.hpp>
#include <polyfem/assembler/AssemblyValsCache.hpp>

#include <polysolve/LinearSolver.hpp>

#include <polyfem/utils/Logger.hpp>

#include <memory>

namespace polyfem
{
	namespace solver
	{
		class NavierStokesSolver
		{
		public:
			NavierStokesSolver(const json &solver_param);

			void minimize(const int n_bases,
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
						  const Eigen::MatrixXd &rhs,
						  Eigen::VectorXd &x);

			void get_info(json &params)
			{
				params = solver_info;
			}

			int error_code() const { return 0; }

		private:
			int minimize_aux(
				const bool is_picard,
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
				std::unique_ptr<polysolve::LinearSolver> &solver, double &nlres_norm,
				Eigen::VectorXd &x);

			const json solver_param;
			const std::string solver_type;
			const std::string precond_type;

			double gradNorm;
			int iterations;

			json solver_info;

			json internal_solver = json::array();

			double assembly_time;
			double inverting_time;
			double stokes_matrix_time;
			double stokes_solve_time;

			bool has_nans(const polyfem::StiffnessMatrix &hessian);
		};
	} // namespace solver
} // namespace polyfem
