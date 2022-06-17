#pragma once

#include <polyfem/Common.hpp>
#include <polyfem/State.hpp>

#include <polysolve/LinearSolver.hpp>

#include <polyfem/utils/Logger.hpp>

#include <memory>

namespace polyfem
{
	namespace solver
	{
		class TransientNavierStokesSolver
		{
		public:
			TransientNavierStokesSolver(const json &solver_param);

			void minimize(const State &state, const double beta_dt, const Eigen::VectorXd &prev_sol,
						  const StiffnessMatrix &velocity_stiffness, const StiffnessMatrix &mixed_stiffness, const StiffnessMatrix &pressure_stiffness,
						  const StiffnessMatrix &velocity_mass,
						  const Eigen::MatrixXd &rhs, Eigen::VectorXd &x);
			void getInfo(json &params)
			{
				params = solver_info;
			}

			int error_code() const { return 0; }

		private:
			int minimize_aux(const std::string &formulation, const std::vector<int> &skipping, const State &state,
							 const StiffnessMatrix &velocity_stiffness, const StiffnessMatrix &mixed_stiffness, const StiffnessMatrix &pressure_stiffness,
							 const StiffnessMatrix &velocity_mass,
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

			bool
			has_nans(const polyfem::StiffnessMatrix &hessian);
		};
	} // namespace solver
} // namespace polyfem
