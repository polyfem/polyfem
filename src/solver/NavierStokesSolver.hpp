#pragma once

#include <polyfem/Common.hpp>
#include <polyfem/State.hpp>

#include <polyfem/Logger.hpp>

namespace polyfem
{

class NavierStokesSolver
{
public:
	NavierStokesSolver(double viscosity, const json &solver_param, const json &problem_params, const std::string &solver_type, const std::string &precond_type);

	void minimize(const State &state, const Eigen::MatrixXd &rhs, Eigen::VectorXd &x);
	void getInfo(json &params)
	{
		params = solver_info;
	}

	int error_code() const { return 0; }

private:
	double viscosity;

	const json solver_param;
	const std::string solver_type;
	const std::string precond_type;

	double gradNorm;
	int iterations;

	json solver_info;
	json problem_params;

	json internal_solver = json::array();

	double assembly_time;
	double inverting_time;
	double stokes_matrix_time;
	double stokes_solve_time;

	bool
	has_nans(const polyfem::StiffnessMatrix &hessian);
};
} // namespace polyfem
