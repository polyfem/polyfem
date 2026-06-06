#pragma once

#include <polyfem/varforms/ElasticVarForm.hpp>

#include <memory>

namespace polysolve::linear
{
	class Solver;
}

namespace polyfem::varform
{
	class LinearElasticVarForm : public ElasticVarForm, public VarFormMatrixTestAccess
	{
	public:
		std::string name() const override { return "LinearElastic"; }
		VarFormDebugData debug_data() const override;
		void build_stiffness_mat_debug(StiffnessMatrix &stiffness) override;
		const StiffnessMatrix *mass_matrix_debug() const override;

	private:
		void solve_problem(Eigen::MatrixXd &sol) override;
		void init_linear_solve(Eigen::MatrixXd &sol, const double t);
		void build_stiffness_mat(StiffnessMatrix &stiffness);
		void solve_linear_system(
			const std::unique_ptr<polysolve::linear::Solver> &solver,
			StiffnessMatrix &A,
			Eigen::VectorXd &b,
			const bool compute_spectrum,
			Eigen::MatrixXd &sol);
		void solve_static_linear(Eigen::MatrixXd &sol);
		void solve_transient_linear(Eigen::MatrixXd &sol);
	};
} // namespace polyfem::varform
