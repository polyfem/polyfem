#pragma once

#include <polyfem/varforms/ElasticVarForm.hpp>
#include <polyfem/solver/SolveData.hpp>

#include <memory>

namespace polysolve::linear
{
	class Solver;
}

namespace polyfem::solver
{
	class BodyForm;
	class ElasticForm;
	class InertiaForm;
} // namespace polyfem::solver

namespace polyfem::varform
{
	class LinearElasticVarForm : public ElasticVarForm
	{
		friend class polyfem::test::VarFormTestAccess;

	public:
		std::string name() const override { return "LinearElastic"; }

		std::vector<io::OutputField> output_fields(
			const io::OutputSample &sample,
			const Eigen::MatrixXd &solution,
			const io::OutputFieldOptions &options) const override;

	protected:
		void reset() override;

		void solve_problem(
			Eigen::MatrixXd &sol,
			const InitialConditionOverride *initial_condition_override,
			const ForwardStepCallback &post_step) override;
		void init_linear_solve(Eigen::MatrixXd &sol, const double t, const InitialConditionOverride *initial_condition_override);
		void build_stiffness_mat(StiffnessMatrix &stiffness);
		void solve_linear_system(
			const std::unique_ptr<polysolve::linear::Solver> &solver,
			StiffnessMatrix &A,
			Eigen::VectorXd &b,
			const bool compute_spectrum,
			Eigen::MatrixXd &sol);
		void solve_static_linear(Eigen::MatrixXd &sol, const ForwardStepCallback &post_step);
		void solve_transient_linear(Eigen::MatrixXd &sol, const ForwardStepCallback &post_step);

		solver::SolveData solve_data_;
	};
} // namespace polyfem::varform
