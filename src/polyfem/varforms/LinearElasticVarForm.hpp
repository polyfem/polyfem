#pragma once

#include <polyfem/varforms/ElasticVarForm.hpp>

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

	private:
		void reset() override;

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

		std::shared_ptr<solver::ElasticForm> elastic_form;
		std::shared_ptr<solver::BodyForm> body_form;
		std::shared_ptr<solver::InertiaForm> inertia_form;

		std::shared_ptr<time_integrator::ImplicitTimeIntegrator> time_integrator;
	};
} // namespace polyfem::varform
