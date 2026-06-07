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

		std::vector<io::OutputField> output_fields(
			const io::OutputSample &sample,
			const Eigen::MatrixXd &solution,
			const io::OutputFieldOptions &options) const override;

	private:
		void save_step_state(const double t0, const double dt, const int t, const Eigen::MatrixXd &sol) const override;

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
