#pragma once

#include <polyfem/varforms/ElasticVarForm.hpp>

#include <memory>

namespace polysolve::linear
{
	class Solver;
}

namespace polyfem::varform
{
	class ScalarVarForm : public ElasticVarForm
	{
	public:
		std::string name() const override { return "Scalar"; }

		void init(const std::string &formulation, const Units &units, const json &args, const std::string &out_path) override;

		io::OutputState output_state() const override;
		std::vector<io::OutputField> output_fields(
			const io::OutputSample &sample,
			const Eigen::MatrixXd &solution,
			const io::OutputFieldOptions &options) const override;

	private:
		void solve_problem(Eigen::MatrixXd &sol) override;
		void build_stiffness_mat(StiffnessMatrix &stiffness) override;
		void solve_linear_system(
			const std::unique_ptr<polysolve::linear::Solver> &solver,
			StiffnessMatrix &A,
			Eigen::VectorXd &b,
			const bool compute_spectrum,
			Eigen::MatrixXd &sol);
		void solve_static(Eigen::MatrixXd &sol);
		void solve_transient(Eigen::MatrixXd &sol);
	};
} // namespace polyfem::varform
