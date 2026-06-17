#pragma once

#include <polyfem/varforms/VarForm.hpp>

#include <memory>

namespace polysolve::linear
{
	class Solver;
}

namespace polyfem::varform
{
	class ScalarVarForm : public VarForm
	{
		friend class polyfem::test::VarFormTestAccess;

	public:
		std::string name() const override { return "Scalar"; }

		void init(const std::string &formulation, const Units &units, const json &args, const std::string &out_path) override;
		void save_json(const Eigen::MatrixXd &solution, std::ostream &out) const override;
		void export_data(const Eigen::MatrixXd &solution) const override;
		io::OutputSpace output_space() const override;
		io::OutStatsData compute_errors(const Eigen::MatrixXd &solution) override;

		std::vector<io::OutputField> output_fields(
			const io::OutputSample &sample,
			const Eigen::MatrixXd &solution,
			const io::OutputFieldOptions &options) const override;

	protected:
		void reset() override;
		void build_basis(mesh::Mesh &mesh, const bool iso_parametric, const json &args) override;
		void assemble_rhs(const mesh::Mesh &mesh) override;
		void assemble_mass_mat(const mesh::Mesh &mesh, const json &args) override;

	private:
		void build_scalar_rhs_assembler();
		void initial_scalar_solution(Eigen::MatrixXd &solution) const;

		void build_stiffness_mat(StiffnessMatrix &stiffness);

		void solve_problem(Eigen::MatrixXd &sol) override;
		void solve_linear_system(
			const std::unique_ptr<polysolve::linear::Solver> &solver,
			StiffnessMatrix &A,
			Eigen::VectorXd &b,
			const bool compute_spectrum,
			Eigen::MatrixXd &sol);
		void solve_static(Eigen::MatrixXd &sol);
		void solve_transient(Eigen::MatrixXd &sol);

		FESpace scalar_space_;
		VarFormBoundaryState scalar_boundary_;
		assembler::AssemblyValsCache scalar_ass_vals_cache_;
		assembler::AssemblyValsCache scalar_mass_ass_vals_cache_;
		assembler::AssemblyValsCache scalar_pure_mass_ass_vals_cache_;
		std::shared_ptr<assembler::RhsAssembler> scalar_rhs_assembler_;
		StiffnessMatrix scalar_mass_;
		StiffnessMatrix scalar_pure_mass_;
		double scalar_avg_mass_ = 0;
		Eigen::MatrixXd scalar_rhs_;

		std::shared_ptr<time_integrator::ImplicitTimeIntegrator> time_integrator;
	};
} // namespace polyfem::varform
