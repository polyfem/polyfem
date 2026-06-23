#pragma once

#include <polyfem/varforms/ElasticVarForm.hpp>

#include <polyfem/assembler/Assembler.hpp>
#include <polyfem/assembler/RhsAssembler.hpp>

namespace polysolve::linear
{
	class Solver;
}

namespace polyfem::varform
{
	class IncompressibleElasticVarForm : public ElasticVarForm
	{
	public:
		std::string name() const override { return "IncompressibleElastic"; }

		void init(const std::string &formulation, const Units &units, const json &args, const std::string &out_path) override;
		void save_json(const Eigen::MatrixXd &solution, std::ostream &out) const override;
		io::OutStatsData compute_errors(const Eigen::MatrixXd &solution) override;

		std::vector<io::OutputField> output_fields(
			const io::OutputSample &sample,
			const Eigen::MatrixXd &solution,
			const io::OutputFieldOptions &options) const override;

	private:
		void reset() override;
		void load_mesh(const mesh::Mesh &mesh, const json &args) override;
		void build_basis(mesh::Mesh &mesh, const bool iso_parametric, const json &args) override;
		void assemble_rhs(const mesh::Mesh &mesh) override;
		void assemble_mass_mat(const mesh::Mesh &mesh, const json &args) override;
		void solve_problem(Eigen::MatrixXd &sol) override;

		int primary_ndof() const;
		int stacked_ndof() const;
		void prepare_initial_solution(Eigen::MatrixXd &sol) const;
		void split_solution(const Eigen::MatrixXd &stacked, Eigen::MatrixXd &primary, Eigen::MatrixXd &pressure) const;
		void build_stiffness_mat(StiffnessMatrix &stiffness);
		void solve_linear_system(
			const std::unique_ptr<polysolve::linear::Solver> &solver,
			StiffnessMatrix &A,
			Eigen::VectorXd &b,
			const bool compute_spectrum,
			Eigen::MatrixXd &sol);
		void solve_static_linear(Eigen::MatrixXd &sol);
		void solve_transient_linear(Eigen::MatrixXd &sol);

		void build_rhs_assembler() override;

		FESpace pressure_space_;
		VarFormBoundaryState pressure_boundary_;

		std::shared_ptr<assembler::MixedAssembler> mixed_assembler_ = nullptr;
		std::shared_ptr<assembler::Assembler> pressure_assembler_ = nullptr;
		assembler::AssemblyValsCache pressure_ass_vals_cache_;

		std::shared_ptr<time_integrator::ImplicitTimeIntegrator> time_integrator;
	};
} // namespace polyfem::varform
