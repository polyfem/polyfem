#pragma once

#include <polyfem/varforms/VarForm.hpp>

#include <polyfem/assembler/Assembler.hpp>
#include <polyfem/assembler/RhsAssembler.hpp>

namespace polysolve::linear
{
	class Solver;
}

namespace polyfem::varform
{
	class BilaplacianVarForm : public VarForm
	{
		friend class polyfem::test::VarFormTestAccess;

	public:
		std::string name() const override { return "Bilaplacian"; }

		void init(const std::string &formulation, const Units &units, const json &args, const std::string &out_path) override;
		void save_json(const Eigen::MatrixXd &solution, std::ostream &out) const override;
		void export_data(const Eigen::MatrixXd &solution) const override;
		io::OutputSpace output_space() const override;
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

		void build_rhs_assembler() override
		{
			rhs_assembler_ = build_rhs_assembler(space_.n_bases, space_.basis_list(), mass_ass_vals_cache_);
		}

		std::shared_ptr<assembler::RhsAssembler> build_rhs_assembler(
			const int n_bases,
			const std::vector<basis::ElementBases> &bases,
			const assembler::AssemblyValsCache &ass_vals_cache);

		FESpace space_;
		FESpace pressure_space_;

		VarFormBoundaryState boundary_;
		VarFormBoundaryState pressure_boundary_;

		assembler::AssemblyValsCache ass_vals_cache_;
		assembler::AssemblyValsCache pressure_ass_vals_cache_;
		assembler::AssemblyValsCache mass_ass_vals_cache_;
		assembler::AssemblyValsCache pure_mass_ass_vals_cache_;

		std::shared_ptr<assembler::RhsAssembler> rhs_assembler_;

		StiffnessMatrix mass_;
		StiffnessMatrix pure_mass_;

		double avg_mass_ = 0;
		Eigen::MatrixXd rhs_;

		std::shared_ptr<assembler::Assembler> primary_assembler_ = nullptr;
		std::shared_ptr<assembler::Mass> mass_assembler_ = nullptr;
		std::shared_ptr<assembler::HRZMass> pure_mass_assembler_ = nullptr;
		std::shared_ptr<assembler::MixedAssembler> mixed_assembler_ = nullptr;
		std::shared_ptr<assembler::Assembler> pressure_assembler_ = nullptr;

		double t0 = 0;
		int time_steps = 0;
		double dt = 0;
		std::shared_ptr<time_integrator::ImplicitTimeIntegrator> time_integrator;
	};
} // namespace polyfem::varform
