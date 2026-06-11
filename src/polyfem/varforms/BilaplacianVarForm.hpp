#pragma once

#include <polyfem/varforms/ScalarVarForm.hpp>

#include <polyfem/assembler/Assembler.hpp>
#include <polyfem/assembler/RhsAssembler.hpp>

namespace polysolve::linear
{
	class Solver;
}

namespace polyfem::varform
{
	class BilaplacianVarForm : public ScalarVarForm
	{
	public:
		std::string name() const override { return "Bilaplacian"; }

		BilaplacianVarForm(const std::string &formulation, const Units &units, const json &args, const std::string &out_path);
		void save_json(const Eigen::MatrixXd &solution, std::ostream &out) const override;

		std::vector<io::OutputField> output_fields(
			const io::OutputSample &sample,
			const Eigen::MatrixXd &solution,
			const io::OutputFieldOptions &options) const override;

	private:
		void load_mesh(const mesh::Mesh &mesh, const json &args) override;
		void build_basis(mesh::Mesh &mesh, const bool iso_parametric, const json &args) override;
		void assemble_rhs(const mesh::Mesh &mesh, const json &args) override;
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
			rhs_assembler = build_rhs_assembler(scalar_space.n_bases, scalar_space.bases, scalar_caches.mass);
		}

		std::shared_ptr<assembler::RhsAssembler> build_rhs_assembler(
			const int n_bases,
			const std::vector<basis::ElementBases> &bases,
			const assembler::AssemblyValsCache &ass_vals_cache);

		std::shared_ptr<assembler::MixedAssembler> mixed_assembler = nullptr;
		std::shared_ptr<assembler::Assembler> pressure_assembler = nullptr;
		BilaplacianSpaces bilaplacian_spaces;
		assembler::AssemblyValsCache pressure_ass_vals_cache;
		bool use_avg_pressure = true;
		std::shared_ptr<time_integrator::ImplicitTimeIntegrator> time_integrator;
	};
} // namespace polyfem::varform
