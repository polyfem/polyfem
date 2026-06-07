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
	class FluidVarForm : public VarForm
	{
	public:
		void init(const std::string &formulation, const Units &units, const json &args, const std::string &out_path) override;

		std::vector<io::OutputField> output_fields(
			const io::OutputSample &sample,
			const Eigen::MatrixXd &solution,
			const io::OutputFieldOptions &options) const override;

	protected:
		void reset() override;
		void load_mesh(const mesh::Mesh &mesh, const json &args) override;
		void build_basis(mesh::Mesh &mesh, const bool iso_parametric, const json &args) override;
		void assemble_rhs(const mesh::Mesh &mesh, const json &args) override;
		void assemble_mass_mat(const mesh::Mesh &mesh, const json &args) override;

		int primary_ndof() const;
		int pressure_block_size() const;
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

		void build_rhs_assembler() override;

		std::shared_ptr<assembler::MixedAssembler> mixed_assembler = nullptr;
		std::shared_ptr<assembler::Assembler> pressure_assembler = nullptr;
		std::vector<basis::ElementBases> pressure_bases;
		int n_pressure_bases = 0;
		std::shared_ptr<mesh::MeshNodes> pressure_mesh_nodes;
		assembler::AssemblyValsCache pressure_ass_vals_cache;
		std::vector<int> pressure_boundary_nodes;
		bool use_avg_pressure = true;
	};

	class StokesVarForm : public FluidVarForm
	{
	public:
		std::string name() const override { return "Stokes"; }

	private:
		void solve_problem(Eigen::MatrixXd &sol) override;
		void solve_static_linear(Eigen::MatrixXd &sol);
		void solve_transient_linear(Eigen::MatrixXd &sol);
	};

	class NavierStokesVarForm : public FluidVarForm
	{
	public:
		std::string name() const override { return "NavierStokes"; }

	private:
		void solve_problem(Eigen::MatrixXd &sol) override;
		void solve_static(Eigen::MatrixXd &sol);
		void solve_transient(Eigen::MatrixXd &sol);
	};
} // namespace polyfem::varform
