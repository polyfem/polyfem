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
		FluidVarForm(const std::string &formulation, const Units &units, const json &args, const std::string &out_path);
		void save_json(const Eigen::MatrixXd &solution, std::ostream &out) const override;

		std::vector<io::OutputField> output_fields(
			const io::OutputSample &sample,
			const Eigen::MatrixXd &solution,
			const io::OutputFieldOptions &options) const override;

	protected:
		void load_mesh(const mesh::Mesh &mesh, const json &args) override;
		void build_basis(mesh::Mesh &mesh, const json &args) override;
		void assemble_rhs(const mesh::Mesh &mesh, const json &args) override;
		void assemble_mass_mat(const mesh::Mesh &mesh, const json &args) override;
		FESpace &primary_space() override { return velocity_space; }
		const FESpace &primary_space() const override { return velocity_space; }
		std::shared_ptr<GeometryMapping> &primary_geometry() override { return geometry_mapping; }
		const std::shared_ptr<GeometryMapping> &primary_geometry() const override { return geometry_mapping; }
		AssemblyCaches &primary_caches() override { return velocity_caches; }
		const AssemblyCaches &primary_caches() const override { return velocity_caches; }
		VarFormBoundaryState &boundary_state() override { return boundary; }
		const VarFormBoundaryState &boundary_state() const override { return boundary; }

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
		std::shared_ptr<GeometryMapping> geometry_mapping = std::make_shared<GeometryMapping>();
		FESpace velocity_space;
		AssemblyCaches velocity_caches;
		VarFormBoundaryState boundary;
		FluidSpaces fluid_spaces;
		assembler::AssemblyValsCache pressure_ass_vals_cache;
		bool use_avg_pressure = true;
		std::shared_ptr<time_integrator::ImplicitTimeIntegrator> time_integrator;
	};

	class StokesVarForm : public FluidVarForm
	{
	public:
		using FluidVarForm::FluidVarForm;
		std::string name() const override { return "Stokes"; }

	private:
		void solve_problem(Eigen::MatrixXd &sol) override;
		void solve_static_linear(Eigen::MatrixXd &sol);
		void solve_transient_linear(Eigen::MatrixXd &sol);
	};

	class NavierStokesVarForm : public FluidVarForm
	{
	public:
		using FluidVarForm::FluidVarForm;
		std::string name() const override { return "NavierStokes"; }

	private:
		void solve_problem(Eigen::MatrixXd &sol) override;
		void solve_static(Eigen::MatrixXd &sol);
		void solve_transient(Eigen::MatrixXd &sol);
	};
} // namespace polyfem::varform
