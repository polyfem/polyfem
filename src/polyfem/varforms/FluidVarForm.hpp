#pragma once

#include "polyfem/varforms/FESpace.hpp"
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

		// DESIGN NOTE:
		// The idea is that base VarForm will provide pure virtual method for derive class to implement.
		// In other language like Java, VarForm will be a so-called "Interface". Instead of complex method
		// overwrite via inheritance, we de-duplicate code via focused free function helper in standalone file.

		void build_fe_space(mesh::Mesh &mesh, const json &args) override;
		void build_assembler_cache(const mesh::Mesh &mesh, const json &args) override;
		void build_boundary_condition(mesh::Mesh &mesh, const json &args) override;
		void build_solution_layout() override;

		void assemble_rhs(const mesh::Mesh &mesh, const json &args) override;
		void assemble_mass_mat(const mesh::Mesh &mesh, const json &args) override;

		// LEGACY COMPATIBILITY METHODS.
		FESpace &legacy_primary_space_dont_use() override { return velocity_space; }
		const FESpace &legacy_primary_space_dont_use() const override { return velocity_space; }
		std::shared_ptr<GeometryMapping> &legacy_primary_geometry_dont_use() override { return velocity_space.geometry; }
		const std::shared_ptr<GeometryMapping> &legacy_primary_geometry_dont_use() const override { return velocity_space.geometry; }
		AssemblyCaches &legacy_primary_caches_dont_use() override { return velocity_caches; }
		const AssemblyCaches &legacy_primary_caches_dont_use() const override { return velocity_caches; }

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

		// DESIGN NOTE: Each var form should be able build it's finite element space and specify solution layout.
		//
		// Ex. For thermal elasticity
		// FESpace disp_space;
		// FESpace temp_space;
		// Solution layout: [displacement...] [temperature...]
		//
		// Ex. For fluid elasticity
		// FESpace disp_space; // for solid
		// FESpace velocity_space; // for fluid
		// FESpace pressure_space; // for fluid
		// Solution layout: [displacement...] [velocity...] [pressure...]
		//
		// There will be no primary/aux space and we are not limited to two spaces.

		FESpace velocity_space;
		FESpace pressure_space;

		// DESIGN NOTE:
		// For fluid, the layout is:
		// [velocity...] [pressure...] [optional avg pressure lagrange multiplier]
		//
		// For simple varform with only one solution block, they can skip this class member.
		SolutionLayout solution_layout;

		AssemblyCaches velocity_caches;
		VarFormBoundaryState boundary;
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
