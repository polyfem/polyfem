#pragma once

#include <polyfem/varforms/NonlinearElasticVarForm.hpp>

namespace polysolve::linear
{
	class Solver;
}

namespace polyfem::assembler
{
	class Density;
	class GenericScalarProblem;
	class Laplacian;
	class MixedNLAssembler;
} // namespace polyfem::assembler

namespace polyfem::solver
{
	class AugmentedLagrangianForm;
	class Form;
	class MixedAssemblerForm;
	class NLProblem;
	class StackedForm;
} // namespace polyfem::solver

namespace polyfem::varform
{
	/// Variational form for the (displacement, growth) problem built on the
	/// GrowthElasticity mixed assembler. Structural clone of
	/// ThermoElasticVarForm with the transient machinery removed: Stage 1 of
	/// the growth build is static with the growth field theta prescribed
	/// (Dirichlet on the growth space); the growth block carries NO PDE of its
	/// own -- its stiffness comes from the K_gg quadrant of the growth
	/// correction plus the augmented-Lagrangian boundary forms, and with every
	/// growth DOF prescribed the reduced problem contains only displacements.
	///
	/// Deliberate deltas from the thermo precedent (each commented in the cpp):
	///  - static only: time-dependent configs are rejected with a clear error;
	///  - no growth-block ElasticForm / BodyForm / inertia / integrators;
	///  - the growth initial solution defaults to 1 (identity growth), because
	///    the zero default that is harmless for temperature is singular for
	///    theta (the assembler's positivity guard would throw on the first
	///    energy evaluation);
	///  - output fields "growth", "growth_gradient", and "growth_reaction" --
	///    the last is the growth-block gradient of the stacked energy at the
	///    converged full solution, i.e. the conjugate driving-force/reaction
	///    field of the formulation (the V3 deliverable).
	class GrowthElasticVarForm : public NonlinearElasticVarForm
	{
	public:
		std::string name() const override { return "GrowthElastic"; }

		void init(const std::string &formulation, const Units &units, const json &args, const std::string &out_path) override;

		io::OutStatsData compute_errors(const Eigen::MatrixXd &solution) override;

		void export_data(const Eigen::MatrixXd &solution) const override;

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
		void solve_problem(
			Eigen::MatrixXd &sol,
			const InitialConditionOverride *initial_condition_override,
			const ForwardStepCallback &post_step) override;
		void build_rhs_assembler() override;

		void read_material_space_ids(const json &args);
		json elastic_material_args() const;

		void build_displacement_boundary(mesh::Mesh &mesh);
		void build_growth_basis(mesh::Mesh &mesh, const bool iso_parametric, const json &args);
		void build_growth_boundary(mesh::Mesh &mesh);
		void build_forms(Eigen::MatrixXd &solution, const double t);
		void solve_nonlinear_step(const int step, Eigen::MatrixXd &solution);

		void initial_growth_solution(Eigen::MatrixXd &solution) const;
		void split_solution(
			const Eigen::MatrixXd &solution,
			Eigen::MatrixXd &displacement,
			Eigen::MatrixXd &growth) const;
		Eigen::MatrixXd stacked_solution(
			const Eigen::MatrixXd &displacement,
			const Eigen::MatrixXd &growth) const;

		int displacement_ndof() const { return space_.ndof(); }
		int growth_ndof() const { return growth_space_.ndof(); }
		int total_ndof() const { return displacement_ndof() + growth_ndof(); }

		FESpace growth_space_;
		VarFormBoundaryState growth_boundary_;
		std::shared_ptr<assembler::Problem> growth_problem_;

		assembler::AssemblyValsCache growth_ass_vals_cache_;
		assembler::AssemblyValsCache growth_mass_ass_vals_cache_;

		std::shared_ptr<assembler::Assembler> growth_assembler_;
		std::shared_ptr<assembler::MixedNLAssembler> growthelastic_assembler_;
		std::shared_ptr<assembler::Mass> growth_mass_assembler_;
		std::shared_ptr<assembler::RhsAssembler> growth_rhs_assembler_;
		std::shared_ptr<assembler::Density> growth_rhs_density_;

		StiffnessMatrix growth_mass_;
		StiffnessMatrix stacked_lumped_mass_;
		Eigen::MatrixXd growth_rhs_;

		std::shared_ptr<solver::MixedAssemblerForm> growth_coupling_form_;
		std::shared_ptr<solver::StackedForm> stacked_form_;

		/// Converged stacked (u, growth) solution cached at the end of
		/// solve_problem. The output pipeline does not reliably hand
		/// output_fields the stacked vector (sizes/ordering differ per export
		/// path), so growth sampling and the reaction evaluation use this
		/// authoritative copy whenever the passed solution is not
		/// total_ndof-sized.
		Eigen::MatrixXd converged_solution_;

		int displacement_space_id_ = -1;
		int growth_space_id_ = -1;
		std::string elastic_formulation_ = "MaterialSum";
	};
} // namespace polyfem::varform
