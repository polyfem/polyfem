#pragma once

#include <polyfem/varforms/FluidVarForm.hpp>

namespace polyfem::solver
{
	class ElasticForm;
	class NavierStokesFSIAveragePressureForm;
	class NavierStokesFSIForm;
} // namespace polyfem::solver

namespace polyfem::varform
{
	class NavierStokesFSIVarForm : public FluidVarForm
	{
		friend class polyfem::test::VarFormTestAccess;

	public:
		std::string name() const override { return "NavierStokesFSI"; }
		void init(const std::string &formulation, const Units &units, const json &args, const std::string &out_path) override;
		std::vector<io::OutputField> output_fields(
			const io::OutputSample &sample,
			const Eigen::MatrixXd &solution,
			const io::OutputFieldOptions &options) const override;

	protected:
		void reset() override;
		void load_mesh(const mesh::Mesh &mesh, const json &args) override;
		void build_basis(mesh::Mesh &mesh, bool iso_parametric, const json &args) override;
		void assemble_rhs(const mesh::Mesh &mesh) override;
		void assemble_mass_mat(const mesh::Mesh &mesh, const json &args) override;
		void build_rhs_assembler() override;
		void solve_problem(Eigen::MatrixXd &sol) override;

	private:
		int mesh_displacement_ndof() const;
		int total_ndof() const;
		int pressure_offset() const { return primary_ndof(); }
		int mesh_displacement_offset() const { return primary_ndof() + pressure_space_.n_bases; }
		int average_pressure_offset() const { return mesh_displacement_offset() + mesh_displacement_ndof(); }
		json mesh_material_args() const;
		json time_integrator_args(int fe_space_id) const;
		void build_mesh_displacement_boundary(mesh::Mesh &mesh);
		void prepare_fsi_initial_solution(Eigen::MatrixXd &sol) const;
		void build_forms(Eigen::MatrixXd &sol, double t);
		void solve_nonlinear_step(int step, Eigen::MatrixXd &sol);
		void update_transient_form_weights();
		void save_mesh_integrator_state(int step) const;

		int mesh_displacement_space_id_ = -1;
		std::string mesh_elastic_formulation_ = "NeoHookean";
		FESpace mesh_displacement_space_;
		VarFormBoundaryState mesh_displacement_boundary_;
		std::shared_ptr<assembler::Problem> mesh_displacement_problem_;
		assembler::AssemblyValsCache mesh_displacement_ass_vals_cache_;
		assembler::AssemblyValsCache mesh_displacement_mass_ass_vals_cache_;
		assembler::AssemblyValsCache mesh_displacement_pure_mass_ass_vals_cache_;
		std::shared_ptr<assembler::Assembler> mesh_elastic_assembler_;
		std::shared_ptr<assembler::Mass> mesh_mass_assembler_;
		std::shared_ptr<assembler::HRZMass> mesh_pure_mass_assembler_;
		std::shared_ptr<assembler::RhsAssembler> mesh_rhs_assembler_;
		Eigen::MatrixXd mesh_rhs_;
		Eigen::MatrixXd fluid_zero_rhs_;
		StiffnessMatrix mesh_pure_mass_;

		std::vector<std::shared_ptr<assembler::MultiSpacesNLAssembler>> ale_assemblers_;
		std::shared_ptr<time_integrator::ImplicitTimeIntegrator> mesh_displacement_time_integrator_;
		std::vector<std::shared_ptr<solver::Form>> fsi_forms_;
		std::vector<std::shared_ptr<solver::AugmentedLagrangianForm>> fsi_al_forms_;
		std::shared_ptr<solver::NLProblem> fsi_problem_;
		std::shared_ptr<solver::NavierStokesFSIForm> ale_form_;
		std::shared_ptr<solver::StackedForm> auxiliary_form_;
		std::shared_ptr<solver::ElasticForm> mesh_elastic_form_;
		std::shared_ptr<solver::BodyForm> fluid_neumann_form_;
		std::shared_ptr<solver::BodyForm> mesh_body_form_;
		std::shared_ptr<solver::NavierStokesFSIAveragePressureForm> average_pressure_form_;
	};
} // namespace polyfem::varform
