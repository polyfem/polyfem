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
	class BodyForm;
	class ElasticForm;
	class Form;
	class InertiaForm;
	class MixedAssemblerForm;
	class NLProblem;
	class StackedForm;
} // namespace polyfem::solver

namespace polyfem::time_integrator
{
	class ImplicitTimeIntegrator;
} // namespace polyfem::time_integrator

namespace polyfem::varform
{
	class ThermoElasticVarForm : public NonlinearElasticVarForm
	{
	public:
		std::string name() const override { return "ThermoElastic"; }

		void init(const std::string &formulation, const Units &units, const json &args, const std::string &out_path) override;

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
		void build_rhs_assembler() override;

		void read_material_space_ids(const json &args);
		json elastic_material_args() const;
		json time_integrator_args(const int fe_space_id) const;

		void build_displacement_boundary(mesh::Mesh &mesh);
		void build_temperature_basis(mesh::Mesh &mesh, const bool iso_parametric, const json &args);
		void build_temperature_boundary(mesh::Mesh &mesh);
		void build_forms(Eigen::MatrixXd &solution, const double t);
		void update_transient_form_weights();
		void solve_nonlinear_step(const int step, Eigen::MatrixXd &solution);

		void initial_temperature_solution(Eigen::MatrixXd &solution) const;
		void split_solution(
			const Eigen::MatrixXd &solution,
			Eigen::MatrixXd &displacement,
			Eigen::MatrixXd &temperature) const;
		Eigen::MatrixXd stacked_solution(
			const Eigen::MatrixXd &displacement,
			const Eigen::MatrixXd &temperature) const;

		int displacement_ndof() const { return space_.ndof(); }
		int temperature_ndof() const { return temperature_space_.ndof(); }
		int total_ndof() const { return displacement_ndof() + temperature_ndof(); }

		FESpace temperature_space_;
		VarFormBoundaryState temperature_boundary_;
		std::shared_ptr<assembler::Problem> temperature_problem_;

		assembler::AssemblyValsCache temperature_ass_vals_cache_;
		assembler::AssemblyValsCache temperature_mass_ass_vals_cache_;
		assembler::AssemblyValsCache temperature_pure_mass_ass_vals_cache_;

		std::shared_ptr<assembler::Assembler> temperature_assembler_;
		std::shared_ptr<assembler::MixedNLAssembler> thermoelastic_assembler_;
		std::shared_ptr<assembler::Mass> temperature_mass_assembler_;
		std::shared_ptr<assembler::HRZMass> temperature_pure_mass_assembler_;
		std::shared_ptr<assembler::RhsAssembler> temperature_rhs_assembler_;
		std::shared_ptr<assembler::Density> temperature_rhs_density_;

		StiffnessMatrix temperature_mass_;
		StiffnessMatrix temperature_pure_mass_;
		StiffnessMatrix stacked_lumped_mass_;
		Eigen::MatrixXd temperature_rhs_;

		std::shared_ptr<time_integrator::ImplicitTimeIntegrator> temperature_time_integrator_;

		std::shared_ptr<solver::ElasticForm> temperature_form_;
		std::shared_ptr<solver::MixedAssemblerForm> thermoelastic_form_;
		std::shared_ptr<solver::BodyForm> temperature_body_form_;
		std::shared_ptr<solver::InertiaForm> temperature_inertia_form_;
		std::shared_ptr<solver::StackedForm> stacked_form_;

		int displacement_space_id_ = -1;
		int temperature_space_id_ = -1;
		std::string elastic_formulation_ = "NeoHookean";
	};
} // namespace polyfem::varform
