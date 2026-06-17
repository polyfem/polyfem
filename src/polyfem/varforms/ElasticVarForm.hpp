#pragma once

#include <polyfem/varforms/VarForm.hpp>

namespace polyfem::mesh
{
	class Obstacle;
}

namespace polyfem::solver
{
	class ContactForm;
	class Form;
} // namespace polyfem::solver

namespace polyfem::time_integrator
{
	class ImplicitTimeIntegrator;
}

namespace polyfem::varform
{
	class ElasticVarForm : public VarForm
	{
		friend class polyfem::test::VarFormTestAccess;

	public:
		void init(const std::string &formulation, const Units &units, const json &args, const std::string &out_path) override;

		void save_json(const Eigen::MatrixXd &solution, std::ostream &out) const override;
		void export_data(const Eigen::MatrixXd &solution) const override;
		io::OutputSpace output_space() const override;
		io::OutStatsData compute_errors(const Eigen::MatrixXd &solution) override;

	protected:
		void reset() override;
		void load_mesh(const mesh::Mesh &mesh, const json &args) override;
		void build_basis(mesh::Mesh &mesh, const bool iso_parametric, const json &args) override;
		void assemble_rhs(const mesh::Mesh &mesh) override;
		void assemble_mass_mat(const mesh::Mesh &mesh, const json &args) override;
		void build_rhs_assembler() override;

		void initial_velocity(Eigen::MatrixXd &velocity) const;
		void initial_acceleration(Eigen::MatrixXd &acceleration) const;
		void initial_elastic_solution(Eigen::MatrixXd &solution) const;
		QuadratureOrders elastic_boundary_samples() const;
		std::vector<int> elastic_primitive_to_node() const;
		std::vector<int> elastic_node_to_primitive() const;
		void build_mesh_matrices(Eigen::MatrixXd &V, Eigen::MatrixXi &F) const;
		void save_elastic_step_state(
			const double t0,
			const double dt,
			const int t,
			const time_integrator::ImplicitTimeIntegrator *time_integrator) const;
		std::vector<io::OutputField> elastic_output_fields(
			const io::OutputSample &sample,
			const Eigen::MatrixXd &solution,
			const io::OutputFieldOptions &options,
			const mesh::Obstacle *obstacle,
			const time_integrator::ImplicitTimeIntegrator *time_integrator,
			const std::vector<std::pair<std::string, std::shared_ptr<solver::Form>>> &named_forms,
			const solver::Form *elastic_form,
			const solver::ContactForm *contact_form = nullptr) const;
		void append_primary_output_fields(
			std::vector<io::OutputField> &fields,
			const io::OutputSample &sample,
			const Eigen::MatrixXd &solution,
			const io::OutputFieldOptions &options,
			const mesh::Obstacle *obstacle = nullptr) const;
		Eigen::MatrixXd displaced_output_normals(
			const io::OutputSample &sample,
			const Eigen::MatrixXd &solution) const;

		virtual int n_obstacle_vertices() const { return 0; }

		FESpace space_;
		VarFormBoundaryState boundary_;

		assembler::AssemblyValsCache ass_vals_cache_;
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

		double t0 = 0;
		int time_steps = 0;
		double dt = 0;
	};
} // namespace polyfem::varform
