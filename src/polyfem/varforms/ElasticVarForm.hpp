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
	public:
		void init(const std::string &formulation, const Units &units, const json &args, const std::string &out_path) override;

		void save_json(const Eigen::MatrixXd &solution, std::ostream &out) const override;

	protected:
		void load_mesh(const mesh::Mesh &mesh, const json &args) override;

		void initial_velocity(Eigen::MatrixXd &velocity) const;
		void initial_acceleration(Eigen::MatrixXd &acceleration) const;
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
	};
} // namespace polyfem::varform
