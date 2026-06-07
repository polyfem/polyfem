#pragma once

#include <polyfem/varforms/VarForm.hpp>

namespace polyfem::varform
{
	class ElasticVarForm : public VarForm
	{
	public:
		void init(const std::string &formulation, const Units &units, const json &args, const std::string &out_path) override;

		void save_json(const Eigen::MatrixXd &solution, std::ostream &out) const override;

		std::vector<io::OutputField> output_fields(
			const io::OutputSample &sample,
			const Eigen::MatrixXd &solution,
			const io::OutputFieldOptions &options) const override;

	protected:
		void load_mesh(const mesh::Mesh &mesh, const json &args) override;
		void save_step_state(const double t0, const double dt, const int t, const Eigen::MatrixXd &sol) const override;

		void initial_velocity(Eigen::MatrixXd &velocity) const;
		void initial_acceleration(Eigen::MatrixXd &acceleration) const;
		void build_mesh_matrices(Eigen::MatrixXd &V, Eigen::MatrixXi &F) const;

		virtual int n_obstacle_vertices() const { return 0; };
	};
} // namespace polyfem::varform
