#pragma once

#include <polyfem/varforms/VarForm.hpp>

#include <polyfem/mesh/Obstacle.hpp>

namespace polyfem::varform
{
	class ElasticVarForm : public VarForm, public VarFormMatrixDebugAccess
	{
	public:
		void init(const std::string &formulation, const Units &units, const json &args, const std::string &out_path) override;
		io::OutStatsData compute_errors(const Eigen::MatrixXd &solution) override;

		void save_json(const Eigen::MatrixXd &solution, std::ostream &out) const override;

		io::OutputSpace output_space() const override;
		std::vector<io::OutputField> output_fields(
			const io::OutputSample &sample,
			const Eigen::MatrixXd &solution,
			const io::OutputFieldOptions &options) const override;

		// DEBUG/TEST stuff
		VarFormDebugData debug_data() const override;
		void build_stiffness_mat_debug(StiffnessMatrix &stiffness) override;
		const StiffnessMatrix *mass_matrix_debug() const override { return &mass; }

	protected:
		void reset() override;
		void load_mesh(const mesh::Mesh &mesh, const json &args) override;
		void build_basis(mesh::Mesh &mesh, const bool iso_parametric, const json &args) override;
		void save_step_state(const double t0, const double dt, const int t, const Eigen::MatrixXd &sol) const override;
		void set_materials(assembler::Assembler &assembler) const override;
		virtual void build_stiffness_mat(StiffnessMatrix &stiffness);

		QuadratureOrders n_boundary_samples() const;
		void initial_solution(Eigen::MatrixXd &solution) const;
		void initial_velocity(Eigen::MatrixXd &velocity) const;
		void initial_acceleration(Eigen::MatrixXd &acceleration) const;
		void build_mesh_matrices(Eigen::MatrixXd &V, Eigen::MatrixXi &F) const;

		mesh::Obstacle obstacle;
		bool remesh_enabled = false;
	};
} // namespace polyfem::varform
