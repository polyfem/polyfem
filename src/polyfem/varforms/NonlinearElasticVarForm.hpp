#pragma once

#include <polyfem/varforms/ElasticVarForm.hpp>

#include <polyfem/assembler/PressureAssembler.hpp>
#include <polyfem/assembler/ViscousDamping.hpp>
#include <polyfem/solver/SolveData.hpp>

#include <ipc/collision_mesh.hpp>

#include <functional>

namespace polyfem::varform
{
	class NonlinearElasticVarForm : public ElasticVarForm
	{
	public:
		void init(const std::string &formulation, const Units &units, const json &args, const std::string &out_path) override;

		bool is_contact_enabled() const override
		{
			return args["contact"]["enabled"];
		}
		solver::SolveData *solve_data() override { return &solve_data_; }
		const solver::SolveData *solve_data() const override { return &solve_data_; }
		const ipc::CollisionMesh &collision_mesh() const override { return collision_mesh_; }
		const mesh::Obstacle &get_obstacle() const override { return obstacle; }
		const assembler::ViscousDamping *damping_assembler() const override { return damping_assembler_.get(); }
		const assembler::ViscousDampingPrev *damping_prev_assembler() const override { return damping_prev_assembler_.get(); }

		io::OutputSpace output_space() const override;
		std::vector<io::OutputField> output_fields(
			const io::OutputSample &sample,
			const Eigen::MatrixXd &solution,
			const io::OutputFieldOptions &options) const override;

		static void build_collision_mesh(
			const mesh::Mesh &mesh,
			const int n_bases,
			const std::vector<basis::ElementBases> &bases,
			const std::vector<basis::ElementBases> &geom_bases,
			const std::vector<mesh::LocalBoundary> &total_local_boundary,
			const mesh::Obstacle &obstacle,
			const json &args,
			const std::function<std::string(const std::string &)> &resolve_input_path,
			const Eigen::VectorXi &in_node_to_node,
			ipc::CollisionMesh &collision_mesh);

	protected:
		void invalidate_after_geometry_update() override;
		void invalidate_after_parameter_update() override;
		void reset() override;
		void load_mesh(const mesh::Mesh &mesh, const json &args) override;
		void build_basis(mesh::Mesh &mesh, const bool iso_parametric, const json &args) override;
		void build_rhs_assembler() override;
		void init_solve(Eigen::MatrixXd &sol, const double t, const InitialConditionOverride *initial_condition_override, bool is_differentiable);
		void init_forms(const json &args, const int dim, Eigen::MatrixXd &sol, const double t, bool is_differentiable);
		void solve_tensor_nonlinear(int step, Eigen::MatrixXd &sol, bool is_differentiable, const bool init_lagging = true);

		std::shared_ptr<assembler::PressureAssembler> build_pressure_assembler() const;
		void build_collision_mesh(const mesh::Mesh &mesh, const json &args);
		void preprocess_contact_parameters();
		ipc::CollisionMesh collision_mesh_;
		std::shared_ptr<assembler::PressureAssembler> elasticity_pressure_assembler = nullptr;
		std::shared_ptr<assembler::ViscousDamping> damping_assembler_ = nullptr;
		std::shared_ptr<assembler::ViscousDampingPrev> damping_prev_assembler_ = nullptr;

		mesh::Obstacle obstacle;

		solver::SolveData solve_data_;
		std::vector<std::shared_ptr<solver::Form>> forms;
		bool contact_dhat_was_explicit_ = false;

		int n_obstacle_vertices() const override { return obstacle.n_vertices(); }
	};

	class NonlinearElasticTransientVarForm : public NonlinearElasticVarForm
	{
	public:
		std::string name() const override { return "NonlinearElasticTransient"; }

	private:
		void solve_problem(
			Eigen::MatrixXd &sol,
			const InitialConditionOverride *initial_condition_override,
			const ForwardStepCallback &post_step,
			bool is_differentiable) override;
	};

	class NonlinearElasticStaticVarForm : public NonlinearElasticVarForm
	{
	public:
		std::string name() const override { return "NonlinearElasticStatic"; }

	private:
		void solve_problem(
			Eigen::MatrixXd &sol,
			const InitialConditionOverride *initial_condition_override,
			const ForwardStepCallback &post_step,
			bool is_differentiable) override;
	};
} // namespace polyfem::varform
