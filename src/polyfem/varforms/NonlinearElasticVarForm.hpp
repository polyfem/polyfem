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
			return args.contains("contact") && args["contact"].contains("enabled") && args["contact"]["enabled"].get<bool>();
		}

		io::OutputSpace output_space() const override;
		std::vector<io::OutputField> output_fields(
			const io::OutputSample &sample,
			const Eigen::MatrixXd &solution,
			const io::OutputFieldOptions &options) const override;

		/// Prepare the standard nonlinear-elastic assembly state for use as a
		/// block in a larger nonlinear problem. This does not construct or solve
		/// the child NLProblem.
		void prepare_for_embedding();
		void initial_solution_for_embedding(Eigen::MatrixXd &solution, const std::string &state_prefix = "") const;
		void init_forms_for_embedding(Eigen::MatrixXd &solution, double t, const std::string &state_prefix = "");
		void advance_for_embedding(const Eigen::VectorXd &solution);
		void update_barrier_stiffness_for_embedding(const Eigen::VectorXd &solution);
		bool save_timestep_for_embedding(
			double time, int step, double dt, const Eigen::MatrixXd &solution,
			paraviewo::VTMWriter &vtm, const std::string &block_prefix) const;

		int embedding_ndof() const;
		const std::vector<std::shared_ptr<solver::Form>> &embedding_forms() const { return forms; }
		const std::vector<std::shared_ptr<solver::AugmentedLagrangianForm>> &embedding_al_forms() const { return solve_data.al_form; }
		const StiffnessMatrix &embedding_norm_matrix() const { return pure_mass_; }
		const std::shared_ptr<time_integrator::ImplicitTimeIntegrator> &embedding_time_integrator() const { return solve_data.time_integrator; }
		const FESpace &embedding_space() const { return space_; }

	protected:
		void reset() override;
		void load_mesh(const mesh::Mesh &mesh, const json &args) override;
		void build_basis(mesh::Mesh &mesh, const bool iso_parametric, const json &args) override;
		void build_rhs_assembler() override;
		void init_solve(Eigen::MatrixXd &sol, const double t);
		void init_solve_data(Eigen::MatrixXd &sol, double t, const std::string &state_prefix);
		void init_forms(const json &args, const int dim, Eigen::MatrixXd &sol, const double t);
		void solve_tensor_nonlinear(int step, Eigen::MatrixXd &sol, const bool init_lagging = true);

		std::shared_ptr<assembler::PressureAssembler> build_pressure_assembler() const;
		void build_collision_mesh(const mesh::Mesh &mesh, const json &args);
		void preprocess_contact_parameters();
		void build_collision_mesh(
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

		ipc::CollisionMesh collision_mesh;
		std::shared_ptr<assembler::PressureAssembler> elasticity_pressure_assembler = nullptr;
		std::shared_ptr<assembler::ViscousDamping> damping_assembler = nullptr;
		std::shared_ptr<assembler::ViscousDampingPrev> damping_prev_assembler = nullptr;

		mesh::Obstacle obstacle;

		solver::SolveData solve_data;
		std::vector<std::shared_ptr<solver::Form>> forms;
		bool contact_dhat_was_explicit_ = false;

		int n_obstacle_vertices() const override { return obstacle.n_vertices(); }
	};

	class NonlinearElasticTransientVarForm : public NonlinearElasticVarForm
	{
	public:
		std::string name() const override { return "NonlinearElasticTransient"; }

	private:
		void solve_problem(Eigen::MatrixXd &sol) override;
	};

	class NonlinearElasticStaticVarForm : public NonlinearElasticVarForm
	{
	public:
		std::string name() const override { return "NonlinearElasticStatic"; }

	private:
		void solve_problem(Eigen::MatrixXd &sol) override;
	};
} // namespace polyfem::varform
