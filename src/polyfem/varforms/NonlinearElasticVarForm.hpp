#pragma once

#include <polyfem/varforms/ElasticVarForm.hpp>

#include <polyfem/assembler/PressureAssembler.hpp>
#include <polyfem/assembler/ViscousDamping.hpp>

#include <ipc/collision_mesh.hpp>

#include <functional>

namespace polyfem::varform
{
	class NonlinearElasticVarForm : public ElasticVarForm, public VarFormTestAccess
	{
	public:
		bool is_contact_enabled() const override
		{
			return args.contains("contact") && args["contact"].contains("enabled") && args["contact"]["enabled"].get<bool>();
		}

		io::OutputSpace output_space() const override;
		VarFormDebugData debug_data() const override;
		std::vector<io::OutputField> output_fields(
			const io::OutputSample &sample,
			const Eigen::MatrixXd &solution,
			const io::OutputFieldOptions &options) const override;

	protected:
		void reset() override;
		void load_mesh(const mesh::Mesh &mesh, const json &args) override;
		void build_basis(mesh::Mesh &mesh, const bool iso_parametric, const json &args) override;
		std::shared_ptr<assembler::RhsAssembler> build_rhs_assembler(
			const int n_bases,
			const std::vector<basis::ElementBases> &bases,
			const assembler::AssemblyValsCache &ass_vals_cache) const override;
		void init_solve(Eigen::MatrixXd &sol, const double t);
		void init_forms(const json &args, const int dim, Eigen::MatrixXd &sol, const double t);
		void solve_tensor_nonlinear(int step, Eigen::MatrixXd &sol, const bool init_lagging = true);

		std::shared_ptr<assembler::PressureAssembler> build_pressure_assembler() const;
		void build_collision_mesh(const mesh::Mesh &mesh, const json &args);
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

		int n_obstacle_vertices() const override { return obstacle.n_vertices(); };
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
