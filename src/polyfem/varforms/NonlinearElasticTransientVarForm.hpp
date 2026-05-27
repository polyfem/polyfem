#pragma once

#include <polyfem/varforms/ElasticVarForm.hpp>

#include <polyfem/assembler/Assembler.hpp>
#include <polyfem/assembler/Mass.hpp>
#include <polyfem/assembler/ViscousDamping.hpp>
#include <polyfem/assembler/PressureAssembler.hpp>

#include <polyfem/mesh/Mesh.hpp>
#include <polyfem/mesh/MeshNodes.hpp>

#include <polyfem/basis/ElementBases.hpp>
#include <polyfem/basis/InterfaceData.hpp>

namespace polyfem::varform
{
	class NonlinearElasticTransientVarForm : public ElasticVarForm
	{
	public:
		void init(const std::string &formulation, const Units &units, const json &args, const std::string &out_path) override;

		void load_mesh(const mesh::Mesh &mesh, const json &args) override;

		/// Build FE bases and any discretization-specific state.
		void build_basis(mesh::Mesh &mesh, const bool iso_parametric, const json &args) override;

		void assemble_rhs(const mesh::Mesh &mesh, const json &args) override;
		void assemble_mass_mat(const mesh::Mesh &mesh, const json &args) override;
		void solve(Eigen::MatrixXd &sol) override;
		void sync_state(State &state) const override;

		std::string name() const override { return "NonlinearElasticTransient"; }

	protected:
		void reset() override;

	private:
		bool remesh_enabled = false;

		void init_solve(Eigen::MatrixXd &sol, const double t);
		void init_forms(const json &args, const int dim, Eigen::MatrixXd &sol, const double t);
		void solve_tensor_nonlinear(int step, Eigen::MatrixXd &sol, const bool init_lagging = true);

		void initial_solution(Eigen::MatrixXd &solution) const;
		void initial_velocity(Eigen::MatrixXd &velocity) const;
		void initial_acceleration(Eigen::MatrixXd &acceleration) const;
		void set_materials(assembler::Assembler &assembler) const;
		std::shared_ptr<assembler::PressureAssembler> build_pressure_assembler() const;
		std::vector<int> primitive_to_node() const;
		std::vector<int> node_to_primitive() const;

		void build_node_mapping(const mesh::Mesh &mesh, const json &args);
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

		double t0;
		int time_steps;
		double dt;
	};

	class NonlinearElasticStaticVarForm : public NonlinearElasticTransientVarForm
	{
	public:
		std::string name() const override { return "NonlinearElasticStatic"; }
	};
} // namespace polyfem::varform
