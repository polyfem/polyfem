#pragma once

#include <polyfem/varforms/VarForm.hpp>

#include <polyfem/assembler/Assembler.hpp>
#include <polyfem/assembler/Mass.hpp>
#include <polyfem/assembler/PressureAssembler.hpp>
#include <polyfem/assembler/ViscousDamping.hpp>
#include <polyfem/basis/ElementBases.hpp>
#include <polyfem/basis/InterfaceData.hpp>
#include <polyfem/mesh/Obstacle.hpp>
#include <polyfem/mesh/MeshNodes.hpp>
#include <polyfem/solver/SolveData.hpp>

#include <ipc/collision_mesh.hpp>

#include <map>
#include <functional>

namespace polyfem::varform
{
	class ElasticVarForm : public VarForm
	{
	public:
		void init(const std::string &formulation, const Units &units, const json &args, const std::string &out_path) override;
		void load_mesh(const mesh::Mesh &mesh, const json &args) override;
		void build_basis(mesh::Mesh &mesh, const bool iso_parametric, const json &args) override;
		void assemble_rhs(const mesh::Mesh &mesh, const json &args) override;
		void assemble_mass_mat(const mesh::Mesh &mesh, const json &args) override;
		const StiffnessMatrix *mass_matrix() const override { return &mass; }

		io::OutputState output_state() const override;
		std::vector<io::OutputField> output_fields(
			const io::OutputSample &sample,
			const Eigen::MatrixXd &solution,
			const io::OutputFieldOptions &options) const override;

	protected:
		void reset() override;

		QuadratureOrders n_boundary_samples() const;
		void initial_solution(Eigen::MatrixXd &solution) const;
		void initial_velocity(Eigen::MatrixXd &velocity) const;
		void initial_acceleration(Eigen::MatrixXd &acceleration) const;

		/// @brief Get a constant reference to the geometry mapping bases.
		/// @return A constant reference to the geometry mapping bases.
		const std::vector<basis::ElementBases> &geom_bases() const
		{
			return iso_parametric ? bases : geom_bases_;
		}

		/// assembler corresponding to governing physical equations
		std::shared_ptr<assembler::Assembler> assembler = nullptr;
		std::shared_ptr<assembler::Mass> mass_matrix_assembler = nullptr;
		std::shared_ptr<assembler::HRZMass> pure_mass_matrix_assembler = nullptr;

		/// FE bases, the size is #elements
		std::vector<basis::ElementBases> bases;

		/// number of bases
		int n_bases = 0;

		/// vector of discretization orders, used when not all elements have the same degree, one per element
		Eigen::VectorXi disc_orders, disc_ordersq;

		/// nodes on the boundary of polygonal elements, used for harmonic bases
		std::map<int, basis::InterfaceData> poly_edge_to_data;

		/// Mapping from input nodes to FE nodes
		std::shared_ptr<polyfem::mesh::MeshNodes> mesh_nodes;

		/// used to store assembly values for small problems
		assembler::AssemblyValsCache ass_vals_cache;
		assembler::AssemblyValsCache mass_ass_vals_cache;
		assembler::AssemblyValsCache pure_mass_ass_vals_cache;

		/// Mass matrix, it is computed only for time dependent problems
		StiffnessMatrix mass;
		StiffnessMatrix pure_mass;
		/// average system mass, used for contact with IPC
		double avg_mass = 0;
		Eigen::MatrixXd rhs;

		solver::SolveData solve_data;

		mesh::Obstacle obstacle;
		/// @brief IPC collision mesh
		ipc::CollisionMesh collision_mesh;

		std::shared_ptr<assembler::PressureAssembler> elasticity_pressure_assembler = nullptr;

		std::shared_ptr<assembler::ViscousDamping> damping_assembler = nullptr;
		std::shared_ptr<assembler::ViscousDampingPrev> damping_prev_assembler = nullptr;

		bool remesh_enabled = false;
		double t0 = 0;
		int time_steps = 0;
		double dt = 0;

		void set_materials(assembler::Assembler &assembler) const;

	protected:
		void build_polygonal_basis(const mesh::Mesh &mesh);
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
	};
} // namespace polyfem::varform
