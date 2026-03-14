#pragma once

#include <polyfem/varforms/VarForm.hpp>

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
	class NonlinearElasticTransientVarForm : public VarForm
	{
	public:
		void init(const std::string &formulation, const Units &units, const json &args, const std::string &out_path) override;

		void load_mesh(const mesh::Mesh &mesh, const json &args) override;

		/// Build FE bases and any discretization-specific state.
		void build_basis(mesh::Mesh &mesh, const bool iso_parametric, const json &args) override;

		void solve(Eigen::MatrixXd &sol) override;

	protected:
		void reset() override;

		/// @brief Get a constant reference to the geometry mapping bases.
		/// @return A constant reference to the geometry mapping bases.
		const std::vector<basis::ElementBases> &geom_bases() const
		{
			return iso_parametric ? bases : geom_bases_;
		}

	private:
		bool remesh_enabled;

		void init_solve(Eigen::MatrixXd &sol, const double t);

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

		mesh::Obstacle obstacle;
		/// @brief IPC collision mesh
		ipc::CollisionMesh collision_mesh;

		/// assembler corresponding to governing physical equations
		std::shared_ptr<assembler::Assembler> assembler = nullptr;
		std::shared_ptr<assembler::Mass> mass_matrix_assembler = nullptr;

		std::shared_ptr<assembler::PressureAssembler> elasticity_pressure_assembler = nullptr;

		std::shared_ptr<assembler::ViscousDamping> damping_assembler = nullptr;
		std::shared_ptr<assembler::ViscousDampingPrev> damping_prev_assembler = nullptr;

		/// FE bases, the size is #elements
		std::vector<basis::ElementBases> bases;
		/// Geometric mapping bases, if the elements are isoparametric, this list is empty
		std::vector<basis::ElementBases> geom_bases_;

		/// number of bases
		int n_bases;
		/// number of geometric bases
		int n_geom_bases;

		/// polygons, used since poly have no geom mapping
		std::map<int, Eigen::MatrixXd> polys;
		/// polyhedra, used since poly have no geom mapping
		std::map<int, std::pair<Eigen::MatrixXd, Eigen::MatrixXi>> polys_3d;

		/// vector of discretization orders, used when not all elements have the same degree, one per element
		Eigen::VectorXi disc_orders, disc_ordersq;

		/// Mapping from input nodes to FE nodes
		std::shared_ptr<polyfem::mesh::MeshNodes> mesh_nodes, geom_mesh_nodes;

		/// used to store assembly values for small problems
		assembler::AssemblyValsCache ass_vals_cache;
		assembler::AssemblyValsCache mass_ass_vals_cache;

		/// Mass matrix, it is computed only for time dependent problems
		StiffnessMatrix mass;
		/// average system mass, used for contact with IPC
		double avg_mass;

		/// list of boundary nodes
		std::vector<int> boundary_nodes;
		/// mapping from elements to nodes for all mesh
		std::vector<mesh::LocalBoundary> total_local_boundary;
		/// mapping from elements to nodes for dirichlet boundary conditions
		std::vector<mesh::LocalBoundary> local_boundary;
		/// mapping from elements to nodes for neumann boundary conditions
		std::vector<mesh::LocalBoundary> local_neumann_boundary;
		/// mapping from elements to nodes for pressure boundary conditions
		std::vector<mesh::LocalBoundary> local_pressure_boundary;
		/// mapping from elements to nodes for pressure boundary conditions
		std::unordered_map<int, std::vector<mesh::LocalBoundary>> local_pressure_cavity;
		/// nodes on the boundary of polygonal elements, used for harmonic bases
		std::map<int, basis::InterfaceData> poly_edge_to_data;
		/// per node dirichlet
		std::vector<int> dirichlet_nodes;
		std::vector<RowVectorNd> dirichlet_nodes_position;
		/// per node neumann
		std::vector<int> neumann_nodes;
		std::vector<RowVectorNd> neumann_nodes_position;

		/// Inpute nodes (including high-order) to polyfem nodes, only for isoparametric
		Eigen::VectorXi in_node_to_node;
		/// maps in vertices/edges/faces/cells to polyfem vertices/edges/faces/cells
		Eigen::VectorXi in_primitive_to_primitive;

		/// timedependent stuff cached
		solver::SolveData solve_data;

		double t0;
		int time_steps;
		double dt;
	};
} // namespace polyfem::varform
