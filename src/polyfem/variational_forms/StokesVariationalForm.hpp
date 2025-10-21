

#pragma once

namespace polyfem
{
	class StokesVariationalForm
	{
	public:
		virtual ~StokesVariationalForm() = default;
		/// splits the solution in solution and pressure for mixed problems
		/// @param[in/out] sol solution
		/// @param[out] pressure pressure
		void sol_to_pressure(Eigen::MatrixXd &sol, Eigen::MatrixXd &pressure);

		/// solves transient navier stokes with operator splitting
		/// @param[in] time_steps number of time steps
		/// @param[in] dt timestep size
		/// @param[out] sol solution
		/// @param[out] pressure pressure
		void solve_transient_navier_stokes_split(const int time_steps, const double dt, Eigen::MatrixXd &sol, Eigen::MatrixXd &pressure);
		/// solves transient navier stokes with FEM
		/// @param[in] time_steps number of time steps
		/// @param[in] t0 initial times
		/// @param[in] dt timestep size
		/// @param[out] sol solution
		/// @param[out] pressure pressure
		void solve_transient_navier_stokes(const int time_steps, const double t0, const double dt, Eigen::MatrixXd &sol, Eigen::MatrixXd &pressure);

	private:
		/// assembler corresponding to governing physical equations
		std::shared_ptr<assembler::Assembler> assembler = nullptr;
		std::shared_ptr<assembler::MixedAssembler> mixed_assembler = nullptr;
		std::shared_ptr<assembler::Assembler> pressure_assembler = nullptr;

		/// FE bases, the size is #elements
		std::vector<basis::ElementBases> bases;
		/// FE pressure bases for mixed elements, the size is #elements
		std::vector<basis::ElementBases> pressure_bases;

		/// number of bases
		int n_bases;
		/// number of pressure bases
		int n_pressure_bases;

		/// polygons, used since poly have no geom mapping
		std::map<int, Eigen::MatrixXd> polys;
		/// polyhedra, used since poly have no geom mapping
		std::map<int, std::pair<Eigen::MatrixXd, Eigen::MatrixXi>> polys_3d;

		/// Mapping from input nodes to FE nodes
		std::shared_ptr<polyfem::mesh::MeshNodes> mesh_nodes, pressure_mesh_nodes;

		/// used to store assembly values for small problems
		assembler::AssemblyValsCache ass_vals_cache;
		/// used to store assembly values for pressure for small problems
		assembler::AssemblyValsCache pressure_ass_vals_cache;

		/// use average pressure for stokes problem to fix the additional dofs, true by default
		/// if false, it will fix one pressure node to zero
		bool use_avg_pressure;

		/// Construct a vector of boundary conditions ids with their dimension flags.
		std::unordered_map<int, std::array<bool, 3>>
		boundary_conditions_ids(const std::string &bc_type) const;

		/// list of boundary nodes
		std::vector<int> boundary_nodes;
		/// list of neumann boundary nodes
		std::vector<int> pressure_boundary_nodes;
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
	};
} // namespace polyfem