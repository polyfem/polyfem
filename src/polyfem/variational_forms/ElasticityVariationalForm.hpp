#pragma once

namespace polyfem
{
	class ElasticityVariationalForm
	{
	public:
		virtual ~ElasticityVariationalForm() = default;

		std::shared_ptr<assembler::PressureAssembler> build_pressure_assembler(
			const int n_bases_,
			const std::vector<basis::ElementBases> &bases_) const;
		std::shared_ptr<assembler::PressureAssembler> build_pressure_assembler() const
		{
			return build_pressure_assembler(n_bases, bases);
		}

	private:
		/// assemblers

		/// assembler corresponding to governing physical equations
		std::shared_ptr<assembler::Assembler> assembler = nullptr;

		std::shared_ptr<assembler::PressureAssembler> elasticity_pressure_assembler = nullptr;
		std::shared_ptr<assembler::ViscousDamping> damping_assembler = nullptr;
		std::shared_ptr<assembler::ViscousDampingPrev> damping_prev_assembler = nullptr;

		/// FE bases, the size is #elements
		std::vector<basis::ElementBases> bases;

		/// number of bases
		int n_bases;

		/// polygons, used since poly have no geom mapping
		std::map<int, Eigen::MatrixXd> polys;
		/// polyhedra, used since poly have no geom mapping
		std::map<int, std::pair<Eigen::MatrixXd, Eigen::MatrixXi>> polys_3d;

		/// Mapping from input nodes to FE nodes
		std::shared_ptr<polyfem::mesh::MeshNodes> mesh_nodes;

		/// used to store assembly values for small problems
		assembler::AssemblyValsCache ass_vals_cache;

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