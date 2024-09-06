#pragma once

#include <polyfem/State.hpp>
#include <polyfem/mesh/Mesh.hpp>
#include <polyfem/mesh/LocalBoundary.hpp>
#include <polyfem/mesh/remesh/wild_remesh/LocalMesh.hpp>

namespace polyfem::mesh
{
	// Things needed for the local relaxation solve
	template <typename M>
	class LocalRelaxationData
	{
	public:
		LocalRelaxationData(
			const State &state,
			LocalMesh<M> &local_mesh,
			const double current_time,
			const bool contact_enabled);

		Eigen::MatrixXd sol() const
		{
			return utils::flatten(local_mesh.displacements());
		}

		bool is_volume() const { return mesh->is_volume(); }
		int dim() const { return mesh->dimension(); }
		int n_bases() const { return m_n_bases; }
		int ndof() const { return dim() * n_bases(); }
		int n_free_dof() const { return ndof() - boundary_nodes.size(); }

		solver::SolveData solve_data;

		LocalMesh<M> &local_mesh;

	private:
		void init_mesh(const State &state);
		void init_bases(const State &state);
		void init_boundary_conditions(const State &state);
		void init_assembler(const State &state);
		void init_mass_matrix(const State &state);
		void init_solve_data(
			const State &state,
			const double current_time,
			const bool contact_enabled);

		// Mesh data
		std::unique_ptr<Mesh> mesh;

		// Basis data
		int m_n_bases;
		std::vector<polyfem::basis::ElementBases> bases;

		/// Assembler data
		std::shared_ptr<assembler::Assembler> assembler;
		assembler::AssemblyValsCache assembly_vals_cache;

		std::shared_ptr<assembler::Mass> mass_matrix_assembler;
		assembler::AssemblyValsCache mass_assembly_vals_cache;
		Eigen::SparseMatrix<double> mass;

		std::shared_ptr<assembler::PressureAssembler> pressure_assembler;

		/// current problem, it contains rhs and bc
		std::shared_ptr<assembler::Problem> problem;

		/// list of boundary nodes
		std::vector<int> boundary_nodes;
		/// mapping from elements to nodes for dirichlet boundary conditions
		std::vector<mesh::LocalBoundary> local_boundary;
		/// mapping from elements to nodes for neumann boundary conditions
		std::vector<mesh::LocalBoundary> local_neumann_boundary;
		/// mapping from elements to nodes for pressure boundary conditions
		std::vector<mesh::LocalBoundary> local_pressure_boundary;
		/// per node dirichlet
		std::vector<int> dirichlet_nodes;
		std::vector<RowVectorNd> dirichlet_nodes_position;
		/// per node neumann
		std::vector<int> neumann_nodes;
		std::vector<RowVectorNd> neumann_nodes_position;
		// per node pressure
		std::unordered_map<int, std::vector<LocalBoundary>> local_pressure_cavity;
		std::vector<int> pressure_boundary_nodes;

		Eigen::MatrixXd rhs;

		ipc::CollisionMesh collision_mesh;
	};
} // namespace polyfem::mesh