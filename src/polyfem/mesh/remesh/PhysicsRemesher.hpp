#pragma once

#include <polyfem/State.hpp>
#include <polyfem/mesh/remesh/WildRemesher.hpp>
#include <polyfem/mesh/remesh/wild_remesh/OperationCache.hpp>
#include <polyfem/mesh/remesh/wild_remesh/LocalMesh.hpp>

namespace polyfem::mesh
{
	template <class WMTKMesh>
	class PhysicsRemesher : public WildRemesher<WMTKMesh>
	{
	private:
		using Super = WildRemesher<WMTKMesh>;
		using This = PhysicsRemesher<WMTKMesh>;

	protected:
		using Super::args;
		using Super::executor;
		using Super::state;

	public:
		using Super::vertex_attrs;
		using Tuple = typename Super::Tuple;
		using Operations = typename Super::Operations;
		using VectorNd = typename Super::VectorNd;

		PhysicsRemesher(
			const State &state,
			const Eigen::MatrixXd &obstacle_displacements,
			const Eigen::MatrixXd &obstacle_vals,
			const double current_time,
			const double starting_energy)
			: Super(state, obstacle_displacements, obstacle_vals, current_time, starting_energy)
		{
		}

		virtual ~PhysicsRemesher(){};

		// Edge splitting
		void split_edges() override;
		bool split_edge_before(const Tuple &t) override;
		bool split_edge_after(const Tuple &t) override;

		// Edge collapse
		void collapse_edges() override;
		bool collapse_edge_before(const Tuple &t) override;
		bool collapse_edge_after(const Tuple &t) override;

	protected:
		/// @brief Renew the neighbor tuples of an operation.
		/// @param op Operation
		/// @param tris Tuples of the operation
		/// @return New operations
		Operations renew_neighbor_tuples(
			const std::string &op, const std::vector<Tuple> &tris) const override;

		/// @brief Relax a local n-ring around a vertex.
		/// @param t Center of the local n-ring
		/// @return If the local relaxation reduced the energy "significantly"
		bool local_relaxation(const Tuple &t, const double acceptance_tolerance);

		/// @brief Get the local n-ring around a vertex.
		/// @param center Center of the local n-ring
		/// @return Tuple of the local n-ring
		std::vector<Tuple> local_mesh_tuples(const VectorNd &center) const;

		/// @brief Get the local n-ring around a vertex.
		/// @param v Center of the local n-ring
		/// @return Tuple of the local n-ring
		std::vector<Tuple> local_mesh_tuples(const Tuple &v) const
		{
			return local_mesh_tuples(this->vertex_attrs[v.vid(*this)].rest_position);
		}

		/// @brief Compute the energy of a local n-ring around a vertex.
		/// @param local_mesh_center Center of the local n-ring.
		/// @return Energy of the local n-ring.
		double local_mesh_energy(const VectorNd &local_mesh_center) const;

		/// @brief Get the energy of the local n-ring around a vertex.
		double local_energy_before() const { return this->op_cache->local_energy; }

		/// @brief Get the boundary nodes of a local_mesh.
		/// @param local_mesh Local mesh.
		/// @return Boundary nodes of the local mesh.
		std::vector<int> local_boundary_nodes(
			const LocalMesh<This> &local_mesh) const;

		/// @brief Initialize the solve data for a local relaxation.
		/// @param local_mesh Local mesh.
		/// @param bases Element bases.
		/// @param boundary_nodes Boundary nodes of the local mesh.
		/// @param assembler Assembler utils.
		/// @param contact_enabled If contact is enabled.
		/// @param solve_data Solve data.
		/// @param ass_vals_cache Assembly values cache.
		/// @param mass Mass matrix.
		/// @param collision_mesh Collision mesh.
		void local_solve_data(
			const LocalMesh<This> &local_mesh,
			const std::vector<polyfem::basis::ElementBases> &bases,
			const std::vector<int> &boundary_nodes,
			const assembler::AssemblerUtils &assembler,
			const bool contact_enabled,
			solver::SolveData &solve_data,
			assembler::AssemblyValsCache &ass_vals_cache,
			Eigen::SparseMatrix<double> &mass,
			ipc::CollisionMesh &collision_mesh) const;

		/// @brief Compute the average elastic energy of the faces containing an edge.
		double edge_elastic_energy(const Tuple &e) const;

		/// @brief Write a visualization mesh of the priority queue
		/// @param e current edge tuple to be split
		void write_priority_queue_mesh(const std::string &path, const Tuple &e) const;
	};

	class PhysicsTriRemesher : public PhysicsRemesher<wmtk::TriMesh>
	{
	private:
		using Super = PhysicsRemesher<wmtk::TriMesh>;

	public:
		using Tuple = typename Super::Tuple;

		PhysicsTriRemesher(
			const State &state,
			const Eigen::MatrixXd &obstacle_displacements,
			const Eigen::MatrixXd &obstacle_vals,
			const double current_time,
			const double starting_energy)
			: Super(state, obstacle_displacements, obstacle_vals, current_time, starting_energy)
		{
		}

	protected:
		// TODO: move this into PhysicsRemesher
		// Smoothing
		bool smooth_before(const Tuple &t) override;
		bool smooth_after(const Tuple &t) override;

		// Edge swap
		bool swap_edge_before(const Tuple &t) override;
		bool swap_edge_after(const Tuple &t) override;
	};

	class PhysicsTetRemesher : public PhysicsRemesher<wmtk::TetMesh>
	{
	private:
		using Super = PhysicsRemesher<wmtk::TetMesh>;

	public:
		using Tuple = typename Super::Tuple;

		PhysicsTetRemesher(
			const State &state,
			const Eigen::MatrixXd &obstacle_displacements,
			const Eigen::MatrixXd &obstacle_vals,
			const double current_time,
			const double starting_energy)
			: Super(state, obstacle_displacements, obstacle_vals, current_time, starting_energy)
		{
		}

	protected:
		// 4-4 Edge swap
		// bool swap_edge_44_before(const Tuple &t) override;
		// bool swap_edge_44_after(const Tuple &t) override;

		// 2-3 Face swap
		// bool swap_face_before(const Tuple &t) override;
		// bool swap_face_after(const Tuple &t) override;

		// 3-2 Edge swap
		// bool swap_edge_before(const Tuple &t) override;
		// bool swap_edge_after(const Tuple &t) override;
	};

} // namespace polyfem::mesh