#pragma once

#include <polyfem/mesh/remesh/WildRemeshingND.hpp>
#include <polyfem/mesh/remesh/wild_remesh/OperationCache.hpp>

#include <wmtk/TriMesh.h>

namespace polyfem::mesh
{
	class WildRemeshing2D : public WildRemeshingND<wmtk::TriMesh>
	{
	public:
		/// @brief Construct a new WildRemeshing2D object
		/// @param state Simulation current state
		WildRemeshing2D(
			const State &state,
			const Eigen::MatrixXd &obstacle_displacements,
			const Eigen::MatrixXd &obstacle_vals,
			const double current_time,
			const double starting_energy)
			: WildRemeshingND(state, obstacle_displacements, obstacle_vals, current_time, starting_energy)
		{
		}

		virtual ~WildRemeshing2D(){};

		// ---------------------------------------------------------------------

		/// @brief Collect all boundary edge tuples.
		std::vector<Tuple> boundary_edges() const;

		// ---------------------------------------------------------------------
		// Remeshing operations

		/// @brief Execute the remeshing
		/// @param split Perform splitting operations
		/// @param collapse Perform collapsing operations
		/// @param smooth Perform smoothing operations
		/// @param swap Perform edge swapping operations
		/// @param max_ops Maximum number of operations to perform (default: unlimited)
		/// @return True if any operation was performed.
		bool execute(
			const bool split = true,
			const bool collapse = false,
			const bool smooth = false,
			const bool swap = false,
			const double max_ops_percent = -1) override;

		// Smoothing
		bool smooth_before(const Tuple &t) override;
		bool smooth_after(const Tuple &t) override;

		// Edge splitting
		bool split_edge_before(const Tuple &t) override;
		bool split_edge_after(const Tuple &t) override;

		// Edge collapse
		bool collapse_edge_before(const Tuple &t) override;
		bool collapse_edge_after(const Tuple &t) override;

		// Edge swap
		bool swap_edge_before(const Tuple &t) override;
		bool swap_edge_after(const Tuple &t) override;

		/// @brief Check if a triangle is inverted
		bool is_inverted(const Tuple &loc) const override;

	protected:
		/// @brief Create an internal mesh representation and associate attributes
		void create_mesh(const size_t num_vertices, const Eigen::MatrixXi &elements) override;

	private:
		/// @brief Compute the average elastic energy of the faces containing an edge.
		double edge_elastic_energy(const Tuple &e) const;

		/// @brief Relax a local n-ring around a vertex.
		/// @param t Center of the local n-ring
		/// @param n_ring Size of the n-ring
		/// @return If the local relaxation reduced the energy "significantly"
		bool local_relaxation(const Tuple &t, const int n_ring);

		/// @brief Create a vector of all the new edge after an operation.
		/// @param tris New triangles.
		std::vector<Tuple> new_edges_after(const std::vector<Tuple> &tris) const;

		double total_area;

		// TODO: make this thread local
		OperationCache2D op_cache;
	};

} // namespace polyfem::mesh
