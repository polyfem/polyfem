#pragma once

#include <polyfem/mesh/remesh/WildRemeshing.hpp>
#include <polyfem/mesh/remesh/wild_remesh/OperationCache.hpp>

#include <wmtk/TriMesh.h>

#include <unordered_map>

namespace polyfem::mesh
{
	class WildRemeshing2D : public WildRemeshing, public wmtk::TriMesh
	{
	public:
		static constexpr int DIM = 2;
		using VertexAttributes = WildRemeshing::VertexAttributes<DIM>;
		using EdgeAttributes = WildRemeshing::BoundaryAttributes;
		using FaceAttributes = WildRemeshing::ElementAttributes;

		/// @brief Construct a new WildRemeshing2D object
		/// @param state Simulation current state
		WildRemeshing2D(
			const State &state,
			const Eigen::MatrixXd &obstacle_displacements,
			const Eigen::MatrixXd &obstacle_vals,
			const double current_time,
			const double starting_energy)
			: WildRemeshing(state, obstacle_displacements, obstacle_vals, current_time, starting_energy),
			  wmtk::TriMesh()
		{
		}

		virtual ~WildRemeshing2D(){};

		/// @brief Initialize the mesh
		/// @param rest_positions Rest positions of the mesh (|V| × 2)
		/// @param positions Current positions of the mesh (|V| × 2)
		/// @param triangles Triangles of the mesh (|T| × 3)
		/// @param projection_quantities Quantities to be projected to the new mesh (2 rows per vertex and 1 column per quantity)
		/// @param edge_to_boundary_id Map from edge to boundary id (of size |E|)
		/// @param body_ids Body ids of the mesh (of size |T|)
		virtual void init(
			const Eigen::MatrixXd &rest_positions,
			const Eigen::MatrixXd &positions,
			const Eigen::MatrixXi &triangles,
			const Eigen::MatrixXd &projection_quantities,
			const EdgeMap<int> &edge_to_boundary_id,
			const std::vector<int> &body_ids) override;

		// ---------------------------------------------------------------------
		// Getters

		/// @brief Dimension of the mesh
		int dim() const override { return DIM; }
		/// @brief Exports rest positions of the stored mesh
		Eigen::MatrixXd rest_positions() const override;
		/// @brief Exports positions of the stored mesh
		Eigen::MatrixXd displacements() const override;
		/// @brief Exports displacements of the stored mesh
		Eigen::MatrixXd positions() const override;
		/// @brief Exports edges of the stored mesh
		Eigen::MatrixXi edges() const override;
		/// @brief Exports triangles of the stored mesh
		Eigen::MatrixXi triangles() const;
		/// @brief Exports triangles of the stored mesh
		Eigen::MatrixXi elements() const override { return triangles(); }
		/// @brief Exports projected quantities of the stored mesh
		Eigen::MatrixXd projected_quantities() const override;
		/// @brief Exports boundary ids of the stored mesh
		EdgeMap<int> boundary_ids() const override;
		/// @brief Exports body ids of the stored mesh
		std::vector<int> body_ids() const override;

		// ---------------------------------------------------------------------
		// Setters

		/// @brief Set rest positions of the stored mesh
		void set_rest_positions(const Eigen::MatrixXd &positions) override;
		/// @brief Set deformed positions of the stored mesh
		void set_positions(const Eigen::MatrixXd &positions) override;
		/// @brief Set projected quantities of the stored mesh
		void set_projected_quantities(const Eigen::MatrixXd &projected_quantities) override;
		/// @brief Set if a vertex is fixed
		void set_fixed(const std::vector<bool> &fixed) override;
		/// @brief Set the boundary IDs of all edges
		void set_boundary_ids(const EdgeMap<int> &edge_to_boundary_id) override;
		/// @brief Set the body IDs of all triangles
		void set_body_ids(const std::vector<int> &body_ids) override;

		// ---------------------------------------------------------------------

		/// @brief Collect all boundary edge tuples.
		std::vector<Tuple> boundary_edges() const;

		/// @brief Compute the area of a triangle.
		double triangle_area(const Tuple &triangle) const;

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

		/// @brief Check if invariants are satisfied
		bool invariants(const std::vector<Tuple> &new_tris) override;

		wmtk::AttributeCollection<VertexAttributes> vertex_attrs;
		wmtk::AttributeCollection<FaceAttributes> face_attrs;
		wmtk::AttributeCollection<EdgeAttributes> edge_attrs;

	protected:
		/// @brief Create an internal mesh representation and associate attributes
		void create_mesh(const size_t num_vertices, const Eigen::MatrixXi &elements) override;

		/// @brief Get the boundary nodes of the stored mesh
		std::vector<int> boundary_nodes() const override;

		/// @brief Number of projection quantities (not including the position)
		int n_quantities() const override { return m_n_quantities; };

	private:
		/// @brief Check if a triangle is inverted
		bool is_inverted(const Tuple &loc) const;

		/// @brief Compute the length of an edge.
		double edge_length(const Tuple &e) const;

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

		/// @brief Number of projection quantities (not including the position)
		int m_n_quantities = 0;

		// TODO: make this thread local
		OperationCache2D op_cache;
	};

} // namespace polyfem::mesh
