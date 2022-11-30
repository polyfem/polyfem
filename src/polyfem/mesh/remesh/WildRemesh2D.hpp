#pragma once

#include <polyfem/State.hpp>

#include <wmtk/TriMesh.h>
#include <wmtk/ExecutionScheduler.hpp>

#include <unordered_map>

namespace polyfem::mesh
{
	class WildRemeshing2D : public wmtk::TriMesh
	{
	public:
		typedef wmtk::TriMesh super;

		/// @brief Construct a new WildRemeshing2D object
		/// @param state Simulation current state
		WildRemeshing2D(const State &state, const Eigen::VectorXd &obstacle_sol)
			: wmtk::TriMesh(), state(state), m_obstacle_displacements(utils::unflatten(obstacle_sol, DIM)) {}

		virtual ~WildRemeshing2D(){};

		/// @brief Dimension of the mesh
		static constexpr int DIM = 2;
		/// @brief Current execuation policy (sequencial or parallel)
		static constexpr wmtk::ExecutionPolicy EXECUTION_POLICY = wmtk::ExecutionPolicy::kSeq;
		/// @brief Map from a (sorted) edge to an integer (ID)
		template <typename T>
		using EdgeMap = std::unordered_map<std::pair<size_t, size_t>, T, polyfem::utils::HashPair>;

		/// @brief Initialize the mesh
		/// @param rest_positions Rest positions of the mesh (|V| × DIM)
		/// @param positions Current positions of the mesh (|V| × DIM)
		/// @param triangles Triangles of the mesh (|T| × 3)
		/// @param projection_quantities Quantities to be projected to the new mesh (DIM rows per vertex and 1 column per quantity)
		/// @param edge_to_boundary_id Map from edge to boundary id (of size |E|)
		/// @param body_ids Body ids of the mesh (of size |T|)
		void init(
			const Eigen::MatrixXd &rest_positions,
			const Eigen::MatrixXd &positions,
			const Eigen::MatrixXi &triangles,
			const Eigen::MatrixXd &projection_quantities,
			const EdgeMap<int> &edge_to_boundary_id,
			const std::vector<int> &body_ids);

		/// @brief Exports rest positions of the stored mesh
		Eigen::MatrixXd rest_positions() const;
		/// @brief Exports positions of the stored mesh
		Eigen::MatrixXd displacements() const;
		/// @brief Exports displacements of the stored mesh
		Eigen::MatrixXd positions() const;
		/// @brief Exports edges of the stored mesh
		Eigen::MatrixXi edges() const;
		/// @brief Exports triangles of the stored mesh
		Eigen::MatrixXi triangles() const;
		/// @brief Exports projected quantities of the stored mesh
		Eigen::MatrixXd projected_quantities() const;
		/// @brief Exports boundary ids of the stored mesh
		EdgeMap<int> boundary_ids() const;
		/// @brief Exports body ids of the stored mesh
		std::vector<int> body_ids() const;

		std::vector<Tuple> boundary_edges() const;
		const Obstacle &obstacle() const { return state.obstacle; }
		const Eigen::MatrixXd &obstacle_displacements() const { return m_obstacle_displacements; }

		/// @brief Set rest positions of the stored mesh
		void set_rest_positions(const Eigen::MatrixXd &positions);
		/// @brief Set deformed positions of the stored mesh
		void set_positions(const Eigen::MatrixXd &positions);
		/// @brief Set projected quantities of the stored mesh
		void set_projected_quantities(const Eigen::MatrixXd &projected_quantities);
		/// @brief Set if a vertex is fixed
		void set_fixed(const std::vector<bool> &fixed);
		/// @brief Set the boundary IDs of all edges
		void set_boundary_ids(const EdgeMap<int> &edge_to_boundary_id);
		/// @brief Set the body IDs of all triangles
		void set_body_ids(const std::vector<int> &body_ids);

		/// @brief Writes a triangle mesh in OBJ format
		/// @param path Output path
		/// @param deformed If true, writes deformed positions, otherwise rest positions
		void write_obj(const std::string &path, bool deformed) const;
		/// @brief Writes a triangle mesh of the rest mesh in OBJ format
		/// @param path Output path
		void write_rest_obj(const std::string &path) const { write_obj(path, false); }
		/// @brief Writes a triangle mesh of the deformed mesh in OBJ format
		/// @param path Output path
		void write_deformed_obj(const std::string &path) const { write_obj(path, true); }

		/// @brief Compute the length of an edge.
		double edge_length(const Tuple &e) const;

		/// @brief Compute the average elastic energy of the faces containing an edge.
		double edge_elastic_energy(const Tuple &e) const;

		/// @brief Check if a triangle is inverted
		bool is_inverted(const Tuple &loc) const;

		/// @brief Check if invariants are satisfied
		bool invariants(const std::vector<Tuple> &new_tris) override;

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
			const double max_ops_percent = -1);

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

		/// @brief Create a vector of all the new edge after an operation.
		/// @param tris New triangles.
		std::vector<Tuple> new_edges_after(const std::vector<Tuple> &tris) const;

		struct VertexAttributes
		{
			Eigen::Vector2d rest_position;
			Eigen::Vector2d position;

			/// @brief Quantaties to be projected (DIM × n_quantities)
			Eigen::MatrixXd projection_quantities;

			bool fixed = false;
			size_t partition_id = 0; // Vertices marked as fixed cannot be modified by any local operation

			Eigen::Vector2d displacement() const { return position - rest_position; }

			// TODO: handle multi-step time integrators
			Eigen::Vector2d prev_displacement() const { return projection_quantities.col(0); }
			Eigen::Vector2d prev_velocity() const { return projection_quantities.col(1); }
			Eigen::Vector2d prev_acceleration() const { return projection_quantities.col(2); }
		};
		wmtk::AttributeCollection<VertexAttributes> vertex_attrs;

		struct FaceAttributes
		{
			int body_id = 0;
		};
		wmtk::AttributeCollection<FaceAttributes> face_attrs;

		struct EdgeAttributes
		{
			int boundary_id = -1;
		};
		wmtk::AttributeCollection<EdgeAttributes> edge_attrs;

		/// @brief Minimum edge length for splitting
		double min_edge_length = 1e-6;
		/// @brief Accept operation if energy decreased by at least (100 * x)%
		double energy_relative_tolerance = 1e-3;
		/// @brief Accept operation if energy decreased by at least x
		double energy_absolute_tolerance = 1e-8;
		/// @brief Size of n-ring for local relaxation
		int n_ring_size = 3;

	protected:
		/// @brief Get the boundary nodes of the stored mesh
		std::vector<int> boundary_nodes() const;

		/// @brief Build bases for a given mesh (V, F)
		/// @param V Matrix of vertex (rest) positions
		/// @param F Matrix of triangle indices
		/// @param bases Output element bases
		/// @param vertex_to_basis Map from vertex to reordered nodes
		/// @return Number of bases
		static int build_bases(
			const Eigen::MatrixXd &V,
			const Eigen::MatrixXi &F,
			const std::string &assembler_formulation,
			std::vector<polyfem::basis::ElementBases> &bases,
			Eigen::VectorXi &vertex_to_basis);

		/// @brief Create an assembler object
		/// @param body_ids One body ID per triangle.
		/// @return Assembler object
		assembler::AssemblerUtils create_assembler(const std::vector<int> &body_ids) const;

		/// @brief Update the mesh positions
		void project_quantities();

		/// @brief Relax a local n-ring around a vertex.
		/// @param t Center of the local n-ring
		/// @param n_ring Size of the n-ring
		/// @return If the local relaxation reduced the energy "significantly"
		bool local_relaxation(const Tuple &t, const int n_ring);

		// --------------------------------------------------------------------

		/// @brief Reference to the simulation state.
		const State &state;
		const Eigen::MatrixXd m_obstacle_displacements;

		/// @brief Number of projection quantities (not including the position)
		int n_quantities;

		/// @brief Cache quantaties before applying an operation
		void cache_before();

		// TODO: Drop this and only use a local EdgeOperationCache
		struct GlobalCache
		{
			/// @brief Rest positions of the mesh before an operation
			Eigen::MatrixXd rest_positions_before;
			/// @brief Deformed positions of the mesh before an operation
			Eigen::MatrixXd positions_before;
			/// @brief Triangled before an operation
			Eigen::MatrixXi triangles_before;
			/// @brief DIM rows per vertex and 1 column per quantity
			Eigen::MatrixXd projected_quantities_before;
			/// @brief Energy before an operation
			double energy_before;
		};
		GlobalCache global_cache;

		class EdgeOperationCache
		{
		public:
			/// @brief Construct a local mesh as an n-ring around a vertex.
			static EdgeOperationCache split(WildRemeshing2D &m, const Tuple &t);
			static EdgeOperationCache swap(WildRemeshing2D &m, const Tuple &t);
			static EdgeOperationCache collapse(WildRemeshing2D &m, const Tuple &t);

			const std::pair<size_t, VertexAttributes> &v0() const { return m_v0; }
			const std::pair<size_t, VertexAttributes> &v1() const { return m_v1; }
			const EdgeMap<EdgeAttributes> &edges() const { return m_edges; }
			const std::vector<FaceAttributes> &faces() const { return m_faces; }

		protected:
			std::pair<size_t, VertexAttributes> m_v0;
			std::pair<size_t, VertexAttributes> m_v1;
			EdgeMap<EdgeAttributes> m_edges;
			std::vector<FaceAttributes> m_faces;
		};

		// TODO: make this thread local
		EdgeOperationCache edge_cache;
	};

} // namespace polyfem::mesh
