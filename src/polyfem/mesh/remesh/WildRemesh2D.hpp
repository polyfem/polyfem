#pragma once

#include <polyfem/State.hpp>

#include <wmtk/TriMesh.h>
#include <wmtk/ExecutionScheduler.hpp>

namespace polyfem::mesh
{
	class WildRemeshing2D : public wmtk::TriMesh
	{
	public:
		typedef wmtk::TriMesh super;

		/// @brief Construct a new WildRemeshing2D object
		/// @param state Simulation current state
		WildRemeshing2D(const State &state) : wmtk::TriMesh(), state(state) {}

		virtual ~WildRemeshing2D(){};

		/// @brief Dimension of the mesh
		static constexpr int DIM = 2;
		/// @brief Current execuation policy (sequencial or parallel)
		static constexpr wmtk::ExecutionPolicy EXECUTION_POLICY = wmtk::ExecutionPolicy::kSeq;
		/// @brief Map from a (sorted) edge to an integer (ID)
		using EdgeMap = std::unordered_map<std::pair<int, int>, int, polyfem::utils::HashPair>;

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
			const EdgeMap &edge_to_boundary_id,
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
		EdgeMap boundary_ids() const;
		/// @brief Exports body ids of the stored mesh
		std::vector<int> body_ids() const;

		/// @brief Set rest positions of the stored mesh
		void set_rest_positions(const Eigen::MatrixXd &positions);
		/// @brief Set deformed positions of the stored mesh
		void set_positions(const Eigen::MatrixXd &positions);
		/// @brief Set projected quantities of the stored mesh
		void set_projected_quantities(const Eigen::MatrixXd &projected_quantities);
		/// @brief Set if a vertex is fixed
		void set_fixed(const std::vector<bool> &fixed);
		/// @brief Set the boundary IDs of all edges
		void set_boundary_ids(const EdgeMap &edge_to_boundary_id);
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

		/// @brief Compute the global energy of the mesh
		double compute_global_energy() const;
		double compute_global_wicke_measure() const;

		/// @brief Check if a triangle is inverted
		bool is_inverted(const Tuple &loc) const;

		/// @brief Check if invariants are satisfied
		bool invariants(const std::vector<Tuple> &new_tris) override;

		/// @brief Update the mesh positions
		void update_positions();

		// Smoothing
		void smooth_all_vertices();
		bool smooth_before(const Tuple &t) override;
		bool smooth_after(const Tuple &t) override;

		// Edge splitting
		void split_all_edges();
		bool split_edge_before(const Tuple &t) override;
		bool split_edge_after(const Tuple &t) override;

		// Edge collapse
		void collapse_all_edges();
		bool collapse_edge_before(const Tuple &t) override;
		bool collapse_edge_after(const Tuple &t) override;

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

	protected:
		std::vector<Tuple> get_n_ring_tris_for_vertex(const Tuple &root, int n) const;

		double local_relaxation(const Tuple &t, const int n_ring);

		void build_local_matricies(
			const std::vector<Tuple> &tris,
			Eigen::MatrixXd &V, // rest positions
			Eigen::MatrixXd &U, // displacement
			Eigen::MatrixXi &F, // triangles as vertex indices
			std::unordered_map<size_t, size_t> &local_to_global,
			std::vector<int> &body_ids) const;

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

		assembler::AssemblerUtils create_assembler(const std::vector<int> &body_ids) const;

		const State &state;

		/// @brief Number of projection quantities (not including the position)
		int n_quantities;

		/// @brief Cache quantaties before applying an operation
		void cache_before();

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

		struct EdgeCache
		{
			EdgeCache() = default;
			EdgeCache(const WildRemeshing2D &m, const Tuple &t);

			VertexAttributes v0;
			VertexAttributes v1;
			std::vector<EdgeAttributes> edges;
			std::vector<FaceAttributes> faces;
		};
		EdgeCache edge_cache;
	};

} // namespace polyfem::mesh
