#pragma once

#include <polyfem/mesh/remesh/Remesher.hpp>
#include <polyfem/solver/SolveData.hpp>

#include <wmtk/TriMesh.h>
#include <wmtk/TetMesh.h>
#include <wmtk/ExecutionScheduler.hpp>

#include <type_traits>

namespace polyfem::mesh
{
	enum class CollapseEdgeTo
	{
		V0,
		V1,
		MIDPOINT,
		ILLEGAL
	};

	class TriOperationCache;
	class TetOperationCache;

	template <class WMTKMesh>
	class WildRemesher : public Remesher, public WMTKMesh
	{
		using This = WildRemesher<WMTKMesh>;

		// --------------------------------------------------------------------
		// typedefs
	public:
		// NOTE: This assumes triangle meshes are only used in 2D.
		static constexpr int DIM = [] {
			if constexpr (std::is_same_v<wmtk::TriMesh, WMTKMesh>)
				return 2;
			else
				return 3;
		}();

		static constexpr int VERTICES_PER_ELEMENT = [] {
			if constexpr (std::is_same_v<wmtk::TriMesh, WMTKMesh>)
				return 3;
			else
				return 4;
		}();

		static constexpr int EDGES_PER_ELEMENT = [] {
			if constexpr (std::is_same_v<wmtk::TriMesh, WMTKMesh>)
				return 3;
			else
				return 6;
		}();

		static constexpr int FACETS_PER_ELEMENT = [] {
			if constexpr (std::is_same_v<wmtk::TriMesh, WMTKMesh>)
				return 3;
			else
				return 4;
		}();

		using VectorNd = Eigen::Matrix<double, DIM, 1>;

		using Tuple = typename WMTKMesh::Tuple;

		/// @brief Current execuation policy (sequencial or parallel)
		static constexpr wmtk::ExecutionPolicy EXECUTION_POLICY = wmtk::ExecutionPolicy::kSeq;

		// --------------------------------------------------------------------
		// constructors
	public:
		/// @brief Construct a new WildRemesher object
		/// @param state Simulation current state
		WildRemesher(
			const State &state,
			const Eigen::MatrixXd &obstacle_displacements,
			const Eigen::MatrixXd &obstacle_vals,
			const double current_time,
			const double starting_energy);

		virtual ~WildRemesher() = default;

		/// @brief Initialize the mesh
		/// @param rest_positions Rest positions of the mesh (|V| × 2)
		/// @param positions Current positions of the mesh (|V| × 2)
		/// @param elements Elements of the mesh (|T| × 3)
		/// @param projection_quantities Quantities to be projected to the new mesh (2 rows per vertex and 1 column per quantity)
		/// @param edge_to_boundary_id Map from edge to boundary id (of size |E|)
		/// @param body_ids Body ids of the mesh (of size |T|)
		virtual void init(
			const Eigen::MatrixXd &rest_positions,
			const Eigen::MatrixXd &positions,
			const Eigen::MatrixXi &elements,
			const Eigen::MatrixXd &projection_quantities,
			const BoundaryMap<int> &boundary_to_id,
			const std::vector<int> &body_ids,
			const EdgeMap<double> &elastic_energy,
			const EdgeMap<double> &contact_energy) override;

	protected:
		/// @brief Create an internal mesh representation and associate attributes
		void init_attributes_and_connectivity(
			const size_t num_vertices,
			const Eigen::MatrixXi &elements) override;

		// --------------------------------------------------------------------
		// main functions
	public:
		/// @brief Execute the remeshing
		/// @param split Perform splitting operations
		/// @param collapse Perform collapsing operations
		/// @param smooth Perform smoothing operations
		/// @param swap Perform edge swapping operations
		/// @param max_ops Maximum number of operations to perform (default: unlimited)
		/// @return True if any operation was performed.
		bool execute() override;

		virtual void split_edges() = 0;
		virtual void collapse_edges() = 0;
		virtual void smooth_vertices();
		virtual void swap_edges() { log_and_throw_error("WildRemesher::swap_edges not implemented!"); }

	protected:
		// Edge splitting
		virtual bool split_edge_before(const Tuple &t) override;
		virtual bool split_edge_after(const Tuple &t) override;

		// Edge collapse
		virtual bool collapse_edge_before(const Tuple &t) override;
		virtual bool collapse_edge_after(const Tuple &t) override;

		// Swap edge
		virtual bool swap_edge_before(const Tuple &t) override;
		virtual bool swap_edge_after(const Tuple &t) override;

		// Smooth_vertex
		virtual bool smooth_before(const Tuple &t) override;
		virtual bool smooth_after(const Tuple &t) override;

		/// @brief Check if invariants are satisfied
		bool invariants(const std::vector<Tuple> &new_tris) override;

		/// @brief Check if a triangle's rest shape is inverted
		bool is_rest_inverted(const Tuple &loc) const;
		/// @brief Check if a triangle's rest and deformed shapes are inverted
		bool is_inverted(const Tuple &loc) const;

		// --------------------------------------------------------------------
		// getters
	public:
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
		/// @brief Exports elements of the stored mesh
		Eigen::MatrixXi elements() const override;
		/// @brief Exports boundary edges of the stored mesh
		Eigen::MatrixXi boundary_edges() const override;
		/// @brief Exports boundary faces of the stored mesh
		Eigen::MatrixXi boundary_faces() const override;
		/// @brief Exports projected quantities of the stored mesh
		Eigen::MatrixXd projection_quantities() const override;
		/// @brief Exports boundary ids of the stored mesh
		BoundaryMap<int> boundary_ids() const override;
		/// @brief Exports body ids of the stored mesh
		std::vector<int> body_ids() const override;
		/// @brief Get the boundary nodes of the stored mesh
		std::vector<int> boundary_nodes(const Eigen::VectorXi &vertex_to_basis) const override;

		/// @brief Number of projection quantities (not including the position)
		int n_quantities() const override { return m_n_quantities; };

		/// @brief Get a vector of all facets (edges or triangles)
		std::vector<Tuple> get_facets() const;
		/// @brief Get a vector of all elements (triangles or tetrahedra)
		std::vector<Tuple> get_elements() const;

		// --------------------------------------------------------------------
		// setters
	public:
		/// @brief Set rest positions of the stored mesh
		void set_rest_positions(const Eigen::MatrixXd &positions) override;
		/// @brief Set deformed positions of the stored mesh
		void set_positions(const Eigen::MatrixXd &positions) override;
		/// @brief Set projected quantities of the stored mesh
		void set_projection_quantities(const Eigen::MatrixXd &projection_quantities) override;
		/// @brief Set if a vertex is fixed
		void set_fixed(const std::vector<bool> &fixed) override;
		/// @brief Set the boundary IDs of all edges
		void set_boundary_ids(const BoundaryMap<int> &boundary_to_id) override;
		/// @brief Set the body IDs of all elements
		void set_body_ids(const std::vector<int> &body_ids) override;

		// --------------------------------------------------------------------
		// utilities
	public:
		/// @brief Compute the length of an edge.
		double rest_edge_length(const Tuple &e) const;
		double deformed_edge_length(const Tuple &e) const;

		/// @brief Compute the center of the edge.
		VectorNd rest_edge_center(const Tuple &e) const;
		VectorNd deformed_edge_center(const Tuple &e) const;

		Eigen::VectorXd edge_adjacent_element_volumes(const Tuple &e) const;

		/// @brief Compute the volume (area) of an tetrahedron (triangle) element.
		double element_volume(const Tuple &e) const;

		/// @brief Is the given vertex tuple on the boundary of the mesh?
		bool is_boundary_vertex(const Tuple &v) const;
		/// @brief Is the given vertex tuple on the boundary of a body?
		bool is_body_boundary_vertex(const Tuple &v) const;
		/// @brief Is the given edge tuple on the boundary of the mesh?
		bool is_boundary_edge(const Tuple &e) const;
		/// @brief Is the given edge tuple on the boundary of a body?
		bool is_body_boundary_edge(const Tuple &e) const;
		/// @brief Is the given tuple on the boundary of the mesh?
		bool is_boundary_facet(const Tuple &t) const;
		/// @brief Is the currently cached operation a boundary operation?
		bool is_boundary_op() const;

		/// @brief Get the boundary facets of the mesh
		std::vector<Tuple> boundary_facets(std::vector<int> *boundary_ids = nullptr) const;

		/// @brief Get the vertex tuples of a facet.
		std::array<Tuple, DIM> facet_vertices(const Tuple &t) const;
		/// @brief Get the vertex ids of a facet.
		std::array<size_t, DIM> facet_vids(const Tuple &t) const;

		/// @brief Get the vertex tuples of an element.
		std::array<Tuple, VERTICES_PER_ELEMENT> element_vertices(const Tuple &t) const;
		/// @brief Get the vertex ids of an element.
		std::array<size_t, VERTICES_PER_ELEMENT> element_vids(const Tuple &t) const;

		/// @brief Get a AABB for an element.
		void element_aabb(const Tuple &t, polyfem::VectorNd &el_min, polyfem::VectorNd &el_max) const;

		/// @brief Reorder the element vertices so that the first vertex is v0.
		/// @param conn The element vertices in oriented order
		/// @param v0 The vertex to be the first vertex
		/// @return The element vertices reordered with the same orientation
		std::array<size_t, VERTICES_PER_ELEMENT> orient_preserve_element_reorder(
			const std::array<size_t, VERTICES_PER_ELEMENT> &conn, const size_t v0) const;

		/// @brief Get the one ring of elements around a vertex.
		std::vector<Tuple> get_one_ring_elements_for_vertex(const Tuple &t) const;
		std::vector<Tuple> get_one_ring_boundary_edges_for_vertex(const Tuple &v) const;
		std::array<Tuple, 2> get_boundary_faces_for_edge(const Tuple &e) const;
		std::vector<Tuple> get_one_ring_boundary_faces_for_vertex(const Tuple &v) const;
		std::vector<Tuple> get_edges_for_elements(const std::vector<Tuple> &elements) const;

		/// @brief Get the id of a facet (edge for triangle, triangle for tetrahedra)
		size_t facet_id(const Tuple &t) const;

		/// @brief Get the id of an element (triangle or tetrahedra)
		size_t element_id(const Tuple &t) const;

		/// @brief Get a tuple of an element.
		Tuple tuple_from_element(size_t elem_id) const;
		/// @brief Get a tuple of an element with a local facet
		Tuple tuple_from_facet(size_t elem_id, int local_facet_id) const;

		/// @brief Get the incident elements for an edge
		std::vector<Tuple> get_incident_elements_for_edge(const Tuple &t) const;

		/// @brief Extend the local patch by including neighboring elements
		/// @param patch local patch of elements
		void extend_local_patch(std::vector<Tuple> &patch) const;

		/// @brief Get the opposite vertex on a face
		/// @param e edge tuple
		/// @return vertex tuple
		Tuple opposite_vertex_on_face(const Tuple &e) const
		{
			return e.switch_edge(*this).switch_vertex(*this);
		}

		/// @brief Determine where to collapse an edge to
		/// @param e edge tuple
		/// @return Enumeration of possible collapse locations
		CollapseEdgeTo collapse_boundary_edge_to(const Tuple &e) const;

	protected:
		using Operations = std::vector<std::pair<std::string, Tuple>>;
		virtual Operations renew_neighbor_tuples(
			const std::string &op, const std::vector<Tuple> &tris) const { return {}; }

		/// @brief Cache the split edge operation
		/// @param e edge tuple
		void cache_split_edge(const Tuple &e);

		/// @brief Cache the edge collapse operation
		/// @param e edge tuple
		/// @param collapse_to collapse to which vertex
		void cache_collapse_edge(const Tuple &e, const CollapseEdgeTo collapse_to);

		/// @brief Cache the edge swap operation
		/// @param e edge tuple
		void cache_swap_edge(const Tuple &e);

		// NOTE: Nothing to cache for vertex smoothing

		void map_edge_split_edge_attributes(const Tuple &t);
		void map_edge_split_boundary_attributes(const Tuple &t);
		void map_edge_split_element_attributes(const Tuple &t);

		void map_edge_collapse_vertex_attributes(const Tuple &t);
		void map_edge_collapse_boundary_attributes(const Tuple &t);
		void map_edge_collapse_edge_attributes(const Tuple &t);

		void map_edge_swap_edge_attributes(const Tuple &t);
		void map_edge_swap_element_attributes(const Tuple &t);

		// NOTE: Nothing to map for vertex smoothing

		// --------------------------------------------------------------------
		// members
	public:
		struct VertexAttributes
		{
			using VectorNd = WildRemesher<WMTKMesh>::VectorNd;

			VectorNd rest_position;
			VectorNd position;

			/// @brief Quantities to be projected (dim × n_quantities)
			Eigen::MatrixXd projection_quantities;

			// Can the point be smoothed or moved around by operations?
			bool fixed = false;
			size_t partition_id = 0; // Vertices marked as fixed cannot be modified by any local operation

			/// @brief Current displacement from rest position to current position
			/// @return displacement of the vertex
			VectorNd displacement() const { return position - rest_position; }

			/// @brief Previous position of the vertex
			/// @param i in [0, n_quantities()//3) the i-th previous position
			/// @return previous position i of the vertex
			VectorNd prev_position(const int i) const
			{
				assert(0 <= i && i < projection_quantities.cols() / 3);
				return rest_position + projection_quantities.col(i);
			}

			/// @brief Get the position of the vertex at different times
			/// @param i 0: rest position, 1–(n_quantities()//3): previous position, otherwise: current position
			/// @return position i of the vertex
			VectorNd position_i(const int i) const
			{
				assert(i >= 0);
				if (i == 0)
					return rest_position;
				else if (i - 1 < projection_quantities.cols() / 3)
					return prev_position(i - 1);
				else
					return position;
			}

			static VertexAttributes edge_collapse(
				const VertexAttributes &v0,
				const VertexAttributes &v1,
				const CollapseEdgeTo collapse_to);
		};

		struct EdgeAttributes
		{
			int op_depth = 0;
			int op_attempts = 0;
			// clang-format off
			enum class EnergyRank { BOTTOM, MIDDLE, TOP };
			// clang-format on
			EnergyRank energy_rank = EnergyRank::MIDDLE;
		};

		struct BoundaryAttributes : public EdgeAttributes
		{
			int boundary_id = -1;
		};

		struct ElementAttributes
		{
			int body_id = 0;
		};

		void write_edge_ranks_mesh(
			const EdgeMap<typename EdgeAttributes::EnergyRank> &elastic_ranks,
			const EdgeMap<typename EdgeAttributes::EnergyRank> &contact_ranks) const;

		/// @brief Get a reference to an edge's attributes
		/// @param e_id edge id
		/// @return reference to the edge's attributes
		EdgeAttributes &edge_attr(const size_t e_id);

		/// @brief Get a const reference to an edge's attributes
		/// @param e_id edge id
		/// @return const reference to the edge's attributes
		const EdgeAttributes &edge_attr(const size_t e_id) const;

		wmtk::AttributeCollection<VertexAttributes> vertex_attrs;
		wmtk::AttributeCollection<BoundaryAttributes> boundary_attrs;
		wmtk::AttributeCollection<ElementAttributes> element_attrs;

	protected:
		wmtk::ExecutePass<WildRemesher, EXECUTION_POLICY> executor;
		int m_n_quantities;
		double total_volume;

		// TODO: make this thread local
		typename std::conditional<
			std::is_same<WMTKMesh, wmtk::TriMesh>::value,
			std::shared_ptr<TriOperationCache>,
			std::shared_ptr<TetOperationCache>>::type op_cache;

	private:
		wmtk::AttributeCollection<EdgeAttributes> edge_attrs; // not used for tri mesh
	};

	using WildTriRemesher = WildRemesher<wmtk::TriMesh>;
	using WildTetRemesher = WildRemesher<wmtk::TetMesh>;

} // namespace polyfem::mesh
