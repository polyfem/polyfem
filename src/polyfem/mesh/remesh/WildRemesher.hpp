#pragma once

#include <polyfem/mesh/remesh/Remesher.hpp>
#include <polyfem/mesh/remesh/wild_remesh/LocalMesh.hpp>
#include <polyfem/solver/SolveData.hpp>

#include <wmtk/TriMesh.h>
#include <wmtk/TetMesh.h>
#include <wmtk/ExecutionScheduler.hpp>

#include <ipc/collision_mesh.hpp>

namespace polyfem::mesh
{
	enum class CollapseEdgeTo
	{
		V0,
		V1,
		MIDPOINT,
		ILLEGAL
	};

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

		virtual ~WildRemesher(){};

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
		bool execute(
			const bool split = true,
			const bool collapse = false,
			const bool smooth = false,
			const bool swap = false) override;

		void split_edges();
		void collapse_edges();
		void smooth_vertices() { throw std::runtime_error("Not implemented!"); }
		void swap_edges() { throw std::runtime_error("Not implemented!"); }

	protected:
		// Edge splitting
		bool split_edge_before(const Tuple &t) override;

		// Edge collapse
		bool collapse_edge_before(const Tuple &t) override;
		bool collapse_edge_after(const Tuple &t) override;

		/// @brief Check if invariants are satisfied
		bool invariants(const std::vector<Tuple> &new_tris) override;

		/// @brief Check if a triangle is inverted
		virtual bool is_inverted(const Tuple &loc) const = 0;

		/// @brief Relax a local n-ring around a vertex.
		/// @param t Center of the local n-ring
		/// @return If the local relaxation reduced the energy "significantly"
		bool local_relaxation(
			const Tuple &t,
			const double local_energy_before,
			const double acceptance_tolerance);

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
		virtual Eigen::MatrixXi boundary_edges() const = 0;
		/// @brief Exports boundary faces of the stored mesh
		virtual Eigen::MatrixXi boundary_faces() const = 0;
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

		/// @brief Get a vector of all elements (elements or tetrahedra)
		virtual std::vector<Tuple> get_elements() const = 0;

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
		double edge_length(const Tuple &e) const;

		/// @brief Compute the center of the edge.
		VectorNd edge_center(const Tuple &e) const;

		Eigen::VectorXd edge_adjacent_element_volumes(const Tuple &e) const;

		/// @brief Compute the average elastic energy of the faces containing an edge.
		double edge_elastic_energy(const Tuple &e) const;

		/// @brief Compute the volume (area) of an tetrahedron (triangle) element.
		virtual double element_volume(const Tuple &e) const = 0;

		/// @brief Is the given tuple on the boundary of the mesh?
		virtual bool is_on_boundary(const Tuple &t) const = 0;

		/// @brief Get the boundary facets of the mesh
		virtual std::vector<Tuple> boundary_facets() const = 0;

		/// @brief Get the vertex ids of a boundary facet.
		virtual std::array<size_t, DIM> boundary_facet_vids(const Tuple &t) const = 0;

		/// @brief Get the vertex ids of an element.
		virtual std::array<Tuple, DIM + 1> element_vertices(const Tuple &t) const = 0;

		/// @brief Get the vertex ids of an element.
		virtual std::array<size_t, DIM + 1> element_vids(const Tuple &t) const = 0;

		/// @brief Get the one ring of elements around a vertex.
		virtual std::vector<Tuple> get_one_ring_elements_for_vertex(const Tuple &t) const = 0;

		/// @brief Get the id of a facet (edge for triangle, triangle for tetrahedra)
		virtual size_t facet_id(const Tuple &t) const = 0;

		/// @brief Get the id of an element (triangle or tetrahedra)
		virtual size_t element_id(const Tuple &t) const = 0;

		/// @brief Get a tuple of a element with a local facet
		virtual Tuple tuple_from_facet(size_t elem_id, int local_facet_id) const = 0;

		/// @brief Get the incident elements for an edge
		virtual std::vector<Tuple> get_incident_elements_for_edge(const Tuple &t) const = 0;

		void extend_local_patch(std::vector<Tuple> &patch) const;

		std::vector<Tuple> local_mesh_tuples(const VectorNd &center) const;
		std::vector<Tuple> local_mesh_tuples(const Tuple &t) const
		{
			return local_mesh_tuples(vertex_attrs[t.vid(*this)].rest_position);
		}

		double local_mesh_energy(const VectorNd &local_mesh_center) const;

		virtual bool is_edge_on_body_boundary(const Tuple &e) const = 0;
		virtual bool is_vertex_on_boundary(const Tuple &v) const = 0;
		virtual bool is_vertex_on_body_boundary(const Tuple &v) const = 0;

		virtual CollapseEdgeTo collapse_boundary_edge_to(const Tuple &e) const = 0;

	protected:
		/// @brief Write a visualization mesh of the priority queue
		/// @param e current edge tuple to be split
		void write_priority_queue_mesh(const std::string &path, const Tuple &e) const;

		using Operations = std::vector<std::pair<std::string, Tuple>>;
		Operations renew_neighbor_tuples(
			const std::string &op, const std::vector<Tuple> &tris) const;

		std::vector<polyfem::basis::ElementBases> local_bases(LocalMesh<This> &local_mesh) const;

		std::vector<int> local_boundary_nodes(const LocalMesh<This> &local_mesh) const;

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

		virtual double local_energy() const = 0;

		// edge split

		/// @brief Cache the split edge operation
		/// @param e edge tuple
		/// @param local_energy local energy
		virtual void cache_split_edge(const Tuple &e, const double local_energy) = 0;

		// edge collapse

		/// @brief Cache the edge collapse operation
		/// @param e edge tuple
		/// @param local_energy local energy
		/// @param collapse_to collapse to which vertex
		virtual void cache_collapse_edge(const Tuple &e, const double local_energy, const CollapseEdgeTo collapse_to) = 0;

		/// @brief Map the vertex attributes for edge collapse
		/// @param t new vertex tuple
		virtual void map_edge_collapse_vertex_attributes(const Tuple &t) = 0;

		/// @brief Map the edge attributes for edge collapse
		/// @param t new vertex tuple
		virtual void map_edge_collapse_boundary_attributes(const Tuple &t) = 0;

		// No need to map boundary attributes for edge collapse because no new boundary is created
		// No need to map element attributes for edge collapse because no new element is created

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
		virtual EdgeAttributes &edge_attr(const size_t e_id) = 0;

		/// @brief Get a const reference to an edge's attributes
		/// @param e_id edge id
		/// @return const reference to the edge's attributes
		virtual const EdgeAttributes &edge_attr(const size_t e_id) const = 0;

		wmtk::AttributeCollection<VertexAttributes> vertex_attrs;
		wmtk::AttributeCollection<BoundaryAttributes> boundary_attrs;
		wmtk::AttributeCollection<ElementAttributes> element_attrs;

	protected:
		wmtk::ExecutePass<WildRemesher, EXECUTION_POLICY> executor;
		int vis_counter = 0;
		int m_n_quantities;
		double total_volume;
	};

} // namespace polyfem::mesh