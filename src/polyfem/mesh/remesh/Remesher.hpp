#pragma once

#include <polyfem/State.hpp>
#include <polyfem/utils/Types.hpp>
#include <polyfem/utils/Timer.hpp>

#include <unordered_map>
#include <variant>

namespace polyfem::time_integrator
{
	class ImplicitTimeIntegrator;
} // namespace polyfem::time_integrator

#define POLYFEM_REMESHER_SCOPED_TIMER(name) polyfem::utils::Timer __polyfem_timer(Remesher::timings[name])

namespace polyfem::mesh
{
	class Remesher
	{
		// --------------------------------------------------------------------
		// typedefs
	public:
		/// @brief Map from a (sorted) edge to an integer (ID)
		template <typename T>
		using EdgeMap = std::unordered_map<
			std::array<size_t, 2>, T,
			polyfem::utils::HashUnorderedArray<size_t, 2>,
			polyfem::utils::EqualUnorderedArray<size_t, 2>>;
		/// @brief Map from a (sorted) edge to an integer (ID)
		template <typename T>
		using FaceMap = std::unordered_map<
			std::array<size_t, 3>, T,
			polyfem::utils::HashUnorderedArray<size_t, 3>,
			polyfem::utils::EqualUnorderedArray<size_t, 3>>;
		template <typename T>
		using TetMap = std::unordered_map<
			std::array<size_t, 4>, T,
			polyfem::utils::HashUnorderedArray<size_t, 4>,
			polyfem::utils::EqualUnorderedArray<size_t, 4>>;
		template <typename T>
		using BoundaryMap = std::variant<EdgeMap<T>, FaceMap<T>>;

		// --------------------------------------------------------------------
		// constructors
	public:
		/// @brief Construct a new Remesher object
		/// @param state Simulation current state
		Remesher(const State &state,
				 const Eigen::MatrixXd &obstacle_displacements,
				 const Eigen::MatrixXd &obstacle_quantities,
				 const double current_time,
				 const double starting_energy);

		virtual ~Remesher() = default;

		/// @brief Initialize the mesh
		/// @param rest_positions Rest positions of the mesh (|V| × dim)
		/// @param positions Current positions of the mesh (|V| × dim)
		/// @param elements Elements of the mesh (|T| × (dim + 1))
		/// @param projection_quantities Quantities to be projected to the new mesh (dim rows per vertex and 1 column per quantity)
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
			const EdgeMap<double> &contact_energy);

	protected:
		/// @brief Create an internal mesh representation and associate attributes
		virtual void init_attributes_and_connectivity(
			const size_t num_vertices,
			const Eigen::MatrixXi &elements) = 0;

		// --------------------------------------------------------------------
		// main functions
	public:
		/// @brief Execute the remeshing
		/// @return True if any operation was performed.
		virtual bool execute() = 0;

	protected:
		/// @brief Update the mesh positions and other projection quantities
		void project_quantities();

		/// @brief Cache quantities before applying an operation
		void cache_before();

		// --------------------------------------------------------------------
		// getters
	public:
		/// @brief Dimension of the mesh
		virtual int dim() const = 0;

		/// @brief Is the mesh a volumetric mesh
		/// @note Assumes non-volumetric meshes are 2D
		virtual bool is_volume() const { return dim() == 3; }

		/// @brief Exports rest positions of the stored mesh
		virtual Eigen::MatrixXd rest_positions() const = 0;
		/// @brief Exports positions of the stored mesh
		virtual Eigen::MatrixXd displacements() const = 0;
		/// @brief Exports displacements of the stored mesh
		virtual Eigen::MatrixXd positions() const = 0;
		/// @brief Exports edges of the stored mesh
		virtual Eigen::MatrixXi edges() const = 0;
		/// @brief Exports elements of the stored mesh
		virtual Eigen::MatrixXi elements() const = 0;
		/// @brief Exports boundary edges of the stored mesh
		virtual Eigen::MatrixXi boundary_edges() const = 0;
		/// @brief Exports boundary faces of the stored mesh
		virtual Eigen::MatrixXi boundary_faces() const = 0;
		/// @brief Exports projected quantities of the stored mesh
		virtual Eigen::MatrixXd projection_quantities() const = 0;
		/// @brief Exports boundary ids of the stored mesh
		virtual BoundaryMap<int> boundary_ids() const = 0;
		/// @brief Exports body ids of the stored mesh
		virtual std::vector<int> body_ids() const = 0;
		/// @brief Get the boundary nodes of the stored mesh
		virtual std::vector<int> boundary_nodes(const Eigen::VectorXi &vertex_to_basis) const = 0;

		/// @brief Number of projection quantities (not including the position)
		virtual int n_quantities() const = 0;

		/// @brief Get a reference to the collision obstacles
		const Obstacle &obstacle() const { return state.obstacle; }
		/// @brief Get a reference to the collision obstacles' displacements
		const Eigen::MatrixXd &obstacle_displacements() const { return m_obstacle_displacements; }
		/// @brief Get a reference to the collision obstacles' extra quantities
		const Eigen::MatrixXd &obstacle_quantities() const { return m_obstacle_quantities; }

		// --------------------------------------------------------------------
		// setters
	public:
		/// @brief Set rest positions of the stored mesh
		virtual void set_rest_positions(const Eigen::MatrixXd &positions) = 0;
		/// @brief Set deformed positions of the stored mesh
		virtual void set_positions(const Eigen::MatrixXd &positions) = 0;
		/// @brief Set projected quantities of the stored mesh
		virtual void set_projection_quantities(const Eigen::MatrixXd &projection_quantities) = 0;
		/// @brief Set if a vertex is fixed
		virtual void set_fixed(const std::vector<bool> &fixed) = 0;
		/// @brief Set the boundary IDs of all edges
		virtual void set_boundary_ids(const BoundaryMap<int> &boundary_to_id) = 0;
		/// @brief Set the body IDs of all elements
		virtual void set_body_ids(const std::vector<int> &body_ids) = 0;

		// --------------------------------------------------------------------
		// utilities
	public:
		/// @brief Writes a VTU mesh file.
		/// @param path Output path
		void write_mesh(const std::string &path) const;

		/// @brief Combine the quantities of a time integrator into a single matrix (one column per quantity)
		static Eigen::MatrixXd combine_time_integrator_quantities(
			const std::shared_ptr<time_integrator::ImplicitTimeIntegrator> &time_integrator);

		/// @brief Split the quantities of a time integrator into separate vectors
		static void split_time_integrator_quantities(
			const Eigen::MatrixXd &quantities,
			const int dim,
			Eigen::MatrixXd &x_prevs,
			Eigen::MatrixXd &v_prevs,
			Eigen::MatrixXd &a_prevs);

		/// @brief Create an assembler object
		/// @param body_ids One body ID per element.
		/// @return Assembler object
		void init_assembler(const std::vector<int> &body_ids) const;

		/// @brief Build bases for a given mesh (V, F)
		/// @param V Matrix of vertex (rest) positions
		/// @param F Matrix of elements indices
		/// @param bases Output element bases
		/// @param vertex_to_basis Map from vertex to reordered nodes
		/// @return Number of bases
		static int build_bases(
			const Mesh &mesh,
			const std::string &assembler_formulation,
			std::vector<polyfem::basis::ElementBases> &bases,
			std::vector<LocalBoundary> &local_boundary,
			Eigen::VectorXi &vertex_to_basis);

		/// @brief Reference to the simulation state.
		const State &state;

		// --------------------------------------------------------------------
		// members
	public:
		int max_op_attempts = 1;

	protected:
		// TODO: Drop this and only use a local EdgeOperationCache
		struct GlobalProjectionCache
		{
			/// @brief Rest positions of the mesh before an operation
			Eigen::MatrixXd rest_positions;
			/// @brief Elements before an operation
			Eigen::MatrixXi elements;
			/// @brief dim rows per vertex and 1 column per quantity
			Eigen::MatrixXd projection_quantities;
		};

		GlobalProjectionCache global_projection_cache;

		/// @brief Copy of remesh args.
		const json args;
		/// @brief Collision obstacles' displacements
		const Eigen::MatrixXd m_obstacle_displacements;
		/// @brief Collision obstacles' extra quantities
		Eigen::MatrixXd m_obstacle_quantities;
		/// @brief Current time
		const double current_time;
		/// @brief Starting energy
		const double starting_energy;

		// --------------------------------------------------------------------
		// statistics
	public:
		static void log_timings();

		/// @brief Timings for the remeshing operations.
		static std::unordered_map<std::string, utils::Timing> timings;
		static double total_time;  // = 0;
		static size_t num_solves;  // = 0;
		static size_t total_ndofs; // = 0;
	};

} // namespace polyfem::mesh
