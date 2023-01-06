#pragma once

#include <polyfem/mesh/remesh/wild_remesh/Timings.hpp>
#include <polyfem/State.hpp>

#include <unordered_map>

namespace polyfem::time_integrator
{
	class ImplicitTimeIntegrator;
} // namespace polyfem::time_integrator

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
		/// @brief Construct a new WildTriRemesher object
		/// @param state Simulation current state
		Remesher(const State &state,
				 const Eigen::MatrixXd &obstacle_displacements,
				 const Eigen::MatrixXd &obstacle_quantities,
				 const double current_time,
				 const double starting_energy);

		virtual ~Remesher(){};

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
			const Eigen::VectorXd &element_energies);

	protected:
		/// @brief Create an internal mesh representation and associate attributes
		virtual void init_attributes_and_connectivity(
			const size_t num_vertices,
			const Eigen::MatrixXi &elements) = 0;

		// --------------------------------------------------------------------
		// main functions
	public:
		/// @brief Execute the remeshing
		/// @param split Perform splitting operations
		/// @param collapse Perform collapsing operations
		/// @param smooth Perform smoothing operations
		/// @param swap Perform edge swapping operations
		/// @param max_ops_percent Maximum number of operations to perform (default: unlimited)
		/// @return True if any operation was performed.
		virtual bool execute(
			const bool split = true,
			const bool collapse = false,
			const bool smooth = false,
			const bool swap = false,
			const double max_ops_percent = -1) = 0;

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
		/// @brief Exports projected quantities of the stored mesh
		virtual Eigen::MatrixXd projection_quantities() const = 0;
		/// @brief Exports boundary ids of the stored mesh
		virtual BoundaryMap<int> boundary_ids() const = 0;
		/// @brief Exports body ids of the stored mesh
		virtual std::vector<int> body_ids() const = 0;
		/// @brief Get the boundary nodes of the stored mesh
		virtual std::vector<int> boundary_nodes() const = 0;

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
		/// @brief Writes a mesh file.
		/// @param path Output path
		/// @param deformed If true, writes deformed positions, otherwise rest positions
		void write_mesh(const std::string &path, bool deformed) const;
		/// @brief Writes a mesh file of the rest mesh.
		/// @param path Output path
		void write_rest_mesh(const std::string &path) const { write_mesh(path, false); }
		/// @brief Writes a mesh file of the deformed mesh.
		/// @param path Output path
		void write_deformed_mesh(const std::string &path) const { write_mesh(path, true); }

		/// @brief Combine the quantities of a time integrator into a single matrix (one column per quantity)
		static Eigen::MatrixXd combine_time_integrator_quantities(
			const std::shared_ptr<time_integrator::ImplicitTimeIntegrator> &time_integrator);

		/// @brief Split the quantities of a time integrator into separate vectors
		static void split_time_integrator_quantities(
			const Eigen::MatrixXd &quantities,
			const int dim,
			std::vector<Eigen::VectorXd> &x_prevs,
			std::vector<Eigen::VectorXd> &v_prevs,
			std::vector<Eigen::VectorXd> &a_prevs);

		/// @brief Timers for the remeshing operations.
		mutable RemesherTimings timings;

	protected:
		/// @brief Create an assembler object
		/// @param body_ids One body ID per element.
		/// @return Assembler object
		assembler::AssemblerUtils create_assembler(const std::vector<int> &body_ids) const;

		/// @brief Build bases for a given mesh (V, F)
		/// @param V Matrix of vertex (rest) positions
		/// @param F Matrix of elements indices
		/// @param bases Output element bases
		/// @param vertex_to_basis Map from vertex to reordered nodes
		/// @return Number of bases
		static int build_bases(
			const Eigen::MatrixXd &V,
			const Eigen::MatrixXi &F,
			const std::string &assembler_formulation,
			std::vector<polyfem::basis::ElementBases> &bases,
			Eigen::VectorXi &vertex_to_basis);

		// --------------------------------------------------------------------
		// members
	public:
		/// @brief Minimum edge length for splitting
		double min_edge_length = 1e-6;
		/// @brief Accept operation if energy decreased by at least (100 * x)%
		double energy_relative_tolerance = 1e-3;
		/// @brief Accept operation if energy decreased by at least x
		double energy_absolute_tolerance = 1e-8;
		/// @brief Size of n-ring for local relaxation
		int n_ring_size = 3;
		/// @brief Flood fill relative area
		double flood_fill_rel_area = 0.1;
		double threshold = 0;
		int max_split_depth = 3;

		// --------------------------------------------------------------------
		// members
	protected:
		// TODO: Drop this and only use a local EdgeOperationCache
		struct GlobalCache
		{
			/// @brief Rest positions of the mesh before an operation
			Eigen::MatrixXd rest_positions_before;
			/// @brief Deformed positions of the mesh before an operation
			Eigen::MatrixXd positions_before;
			/// @brief Elements before an operation
			Eigen::MatrixXi elements_before;
			/// @brief dim rows per vertex and 1 column per quantity
			Eigen::MatrixXd projection_quantities_before;
			/// @brief Energy before an operation
			double energy_before;
		};

		GlobalCache global_cache;

		/// @brief Reference to the simulation state.
		const State &state;
		/// @brief Collision obstacles' displacements
		const Eigen::MatrixXd m_obstacle_displacements;
		/// @brief Collision obstacles' extra quantities
		Eigen::MatrixXd m_obstacle_quantities;
		/// @brief Current time
		const double current_time;
		/// @brief Starting energy
		const double starting_energy;
	};

} // namespace polyfem::mesh
