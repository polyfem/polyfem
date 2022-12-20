#pragma once

#include <polyfem/mesh/remesh/wild_remesh/Timings.hpp>
#include <polyfem/State.hpp>

#include <wmtk/ExecutionScheduler.hpp>

#include <unordered_map>

namespace polyfem::mesh
{
	class WildRemeshing
	{
	public:
		/// @brief Construct a new WildRemeshing2D object
		/// @param state Simulation current state
		WildRemeshing(
			const State &state,
			const Eigen::VectorXd &obstacle_displacements,
			const Eigen::MatrixXd &obstacle_vals,
			const double current_time,
			const double starting_energy)
			: state(state),
			  m_obstacle_displacements(obstacle_displacements),
			  m_obstacle_vals(obstacle_vals),
			  current_time(current_time),
			  starting_energy(starting_energy)
		{
		}

		virtual ~WildRemeshing(){};

		/// @brief Current execuation policy (sequencial or parallel)
		static constexpr wmtk::ExecutionPolicy EXECUTION_POLICY = wmtk::ExecutionPolicy::kSeq;
		/// @brief Map from a (sorted) edge to an integer (ID)
		template <typename T>
		using EdgeMap = std::unordered_map<std::pair<size_t, size_t>, T, polyfem::utils::HashPair>;
		/// @brief Map from a (sorted) edge to an integer (ID)
		template <typename T>
		using FaceMap = std::unordered_map<std::vector<size_t>, T, polyfem::utils::HashVector>;

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
			const EdgeMap<int> &edge_to_boundary_id, // TODO: this has to change for 3D
			const std::vector<int> &body_ids);

		/// @brief Dimension of the mesh
		virtual int dim() const = 0;
		bool is_volume() const { return dim() == 3; }
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
		virtual Eigen::MatrixXd projected_quantities() const = 0;
		/// @brief Exports boundary ids of the stored mesh
		/// @todo This has to change for 3D. Use a FaceMap<int>.
		virtual EdgeMap<int> boundary_ids() const = 0;
		/// @brief Exports body ids of the stored mesh
		virtual std::vector<int> body_ids() const = 0;

		const Obstacle &obstacle() const { return state.obstacle; }
		const Eigen::MatrixXd &obstacle_displacements() const { return m_obstacle_displacements; }
		// TODO: this does not handle multi-stepping
		Eigen::MatrixXd obstacle_prev_displacement() const { return utils::unflatten(m_obstacle_vals.col(0), dim()); }
		Eigen::MatrixXd obstacle_prev_velocities() const { return utils::unflatten(m_obstacle_vals.col(1), dim()); }
		Eigen::MatrixXd obstacle_prev_accelerations() const { return utils::unflatten(m_obstacle_vals.col(2), dim()); }
		Eigen::MatrixXd obstacle_friction_gradient() const { return utils::unflatten(m_obstacle_vals.col(3), dim()); }

		/// @brief Set rest positions of the stored mesh
		virtual void set_rest_positions(const Eigen::MatrixXd &positions) = 0;
		/// @brief Set deformed positions of the stored mesh
		virtual void set_positions(const Eigen::MatrixXd &positions) = 0;
		/// @brief Set projected quantities of the stored mesh
		virtual void set_projected_quantities(const Eigen::MatrixXd &projected_quantities) = 0;
		/// @brief Set if a vertex is fixed
		virtual void set_fixed(const std::vector<bool> &fixed) = 0;
		/// @brief Set the boundary IDs of all edges
		/// @todo This has to change for 3D. Use a FaceMap<int>.
		virtual void set_boundary_ids(const EdgeMap<int> &edge_to_boundary_id) = 0;
		/// @brief Set the body IDs of all elements
		virtual void set_body_ids(const std::vector<int> &body_ids) = 0;

		/// @brief Execute the remeshing
		/// @param split Perform splitting operations
		/// @param collapse Perform collapsing operations
		/// @param smooth Perform smoothing operations
		/// @param swap Perform edge swapping operations
		/// @param max_ops Maximum number of operations to perform (default: unlimited)
		/// @return True if any operation was performed.
		virtual bool execute(
			const bool split = true,
			const bool collapse = false,
			const bool smooth = false,
			const bool swap = false,
			const double max_ops_percent = -1) = 0;

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

		template <int DIM>
		struct VertexAttributes
		{
			using VectorNd = Eigen::Matrix<double, DIM, 1>;

			VectorNd rest_position;
			VectorNd position;

			/// @brief Quantities to be projected (dim × n_quantities)
			Eigen::MatrixXd projection_quantities;

			bool fixed = false;
			size_t partition_id = 0; // Vertices marked as fixed cannot be modified by any local operation

			VectorNd displacement() const { return position - rest_position; }

			// TODO: handle multi-step time integrators
			VectorNd prev_displacement() const { return projection_quantities.col(0); }
			VectorNd prev_velocity() const { return projection_quantities.col(1); }
			VectorNd prev_acceleration() const { return projection_quantities.col(2); }
			VectorNd friction_gradient() const { return projection_quantities.col(3); }
		};

		struct BoundaryAttributes
		{
			int boundary_id = -1;
		};

		struct ElementAttributes
		{
			int body_id = 0;
		};

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

		/// @brief Timers for the remeshing operations.
		mutable WildRemeshingTimings timings;

	protected:
		/// @brief Create an internal mesh representation and associate attributes
		virtual void create_mesh(const size_t num_vertices, const Eigen::MatrixXi &elements) = 0;

		/// @brief Get the boundary nodes of the stored mesh
		virtual std::vector<int> boundary_nodes() const = 0;

		/// @brief Number of projection quantities (not including the position)
		virtual int n_quantities() const = 0;

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

		/// @brief Create an assembler object
		/// @param body_ids One body ID per element.
		/// @return Assembler object
		assembler::AssemblerUtils create_assembler(const std::vector<int> &body_ids) const;

		/// @brief Update the mesh positions
		void project_quantities();

		/// @brief Relax a local n-ring around a vertex.
		/// @param t Center of the local n-ring
		/// @param n_ring Size of the n-ring
		/// @return If the local relaxation reduced the energy "significantly"
		// bool local_relaxation(const Tuple &t, const int n_ring);

		// --------------------------------------------------------------------

		/// @brief Reference to the simulation state.
		const State &state;
		const Eigen::MatrixXd m_obstacle_displacements;
		Eigen::MatrixXd m_obstacle_vals;
		const double current_time;
		const double starting_energy;

		/// @brief Cache quantities before applying an operation
		void cache_before();

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
			Eigen::MatrixXd projected_quantities_before;
			/// @brief Energy before an operation
			double energy_before;
		};
		GlobalCache global_cache;
	};

} // namespace polyfem::mesh
