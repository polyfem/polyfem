#pragma once

#include <polyfem/Common.hpp>

#include <polyfem/Units.hpp>

#include <polyfem/basis/ElementBases.hpp>
#include <polyfem/basis/InterfaceData.hpp>

#include <polyfem/assembler/ElementAssemblyValues.hpp>
#include <polyfem/assembler/AssemblyValsCache.hpp>
#include <polyfem/assembler/RhsAssembler.hpp>
#include <polyfem/assembler/local/PressureAssembler.hpp>
#include <polyfem/assembler/MacroStrain.hpp>
#include <polyfem/assembler/Problem.hpp>
#include <polyfem/assembler/Assembler.hpp>
#include <polyfem/assembler/AssemblerUtils.hpp>

#include <polyfem/mesh/Mesh.hpp>
#include <polyfem/mesh/Obstacle.hpp>
#include <polyfem/mesh/MeshNodes.hpp>
#include <polyfem/mesh/LocalBoundary.hpp>

#include <polyfem/solver/SolveData.hpp>
#include <polyfem/solver/DiffCache.hpp>

#include <polyfem/utils/StringUtils.hpp>
#include <polyfem/utils/ElasticityUtils.hpp>
#include <polyfem/utils/JSONUtils.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/assembler/PeriodicBoundary.hpp>

#include <polyfem/io/OutData.hpp>

#include <polysolve/linear/Solver.hpp>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <memory>
#include <string>
#include <unordered_map>

#include <spdlog/sinks/basic_file_sink.h>

#include <ipc/collision_mesh.hpp>
#include <ipc/utils/logger.hpp>

// Forward declaration
namespace polysolve::nonlinear
{
	class Solver;
}

namespace polyfem::assembler
{
	class Mass;
	class ViscousDamping;
	class ViscousDampingPrev;
} // namespace polyfem::assembler

namespace polyfem
{
	namespace mesh
	{
		class Mesh2D;
		class Mesh3D;
	} // namespace mesh

	enum class CacheLevel
	{
		None,
		Solution,
		Derivatives
	};

	class VariationalForm;

	/// main class that contains the polyfem solver and all its state
	class State
	{
	public:
		//---------------------------------------------------
		//-----------------initialization--------------------
		//---------------------------------------------------

		~State() = default;
		/// Constructor
		State();

		/// @param[in] max_threads max number of threads
		void set_max_threads(const int max_threads = std::numeric_limits<int>::max());

		/// initialize the polyfem solver with a json settings
		/// @param[in] args input arguments
		/// @param[in] strict_validation strict validation of input
		void init(const json &args, const bool strict_validation);

		/// initialize time settings if args contains "time"
		void init_time();

		/// main input arguments containing all defaults
		json args;

		//---------------------------------------------------
		//-----------------logger----------------------------
		//---------------------------------------------------

		/// initializing the logger
		/// @param[in] log_file is to write it to a file (use log_file="") to output to stdout
		/// @param[in] log_level 0 all message, 6 no message. 2 is info, 1 is debug
		/// @param[in] file_log_level 0 all message, 6 no message. 2 is info, 1 is debug
		/// @param[in] is_quit quiets the log
		void init_logger(
			const std::string &log_file,
			const spdlog::level::level_enum log_level,
			const spdlog::level::level_enum file_log_level,
			const bool is_quiet);

		/// initializing the logger writes to an output stream
		/// @param[in] os output stream
		/// @param[in] log_level 0 all message, 6 no message. 2 is info, 1 is debug
		void init_logger(std::ostream &os, const spdlog::level::level_enum log_level);

		/// change log level
		/// @param[in] log_level 0 all message, 6 no message. 2 is info, 1 is debug
		void set_log_level(const spdlog::level::level_enum log_level);

		/// gets the output log as json
		/// this is *not* what gets printed but more informative
		/// information, e.g., it contains runtimes, errors, etc.
		std::string get_log(const Eigen::MatrixXd &sol)
		{
			std::stringstream ss;
			save_json(sol, ss);
			return ss.str();
		}

	private:
		/// initializing the logger meant for internal usage
		void init_logger(const std::vector<spdlog::sink_ptr> &sinks, const spdlog::level::level_enum log_level);

		/// logger sink to stdout
		spdlog::sink_ptr console_sink_ = nullptr;
		spdlog::sink_ptr file_sink_ = nullptr;

	public:
		Units units;

		std::shared_ptr<VariationalForm> variational_form;

	public:
		/// solves the problems
		/// @param[out] sol solution
		void solve_problem(Eigen::MatrixXd &sol);
		/// solves the problem, call other methods
		/// @param[out] sol solution
		void solve(Eigen::MatrixXd &sol);

		/// solves transient problem
		/// @param[in] time_steps number of time steps
		/// @param[in] t0 initial times
		/// @param[in] dt timestep size
		/// @param[out] sol solution
		void solve_transient(const int time_steps, const double t0, const double dt, Eigen::MatrixXd &sol);

		/// solves a linear problem
		/// @param[out] sol solution
		void solve_static(Eigen::MatrixXd &sol);

		/// @brief Load or compute the initial solution.
		/// @param[out] solution Output solution variable.
		void initial_solution(Eigen::MatrixXd &solution) const;
		/// @brief Load or compute the initial velocity.
		/// @param[out] solution Output velocity variable.
		void initial_velocity(Eigen::MatrixXd &velocity) const;
		/// @brief Load or compute the initial acceleration.
		/// @param[out] solution Output acceleration variable.
		void initial_acceleration(Eigen::MatrixXd &acceleration) const;

		/// factory to create the nl solver depending on input
		/// @return nonlinear solver (eg newton or LBFGS)
		std::shared_ptr<polysolve::nonlinear::Solver> make_nl_solver(bool for_al) const;

		//---------------------------------------------------
		//-----------------Geometry--------------------------
		//---------------------------------------------------
	public:
		/// current mesh, it can be a Mesh2D or Mesh3D
		std::unique_ptr<mesh::Mesh> mesh;
		/// Obstacles used in collisions
		mesh::Obstacle obstacle;

		/// loads the mesh from the json arguments
		/// @param[in] non_conforming creates a conforming/non conforming mesh
		/// @param[in] names keys in the hdf5
		/// @param[in] cells list of cells from hdf5
		/// @param[in] vertices list of vertices from hdf5
		void load_mesh(bool non_conforming = false,
					   const std::vector<std::string> &names = std::vector<std::string>(),
					   const std::vector<Eigen::MatrixXi> &cells = std::vector<Eigen::MatrixXi>(),
					   const std::vector<Eigen::MatrixXd> &vertices = std::vector<Eigen::MatrixXd>());

		/// loads the mesh from a geogram mesh
		/// @param[in] meshin geo mesh
		/// @param[in] boundary_marker the input of the lambda is the face barycenter, the output is the sideset id
		/// @param[in] non_conforming creates a conforming/non conforming mesh
		/// @param[in] skip_boundary_sideset skip_boundary_sideset = false it uses the lambda boundary_marker to assign the sideset
		void load_mesh(GEO::Mesh &meshin, const std::function<int(const size_t, const std::vector<int> &, const RowVectorNd &, bool)> &boundary_marker, bool non_conforming = false, bool skip_boundary_sideset = false);

		/// loads the mesh from V and F,
		/// @param[in] V is #vertices x dim
		/// @param[in] F is #elements x size (size = 3 for triangle mesh, size=4 for a quad mesh if dim is 2)
		/// @param[in] non_conforming creates a conforming/non conforming mesh
		void load_mesh(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, bool non_conforming = false)
		{
			mesh = mesh::Mesh::create(V, F, non_conforming);
			load_mesh(non_conforming);
		}

		/// Resets the mesh
		void reset_mesh();

		/// Build the mesh matrices (vertices and elements) from the mesh using the bases node ordering
		void build_mesh_matrices(Eigen::MatrixXd &V, Eigen::MatrixXi &F);

		/// @brief Remesh the FE space and update solution(s).
		/// @param time Current time.
		/// @param dt Time step size.
		/// @param sol Current solution.
		/// @return True if remeshing performed any changes to the mesh/solution.
		bool remesh(const double time, const double dt, Eigen::MatrixXd &sol);

		//---------------------------------------------------
		//-----------------IPC-------------------------------
		//---------------------------------------------------

		/// @brief IPC collision mesh
		ipc::CollisionMesh collision_mesh;

		/// @brief IPC collision mesh under periodic BC
		ipc::CollisionMesh periodic_collision_mesh;
		/// index mapping from periodic 2x2 collision mesh to FE periodic mesh
		Eigen::VectorXi periodic_collision_mesh_to_basis;

		/// @brief extracts the boundary mesh for collision, called in build_basis
		static void build_collision_mesh(
			const mesh::Mesh &mesh,
			const int n_bases,
			const std::vector<basis::ElementBases> &bases,
			const std::vector<basis::ElementBases> &geom_bases,
			const std::vector<mesh::LocalBoundary> &total_local_boundary,
			const mesh::Obstacle &obstacle,
			const json &args,
			const std::function<std::string(const std::string &)> &resolve_input_path,
			const Eigen::VectorXi &in_node_to_node,
			ipc::CollisionMesh &collision_mesh);

		/// @brief extracts the boundary mesh for collision, called in build_basis
		void build_collision_mesh();
		void build_periodic_collision_mesh();

		/// checks if vertex is obstacle
		/// @param[in] vi vertex index
		/// @return if vertex is obstalce
		bool is_obstacle_vertex(const size_t vi) const
		{
			// The obstalce vertices are at the bottom of the collision mesh vertices
			return vi >= collision_mesh.full_num_vertices() - obstacle.n_vertices();
		}

		/// @brief does the simulation have contact
		///
		/// @return true/false
		bool is_contact_enabled() const
		{
			return args["contact"]["enabled"];
		}

		/// @brief does the simulation have adhesion
		///
		/// @return true/false
		bool is_adhesion_enabled() const
		{
			return args["contact"]["adhesion"]["adhesion_enabled"];
		}

		/// @brief does the simulation has pressure
		///
		/// @return true/false
		bool is_pressure_enabled() const
		{
			return (args["boundary_conditions"]["pressure_boundary"].size() > 0)
				   || (args["boundary_conditions"]["pressure_cavity"].size() > 0);
		}

		/// stores if input json contains dhat
		bool has_dhat = false;

		//---------------------------------------------------
		//-----------------OUTPUT----------------------------
		//---------------------------------------------------
	public:
		/// Directory for output files
		std::string output_dir;
		/// visualization stuff
		io::OutGeometryData out_geom;
		/// runtime statistics
		io::OutRuntimeData timings;
		/// Other statistics
		io::OutStatsData stats;
		double starting_min_edge_length = -1;
		double starting_max_edge_length = -1;
		double min_boundary_edge_length = -1;

		std::function<void(int, int, double, double)> time_callback = nullptr;

		/// saves all data on the disk according to the input params
		/// @param[in] sol solution
		void export_data(const Eigen::MatrixXd &sol);

		/// saves a timestep
		/// @param[in] time time in secs
		/// @param[in] t time index
		/// @param[in] t0 initial time
		/// @param[in] dt delta t
		/// @param[in] sol solution
		void save_timestep(const double time, const int t, const double t0, const double dt, const Eigen::MatrixXd &sol);

		/// saves a subsolve when save_solve_sequence_debug is true
		/// @param[in] i sub solve index
		/// @param[in] t time index
		/// @param[in] sol solution
		void save_subsolve(const int i, const int t, const Eigen::MatrixXd &sol);

		/// saves the output statistic to a stream
		/// @param[in] sol solution
		/// @param[out] out stream to write output
		void save_json(const Eigen::MatrixXd &sol, std::ostream &out);

		/// saves the output statistic to disc according to params
		/// @param[in] sol solution
		void save_json(const Eigen::MatrixXd &sol);

		/// @brief computes all errors
		void compute_errors(const Eigen::MatrixXd &sol);

		/// @brief Save a JSON sim file for restarting the simulation at time t
		/// @param t current time to restart at
		void save_restart_json(const double t0, const double dt, const int t) const;

		//-----------PATH management
		/// Get the root path for the state (e.g., args["root_path"] or ".")
		/// @return root path
		std::string root_path() const;

		/// Resolve input path relative to root_path() if the path is not absolute.
		/// @param[in] path path to resolve
		/// @param[in] only_if_exists resolve only if relative path exists
		/// @return path
		std::string resolve_input_path(const std::string &path, const bool only_if_exists = false) const;

		/// Resolve output path relative to output_dir if the path is not absolute
		/// @param[in] path path to resolve
		/// @return resolvedpath
		std::string resolve_output_path(const std::string &path) const;

		//---------------------------------------------------
		//-----------------differentiable--------------------
		//---------------------------------------------------
	public:
		solver::CacheLevel optimization_enabled = solver::CacheLevel::None;
		void cache_transient_adjoint_quantities(const int current_step, const Eigen::MatrixXd &sol, const Eigen::MatrixXd &disp_grad);
		solver::DiffCache diff_cached;

		std::unique_ptr<polysolve::linear::Solver> lin_solver_cached; // matrix factorization of last linear solve

		// Aux functions for setting up adjoint equations
		void compute_force_jacobian(const Eigen::MatrixXd &sol, const Eigen::MatrixXd &disp_grad, StiffnessMatrix &hessian);
		void compute_force_jacobian_prev(const int force_step, const int sol_step, StiffnessMatrix &hessian_prev) const;
		// Solves the adjoint PDE for derivatives and caches
		void solve_adjoint_cached(const Eigen::MatrixXd &rhs);
		Eigen::MatrixXd solve_adjoint(const Eigen::MatrixXd &rhs) const;
		// Returns cached adjoint solve
		Eigen::MatrixXd get_adjoint_mat(int type) const;
		Eigen::MatrixXd solve_static_adjoint(const Eigen::MatrixXd &adjoint_rhs) const;
		Eigen::MatrixXd solve_transient_adjoint(const Eigen::MatrixXd &adjoint_rhs) const;
		// Change geometric node positions
		void set_mesh_vertex(int v_id, const Eigen::VectorXd &vertex);
		void get_vertices(Eigen::MatrixXd &vertices) const;
		void get_elements(Eigen::MatrixXi &elements) const;

		// Get geometric node indices for surface/volume
		void compute_surface_node_ids(const int surface_selection, std::vector<int> &node_ids) const;
		void compute_total_surface_node_ids(std::vector<int> &node_ids) const;
		void compute_volume_node_ids(const int volume_selection, std::vector<int> &node_ids) const;

		// to replace the initial condition in json during initial condition optimization
		Eigen::MatrixXd initial_sol_update, initial_vel_update;
		// mapping from positions of FE basis nodes to positions of geometry nodes
		StiffnessMatrix basis_nodes_to_gbasis_nodes;

		//---------------------------------------------------
		//-----------------homogenization--------------------
		//---------------------------------------------------
	public:
		assembler::MacroStrainValue macro_strain_constraint;

		/// In Elasticity PDE, solve for "min W(disp_grad + \grad u)" instead of "min W(\grad u)"
		void solve_homogenization_step(Eigen::MatrixXd &sol, const int t = 0, bool adaptive_initial_weight = false); // sol is the extended solution, i.e. [periodic fluctuation, macro strain]
		void init_homogenization_solve(const double t);
		void solve_homogenization(const int time_steps, const double t0, const double dt, Eigen::MatrixXd &sol);
		bool is_homogenization() const
		{
			return args["boundary_conditions"]["periodic_boundary"]["linear_displacement_offset"].size() > 0;
		}
	};

} // namespace polyfem
