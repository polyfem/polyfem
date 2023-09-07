#pragma once

#include <polyfem/Common.hpp>

#include <polyfem/Units.hpp>

#include <polyfem/basis/ElementBases.hpp>
#include <polyfem/basis/InterfaceData.hpp>

#include <polyfem/assembler/ElementAssemblyValues.hpp>
#include <polyfem/assembler/AssemblyValsCache.hpp>
#include <polyfem/assembler/RhsAssembler.hpp>
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

#include <polyfem/io/OutData.hpp>

#include <polysolve/LinearSolver.hpp>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#ifdef POLYFEM_WITH_TBB
#include <tbb/global_control.h>
#endif

#include <memory>
#include <string>
#include <unordered_map>

#include <ipc/collision_mesh.hpp>
#include <ipc/utils/logger.hpp>

// Forward declaration
namespace cppoptlib
{
	template <typename ProblemType>
	class NonlinearSolver;
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
		void set_max_threads(const unsigned int max_threads = std::numeric_limits<unsigned int>::max());

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
		/// @param[in] is_quit quiets the log
		void init_logger(const std::string &log_file, const spdlog::level::level_enum log_level, const bool is_quiet);

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

	public:
		//---------------------------------------------------
		//-----------------assembly--------------------------
		//---------------------------------------------------

		Units units;

		/// assemblers
		std::shared_ptr<assembler::Assembler> assembler = nullptr;
		std::shared_ptr<assembler::Mass> mass_matrix_assembler = nullptr;

		std::shared_ptr<assembler::MixedAssembler> mixed_assembler = nullptr;
		std::shared_ptr<assembler::Assembler> pressure_assembler = nullptr;

		std::shared_ptr<assembler::ViscousDamping> damping_assembler = nullptr;
		std::shared_ptr<assembler::ViscousDampingPrev> damping_prev_assembler = nullptr;

		/// current problem, it contains rhs and bc
		std::shared_ptr<assembler::Problem> problem;

		/// FE bases, the size is #elements
		std::vector<basis::ElementBases> bases;
		/// FE pressure bases for mixed elements, the size is #elements
		std::vector<basis::ElementBases> pressure_bases;
		/// Geometric mapping bases, if the elements are isoparametric, this list is empty
		std::vector<basis::ElementBases> geom_bases_;

		/// number of bases
		int n_bases;
		/// number of pressure bases
		int n_pressure_bases;
		/// number of geometric bases
		int n_geom_bases;

		/// polygons, used since poly have no geom mapping
		std::map<int, Eigen::MatrixXd> polys;
		/// polyhedra, used since poly have no geom mapping
		std::map<int, std::pair<Eigen::MatrixXd, Eigen::MatrixXi>> polys_3d;

		/// vector of discretization orders, used when not all elements have the same degree, one per element
		Eigen::VectorXi disc_orders;

		/// Mapping from input nodes to FE nodes
		std::shared_ptr<polyfem::mesh::MeshNodes> mesh_nodes, geom_mesh_nodes, pressure_mesh_nodes;

		/// used to store assembly values for small problems
		assembler::AssemblyValsCache ass_vals_cache;
		assembler::AssemblyValsCache mass_ass_vals_cache;
		/// used to store assembly values for pressure for small problems
		assembler::AssemblyValsCache pressure_ass_vals_cache;

		/// Mass matrix, it is computed only for time dependent problems
		StiffnessMatrix mass;
		/// average system mass, used for contact with IPC
		double avg_mass;

		/// System right-hand side.
		Eigen::MatrixXd rhs;

		/// use average pressure for stokes problem to fix the additional dofs, true by default
		/// if false, it will fix one pressure node to zero
		bool use_avg_pressure;

		/// return the formulation (checks if the problem is scalar or not and deals with multiphysics)
		/// @return formulation
		std::string formulation() const;

		/// check if using iso parametric bases
		/// @return if basis are isoparametric
		bool iso_parametric() const;

		/// @brief Get a constant reference to the geometry mapping bases.
		/// @return A constant reference to the geometry mapping bases.
		const std::vector<basis::ElementBases> &geom_bases() const
		{
			return iso_parametric() ? bases : geom_bases_;
		}

		/// builds the bases step 2 of solve
		void build_basis();
		/// compute rhs, step 3 of solve
		void assemble_rhs();
		/// assemble mass, step 4 of solve
		void assemble_mass_mat();

		/// build a RhsAssembler for the problem
		std::shared_ptr<assembler::RhsAssembler> build_rhs_assembler(
			const int n_bases,
			const std::vector<basis::ElementBases> &bases,
			const assembler::AssemblyValsCache &ass_vals_cache) const;
		/// build a RhsAssembler for the problem
		std::shared_ptr<assembler::RhsAssembler> build_rhs_assembler() const
		{
			return build_rhs_assembler(n_bases, bases, mass_ass_vals_cache);
		}

		/// quadrature used for projecting boundary conditions
		/// @return the quadrature used for projecting boundary conditions
		int n_boundary_samples() const
		{
			using assembler::AssemblerUtils;
			const int n_b_samples_j = args["space"]["advanced"]["n_boundary_samples"];
			const int gdiscr_order = mesh->orders().size() <= 0 ? 1 : mesh->orders().maxCoeff();
			const int discr_order = std::max(disc_orders.maxCoeff(), gdiscr_order);

			const int n_b_samples = std::max(n_b_samples_j, AssemblerUtils::quadrature_order("Mass", discr_order, AssemblerUtils::BasisType::POLY, mesh->dimension()));

			return n_b_samples;
		}

	private:
		/// splits the solution in solution and pressure for mixed problems
		/// @param[in/out] sol solution
		/// @param[out] pressure pressure
		void sol_to_pressure(Eigen::MatrixXd &sol, Eigen::MatrixXd &pressure);
		/// builds bases for polygons, called inside build_basis
		void build_polygonal_basis();

	public:
		/// set the material and the problem dimension
		/// @param[in/out] list of assembler to set
		void set_materials(std::vector<std::shared_ptr<assembler::Assembler>> &assemblers) const;
		/// utility to set the material and the problem dimension to only 1 assembler
		/// @param[in/out] assembler to set
		void set_materials(assembler::Assembler &assembler) const;

		//---------------------------------------------------
		//-----------------solver----------------------------
		//---------------------------------------------------

	public:
		/// solves the problems
		/// @param[out] sol solution
		/// @param[out] pressure pressure
		void solve_problem(Eigen::MatrixXd &sol, Eigen::MatrixXd &pressure);
		/// solves the problem, call other methods
		/// @param[out] sol solution
		/// @param[out] pressure pressure
		void solve(Eigen::MatrixXd &sol, Eigen::MatrixXd &pressure)
		{
			if (!mesh)
			{
				logger().error("Load the mesh first!");
				return;
			}
			stats.compute_mesh_stats(*mesh);

			build_basis();

			assemble_rhs();
			assemble_mass_mat();

			solve_export_to_file = false;
			solution_frames.clear();
			solve_problem(sol, pressure);
			solve_export_to_file = true;
		}

		/// timedependent stuff cached
		solver::SolveData solve_data;
		/// initialize solver
		/// @param[out] sol solution
		/// @param[out] pressure pressure
		void init_solve(Eigen::MatrixXd &sol, Eigen::MatrixXd &pressure);
		/// solves transient navier stokes with operator splitting
		/// @param[in] time_steps number of time steps
		/// @param[in] dt timestep size
		/// @param[out] sol solution
		/// @param[out] pressure pressure
		void solve_transient_navier_stokes_split(const int time_steps, const double dt, Eigen::MatrixXd &sol, Eigen::MatrixXd &pressure);
		/// solves transient navier stokes with FEM
		/// @param[in] time_steps number of time steps
		/// @param[in] t0 initial times
		/// @param[in] dt timestep size
		/// @param[out] sol solution
		/// @param[out] pressure pressure
		void solve_transient_navier_stokes(const int time_steps, const double t0, const double dt, Eigen::MatrixXd &sol, Eigen::MatrixXd &pressure);
		/// solves transient linear problem
		/// @param[in] time_steps number of time steps
		/// @param[in] t0 initial times
		/// @param[in] dt timestep size
		/// @param[out] sol solution
		/// @param[out] pressure pressure
		void solve_transient_linear(const int time_steps, const double t0, const double dt, Eigen::MatrixXd &sol, Eigen::MatrixXd &pressure);
		/// solves transient tensor nonlinear problem
		/// @param[in] time_steps number of time steps
		/// @param[in] t0 initial times
		/// @param[in] dt timestep size
		/// @param[out] sol solution
		void solve_transient_tensor_nonlinear(const int time_steps, const double t0, const double dt, Eigen::MatrixXd &sol);
		/// initialize the nonlinear solver
		/// @param[out] sol solution
		/// @param[in] t (optional) initial time
		void init_nonlinear_tensor_solve(Eigen::MatrixXd &sol, const double t = 1.0, const bool init_time_integrator = true);
		/// initialize the linear solve
		/// @param[in] t (optional) initial time
		void init_linear_solve(Eigen::MatrixXd &sol, const double t = 1.0);
		/// @brief Load or compute the initial solution.
		/// @param[out] solution Output solution variable.
		void initial_solution(Eigen::MatrixXd &solution) const;
		/// @brief Load or compute the initial velocity.
		/// @param[out] solution Output velocity variable.
		void initial_velocity(Eigen::MatrixXd &velocity) const;
		/// @brief Load or compute the initial acceleration.
		/// @param[out] solution Output acceleration variable.
		void initial_acceleration(Eigen::MatrixXd &acceleration) const;
		/// solves a linear problem
		/// @param[out] sol solution
		/// @param[out] pressure pressure
		void solve_linear(Eigen::MatrixXd &sol, Eigen::MatrixXd &pressure);
		/// solves a navier stokes
		/// @param[out] sol solution
		/// @param[out] pressure pressure
		void solve_navier_stokes(Eigen::MatrixXd &sol, Eigen::MatrixXd &pressure);
		/// solves nonlinear problems
		/// @param[out] sol solution
		/// @param[in] t (optional) time step id
		void solve_tensor_nonlinear(Eigen::MatrixXd &sol, const int t = 0, const bool init_lagging = true);

		/// factory to create the nl solver depending on input
		/// @return nonlinear solver (eg newton or LBFGS)
		template <typename ProblemType>
		std::shared_ptr<cppoptlib::NonlinearSolver<ProblemType>> make_nl_solver(
			const std::string &linear_solver_type = "") const;

		/// @brief Solve the linear problem with the given solver and system.
		/// @param solver Linear solver.
		/// @param A Linear system matrix.
		/// @param b Right-hand side.
		/// @param compute_spectrum If true, compute the spectrum.
		/// @param[out] sol solution
		/// @param[out] pressure pressure
		void solve_linear(
			const std::unique_ptr<polysolve::LinearSolver> &solver,
			StiffnessMatrix &A,
			Eigen::VectorXd &b,
			const bool compute_spectrum,
			Eigen::MatrixXd &sol, Eigen::MatrixXd &pressure);

	public:
		/// @brief utility that builds the stiffness matrix and collects stats, used only for linear problems
		/// @param[out] stiffness matrix
		void build_stiffness_mat(StiffnessMatrix &stiffness);

		//---------------------------------------------------
		//-----------------nodes flags-----------------------
		//---------------------------------------------------

	public:
		/// list of boundary nodes
		std::vector<int> boundary_nodes;
		/// list of neumann boundary nodes
		std::vector<int> pressure_boundary_nodes;
		/// mapping from elements to nodes for all mesh
		std::vector<mesh::LocalBoundary> total_local_boundary;
		/// mapping from elements to nodes for dirichlet boundary conditions
		std::vector<mesh::LocalBoundary> local_boundary;
		/// mapping from elements to nodes for neumann boundary conditions
		std::vector<mesh::LocalBoundary> local_neumann_boundary;
		/// nodes on the boundary of polygonal elements, used for harmonic bases
		std::map<int, basis::InterfaceData> poly_edge_to_data;
		/// per node dirichlet
		std::vector<int> dirichlet_nodes;
		std::vector<RowVectorNd> dirichlet_nodes_position;
		/// per node neumann
		std::vector<int> neumann_nodes;
		std::vector<RowVectorNd> neumann_nodes_position;

		/// Inpute nodes (including high-order) to polyfem nodes, only for isoparametric
		Eigen::VectorXi in_node_to_node;
		/// maps in vertices/edges/faces/cells to polyfem vertices/edges/faces/cells
		Eigen::VectorXi in_primitive_to_primitive;

		std::vector<int> primitive_to_node() const;
		std::vector<int> node_to_primitive() const;

	private:
		/// build the mapping from input nodes to polyfem nodes
		void build_node_mapping();

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
		void load_mesh(GEO::Mesh &meshin, const std::function<int(const RowVectorNd &)> &boundary_marker, bool non_conforming = false, bool skip_boundary_sideset = false);

		/// loads the mesh from V and F,
		/// @param[in] V is #vertices x dim
		/// @param[in] F is #elements x size (size = 3 for triangle mesh, size=4 for a quad mesh if dim is 2)
		/// @param[in] non_conforming creates a conforming/non conforming mesh
		void load_mesh(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, bool non_conforming = false)
		{
			mesh = mesh::Mesh::create(V, F, non_conforming);
			load_mesh(non_conforming);
		}

		/// set the boundary sideset from a lambda that takes the face/edge barycenter
		/// @param[in] boundary_marker function from face/edge barycenter that returns the sideset id
		void set_boundary_side_set(const std::function<int(const RowVectorNd &)> &boundary_marker)
		{
			mesh->compute_boundary_ids(boundary_marker);
		}

		/// set the boundary sideset from a lambda that takes the face/edge barycenter and a flag if the face/edge is boundary or not (used to set internal boundaries)
		/// @param[in] boundary_marker function from face/edge barycenter and a flag if the face/edge is boundary that returns the sideset id
		void set_boundary_side_set(const std::function<int(const RowVectorNd &, bool)> &boundary_marker)
		{
			mesh->compute_boundary_ids(boundary_marker);
		}

		/// set the boundary sideset from a lambda that takes the face/edge vertices and a flag if the face/edge is boundary or not (used to set internal boundaries)
		/// @param[in] boundary_marker function from face/edge vertices and a flag if the face/edge is boundary that returns the sideset id
		void set_boundary_side_set(const std::function<int(const std::vector<int> &, bool)> &boundary_marker)
		{
			mesh->compute_boundary_ids(boundary_marker);
		}

		/// Resets the mesh
		void reset_mesh();

		/// Build the mesh matrices (vertices and elements) from the mesh using the bases node ordering
		void build_mesh_matrices(Eigen::MatrixXd &V, Eigen::MatrixXi &F);

		//---------------------------------------------------
		//-----------------IPC-------------------------------
		//---------------------------------------------------

		/// @brief IPC collision mesh
		ipc::CollisionMesh collision_mesh;

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

		/// checks if vertex is obstacle
		/// @param[in] vi vertex index
		/// @return if vertex is obstalce
		bool is_obstacle_vertex(const size_t vi) const
		{
			// The obstalce vertices are at the bottom of the collision mesh vertices
			return vi >= collision_mesh.full_num_vertices() - obstacle.n_vertices();
		}

		/// @brief does the simulation has contact
		///
		/// @return true/false
		bool is_contact_enabled() const { return args["contact"]["enabled"]; }

		/// stores if input json contains dhat
		bool has_dhat = false;

		//---------------------------------------------------
		//-----------------OUTPUT----------------------------
		//---------------------------------------------------
	public:
		/// Directory for output files
		std::string output_dir;

		/// flag to decide if exporting the time dependent solution to files
		/// or save it in the solution_frames array
		bool solve_export_to_file = true;
		/// saves the frames in a vector instead of VTU
		std::vector<io::SolutionFrame> solution_frames;
		/// visualization stuff
		io::OutGeometryData out_geom;
		/// runtime statistics
		io::OutRuntimeData timings;
		/// Other statistics
		io::OutStatsData stats;

		/// saves all data on the disk according to the input params
		/// @param[in] sol solution
		/// @param[in] pressure pressure
		void export_data(const Eigen::MatrixXd &sol, const Eigen::MatrixXd &pressure);

		/// saves a timestep
		/// @param[in] time time in secs
		/// @param[in] t time index
		/// @param[in] t0 initial time
		/// @param[in] dt delta t
		/// @param[in] sol solution
		/// @param[in] pressure pressure
		void save_timestep(const double time, const int t, const double t0, const double dt, const Eigen::MatrixXd &sol, const Eigen::MatrixXd &pressure);

		/// saves a subsolve when save_solve_sequence_debug is true
		/// @param[in] i sub solve index
		/// @param[in] t time index
		/// @param[in] sol solution
		/// @param[in] pressure pressure
		void save_subsolve(const int i, const int t, const Eigen::MatrixXd &sol, const Eigen::MatrixXd &pressure);

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

#ifdef POLYFEM_WITH_TBB
		/// limits the number of used threads
		std::shared_ptr<tbb::global_control> thread_limiter;
#endif

		//---------------------------------------------------
		//-----------------differentiable--------------------
		//---------------------------------------------------
	public:
		bool optimization_enabled = false;
		void cache_transient_adjoint_quantities(const int current_step, const Eigen::MatrixXd &sol, const Eigen::MatrixXd &disp_grad);
		solver::DiffCache diff_cached;

		std::unique_ptr<polysolve::LinearSolver> lin_solver_cached; // matrix factorization of last linear solve

		int ndof() const
		{
			const int actual_dim = problem->is_scalar() ? 1 : mesh->dimension();
			if (mixed_assembler == nullptr)
				return actual_dim * n_bases;
			else
				return actual_dim * n_bases + n_pressure_bases;
		}

		// Aux functions for setting up adjoint equations
		void compute_force_jacobian(const Eigen::MatrixXd &sol, const Eigen::MatrixXd &disp_grad, StiffnessMatrix &hessian);
		void compute_force_jacobian_prev(const int force_step, const int sol_step, StiffnessMatrix &hessian_prev) const;
		// Solves the adjoint PDE for derivatives and caches
		void solve_adjoint_cached(const Eigen::MatrixXd &rhs);
		Eigen::MatrixXd solve_adjoint(const Eigen::MatrixXd &rhs) const;
		// Returns cached adjoint solve
		Eigen::MatrixXd get_adjoint_mat(int type) const
		{
			assert(diff_cached.adjoint_mat().size() > 0);
			if (problem->is_time_dependent())
			{
				if (type == 0)
					return diff_cached.adjoint_mat().leftCols(diff_cached.adjoint_mat().cols() / 2);
				else if (type == 1)
					return diff_cached.adjoint_mat().middleCols(diff_cached.adjoint_mat().cols() / 2, diff_cached.adjoint_mat().cols() / 2);
				else
					log_and_throw_error("Invalid adjoint type!");
			}

			return diff_cached.adjoint_mat();
		}
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
		// mapping from positions of geometric nodes to positions of FE basis nodes
		StiffnessMatrix gbasis_nodes_to_basis_nodes;
	};

} // namespace polyfem
