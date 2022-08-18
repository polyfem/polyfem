#pragma once

#include <polyfem/basis/ElementBases.hpp>
#include <polyfem/assembler/ElementAssemblyValues.hpp>
#include <polyfem/assembler/AssemblyValsCache.hpp>
#include <polyfem/assembler/RhsAssembler.hpp>
#include <polyfem/assembler/Problem.hpp>
#include <polyfem/mesh/Mesh.hpp>
#include <polyfem/mesh/Obstacle.hpp>
#include <polyfem/mesh/MeshNodes.hpp>
#include <polyfem/utils/RefElementSampler.hpp>
#include <polyfem/mesh/LocalBoundary.hpp>
#include <polyfem/basis/InterfaceData.hpp>
#include <polyfem/assembler/AssemblerUtils.hpp>
#include <polyfem/utils/ElasticityUtils.hpp>
#include <polyfem/Common.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/IntegrableFunctional.hpp>
#include <polyfem/utils/SummableFunctional.hpp>

#include <polyfem/mesh/mesh2D/NCMesh2D.hpp>
#include <polyfem/mesh/mesh2D/CMesh2D.hpp>
#include <polyfem/mesh/mesh3D/CMesh3D.hpp>
#include <polyfem/mesh/mesh3D/NCMesh3D.hpp>
#include <polyfem/utils/StringUtils.hpp>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#ifdef POLYFEM_WITH_TBB
#include <tbb/global_control.h>
#endif

#include <memory>
#include <string>

#include <ipc/collision_mesh.hpp>
#include <ipc/ipc.hpp>
#include <ipc/broad_phase/broad_phase.hpp>
#include <ipc/friction/friction_constraint.hpp>

#include <ipc/utils/logger.hpp>

// Forward declaration
namespace cppoptlib
{
	template <typename ProblemType>
	class NonlinearSolver;
}

namespace polyfem
{
	/// class used to save the solution of time dependent problems in code instead of saving it to the disc
	class SolutionFrame
	{
	public:
		std::string name;
		Eigen::MatrixXd points;
		Eigen::MatrixXi connectivity;
		Eigen::MatrixXd solution;
		Eigen::MatrixXd pressure;
		Eigen::MatrixXd exact;
		Eigen::MatrixXd error;
		Eigen::MatrixXd scalar_value;
		Eigen::MatrixXd scalar_value_avg;
	};

	namespace solver
	{
		class NLProblem;
		class ALNLProblem;
	} // namespace solver

	/// class to store time stepping data
	class StepData
	{
	public:
		std::shared_ptr<assembler::RhsAssembler> rhs_assembler;
		std::shared_ptr<solver::NLProblem> nl_problem;
		std::shared_ptr<solver::ALNLProblem> alnl_problem;
	};

	/// main class that contains the polyfem solver and all its state
	class State
	{
	public:
		//---------------------------------------------------
		//-----------------initializtion---------------------
		//---------------------------------------------------

		~State() = default;
		/// Constructor
		/// @param[in] max_threads max number of threads
		State(const unsigned int max_threads = std::numeric_limits<unsigned int>::max());

		/// initialize the polyfem solver with a json settings
		/// @param[in] args input arguments
		/// @param[in] output_dir output directory
		void init(const json &args, const std::string &output_dir = "");

		/// initialize time settings if args contains "time"
		void init_time();

		/// main input arguments containing all defaults
		json args;

		//---------------------------------------------------
		//-----------------logger----------------------------
		//---------------------------------------------------

		/// initalizing the logger
		/// @param[in] log_file is to write it to a file (use log_file="") to output to stdout
		/// @param[in] log_level 0 all message, 6 no message. 2 is info, 1 is debug
		/// @param[in] is_quit quiets the log
		void init_logger(const std::string &log_file, const spdlog::level::level_enum log_level, const bool is_quiet);

		/// initalizing the logger writes to an output stream
		/// @param[in] os output stream
		/// @param[in] log_level 0 all message, 6 no message. 2 is info, 1 is debug
		void init_logger(std::ostream &os, const spdlog::level::level_enum log_level);

		/// change log level
		/// @param[in] log_level 0 all message, 6 no message. 2 is info, 1 is debug
		void set_log_level(const spdlog::level::level_enum log_level)
		{
			spdlog::set_level(log_level);
			logger().set_level(log_level);
			IPC_LOG(set_level(log_level));
			current_log_level = log_level;
		}

		int current_log_level;

		/// gets the output log as json
		/// this is *not* what gets printed but more informative
		/// information, eg it contains runtimes, errors, etc.
		std::string get_log()
		{
			std::stringstream ss;
			save_json(ss);
			return ss.str();
		}

	private:
		/// initalizing the logger meant for internal usage
		void init_logger(const std::vector<spdlog::sink_ptr> &sinks, const spdlog::level::level_enum log_level);

	public:
		/// Directory for output files
		std::string output_dir;
		/// density of the input, default=1.
		Density density;

		/// average system mass, used for contact with IPC
		double avg_mass;

	private:
		void build_node_mapping();

	public:
		/// Inpute nodes (including high-order) to polyfem nodes, only for isoparametric
		Eigen::VectorXi in_node_to_node;
		/// maps in vertices/edges/faces/cells to polyfem vertices/edges/faces/cells
		Eigen::VectorXi in_primitive_to_primitive;

		//---------------------------------------------------
		//-----------------assembly--------------------------
		//---------------------------------------------------

		/// assembler, it dispatches call to the differnt assembers based on the formulation
		assembler::AssemblerUtils assembler;
		// viscous dissipation assembler
		assembler::ViscousDampingAssembler damping_assembler;
		/// current problem, it contains rhs and bc
		std::shared_ptr<assembler::Problem> problem;

		/// FE bases, the size is #elements
		std::vector<ElementBases> bases;
		/// FE pressure bases for mixed elements, the size is #elements
		std::vector<ElementBases> pressure_bases;
		///Geometric mapping bases, if the elements are isoparametric, this list is empty
		std::vector<ElementBases> geom_bases;

		// Mapping from input nodes to FE nodes
		std::vector<int> primitive_to_bases_node, primitive_to_geom_bases_node, primitive_to_pressure_bases_node;

		/// polygons, used since poly have no geom mapping
		std::map<int, Eigen::MatrixXd> polys;
		/// polyhedra, used since poly have no geom mapping
		std::map<int, std::pair<Eigen::MatrixXd, Eigen::MatrixXi>> polys_3d;

		/// vector of discretization oders, used when not all elements have the same degree, one per element
		Eigen::VectorXi disc_orders;

		/// Mapping from input nodes to FE nodes
		std::shared_ptr<polyfem::mesh::MeshNodes> mesh_nodes, geom_mesh_nodes, pressure_mesh_nodes;

		//list of dirichlet boundary geometry nodes
		std::vector<int> boundary_gnodes;
		std::vector<bool> boundary_gnodes_mask;

		/// used to store assembly values for small problems
		assembler::AssemblyValsCache ass_vals_cache;
		/// used to store assembly values for pressure for small problems
		assembler::AssemblyValsCache pressure_ass_vals_cache;

		/// stiffness and mass matrix.
		/// Stiffness is not compute for non linear problems
		StiffnessMatrix stiffness;
		/// Mass matrix, it is computed only for time dependent problems
		StiffnessMatrix mass;
		/// System righ-hand side.
		Eigen::MatrixXd rhs;

		/// solution
		Eigen::MatrixXd sol;
		/// pressure solution, if the problem is not mixed, pressure is empty
		Eigen::MatrixXd pressure;

		Eigen::MatrixXd pre_sol;
		std::vector<Eigen::MatrixXd> pre_sols;

		/// use average pressure for stokes problem to fix the additional dofs, true by default
		/// if false, it will fix one pressure node to zero
		bool use_avg_pressure;

		/// number of bases
		int n_bases, n_geom_bases;
		/// number of pressure bases
		int n_pressure_bases;

		/// return the formulation (checks if the problem is scalar or not and delas wih multiphisics)
		/// @return fomulation
		std::string formulation() const;

		/// check if using iso parametric bases
		/// @return if basis are isoparametric
		bool iso_parametric() const;

		/// builds the bases step 2 of solve
		void build_basis();
		/// compute rhs, step 3 of solve
		void assemble_rhs();
		/// assemble matrices, step 4 of solve
		void assemble_stiffness_mat();

		/// quadrature used for projecting boundary conditions
		/// @return the quadrature used for projecting boundary conditions
		int n_boundary_samples() const
		{
			const int n_b_samples_j = args["space"]["advanced"]["n_boundary_samples"];
			const int discr_order = mesh->orders().size() <= 0 ? 1 : mesh->orders().maxCoeff();
			// TODO: verify me
			const int n_b_samples = std::max(n_b_samples_j, discr_order * 2 + 1);

			return n_b_samples;
		}

		/// compute a priori prefinement in 2d, fills disc_orders
		/// @param[in] mesh2d mesh
		void p_refinement(const mesh::Mesh2D &mesh2d);
		/// compute a priori prefinement in 3d, fills disc_orders
		/// @param[in] mesh2d mesh
		void p_refinement(const mesh::Mesh3D &mesh3d);

		//---------------------------------------------------
		//-----------------solver----------------------------
		//---------------------------------------------------

		/// solves the proble, step 5
		void solve_problem();
		void solve_homogenization();
		/// solves the problem, call other methods
		void solve()
		{
			compute_mesh_stats();

			build_basis();

			assemble_rhs();
			assemble_stiffness_mat();

			solve_export_to_file = false;
			solution_frames.clear();
			solve_problem();
			solve_export_to_file = true;
		}

		void compute_homogenized_tensor(Eigen::MatrixXd &C);

		/// timedependent stuff cached
		StepData step_data;
		/// initialize transient solver
		/// @param[in] c_sol current solution
		void init_transient(Eigen::VectorXd &c_sol);
		/// solves transient navier stokes with operator splitting
		/// @param[in] time_steps number of time steps
		/// @param[in] dt timestep size
		/// @param[in] rhs_assembler rhs assembler
		void solve_transient_navier_stokes_split(const int time_steps, const double dt, const assembler::RhsAssembler &rhs_assembler);
		/// solves transient navier stokes with FEM
		/// @param[in] time_steps number of time steps
		/// @param[in] t0 initial times
		/// @param[in] dt timestep size
		/// @param[in] rhs_assembler rhs assembler
		/// @param[out] c_sol solution
		void solve_transient_navier_stokes(const int time_steps, const double t0, const double dt, const assembler::RhsAssembler &rhs_assembler, Eigen::VectorXd &c_sol);
		/// solves transient scalar problem
		/// @param[in] time_steps number of time steps
		/// @param[in] t0 initial times
		/// @param[in] dt timestep size
		/// @param[in] rhs_assembler rhs assembler
		/// @param[out] x solution
		void solve_transient_scalar(const int time_steps, const double t0, const double dt, const assembler::RhsAssembler &rhs_assembler, Eigen::VectorXd &x);
		/// solves transient linear problem
		/// @param[in] time_steps number of time steps
		/// @param[in] t0 initial times
		/// @param[in] dt timestep size
		/// @param[in] rhs_assembler rhs assembler
		/// @param[out] x solution
		void solve_transient_tensor_linear(const int time_steps, const double t0, const double dt, const assembler::RhsAssembler &rhs_assembler);
		/// solves transient tensor non linear problem
		/// @param[in] time_steps number of time steps
		/// @param[in] t0 initial times
		/// @param[in] dt timestep size
		/// @param[in] rhs_assembler rhs assembler
		void solve_transient_tensor_non_linear(const int time_steps, const double t0, const double dt, const assembler::RhsAssembler &rhs_assembler);
		/// initialized the non linear solver
		/// @param[in] t0 initial times
		/// @param[in] dt timestep size
		/// @param[in] rhs_assembler rhs assembler
		void solve_transient_tensor_non_linear_init(const double t0, const double dt, const assembler::RhsAssembler &rhs_assembler);
		/// steps trought time
		/// @param[in] t0 initial times
		/// @param[in] dt timestep size
		/// @param[in] t time
		/// @param[out] solver_info output solver stats
		void solve_transient_tensor_non_linear_step(const double t0, const double dt, const int t, json &solver_info);
		/// solves a linear problem
		void solve_linear();
		/// solves a navier stokes
		void solve_navier_stokes();
		/// solves nonlinear problems
		void solve_non_linear();

		/// factory to create the nl solver depdending on input
		/// @return non linear solver (eg newton or LBFGS)
		template <typename ProblemType>
		std::shared_ptr<cppoptlib::NonlinearSolver<ProblemType>> make_nl_solver() const;

		//---------------------------------------------------
		//-----------------nodes flags-----------------------
		//---------------------------------------------------

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
		std::map<int, InterfaceData> poly_edge_to_data;
		/// Matrices containing the input per node dirichelt
		std::vector<Eigen::MatrixXd> input_dirichelt;

		/// stores if input json contains dhat
		bool has_dhat = false;

		//---------------------------------------------------
		//-----------------Geometry--------------------------
		//---------------------------------------------------

		/// current mesh, it can be a Mesh2D or Mesh3D
		std::unique_ptr<mesh::Mesh> mesh;
		/// Obstacles used in collisions
		mesh::Obstacle obstacle;
		/// used to sample the solution
		utils::RefElementSampler ref_element_sampler;

		/// computes the mesh size, it samples every edges n_samples times
		/// uses curved_mesh_size (false by default) to compute the size of
		/// the linear mesh
		/// @param[in] mesh to compute stats
		/// @param[in] bases geom bases
		/// @param[in] n_samples used for curved meshes
		void compute_mesh_size(const mesh::Mesh &mesh, const std::vector<ElementBases> &bases, const int n_samples);

		/// loads the mesh from the json arguments
		/// @param[in] non_conforming creates a conforming/non conforming mesh
		/// @param[in] names keys in the hdf5
		/// @param[in] cells list of cells from hdf5
		/// @param[in] vertices list of vertices from hdf5
		void load_mesh(bool non_conforming = false,
					   const std::vector<std::string> &names = std::vector<std::string>(),
					   const std::vector<Eigen::MatrixXi> &cells = std::vector<Eigen::MatrixXi>(),
					   const std::vector<Eigen::MatrixXd> &vertices = std::vector<Eigen::MatrixXd>());

		/// loads a febio file, uses args_in for default, [DEPRECATED]
		void load_febio(const std::string &path, const json &args_in);

		/// loads the mesh from a geogram mesh
		/// @param[in] meshin geo mesh
		/// @param[in] boundary_marker the input of the lambda is the face barycenter, the output is the sideset id
		/// @param[in] non_conforming creates a conforming/non conforming mesh
		/// @param[in] skip_boundary_sideset skip_boundary_sideset = false it uses the lambda boundary_marker to assign the sideset
		void load_mesh(GEO::Mesh &meshin, const std::function<int(const RowVectorNd &)> &boundary_marker, bool non_conforming = false, bool skip_boundary_sideset = false);

		/// loads the mesh from V and F,
		/// @param[in] V is #vertices x dim
		/// @param[in] F is #elements x size (size = 3 for triangle mesh, size=4 for a quaud mesh if dim is 2)
		/// @param[in] non_conforming creates a conforming/non conforming mesh
		void load_mesh(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, bool non_conforming = false)
		{
			mesh = mesh::Mesh::create(V, F, non_conforming);
			load_mesh(non_conforming);
		}

		///set the boundary sideset from a lambda that takes the face/edge barycenter
		/// @param[in] boundary_marker function from face/edge barycenter that returns the sideset id
		void set_boundary_side_set(const std::function<int(const RowVectorNd &)> &boundary_marker) { mesh->compute_boundary_ids(boundary_marker); }
		///set the boundary sideset from a lambda that takes the face/edge barycenter and a flag if the face/edge is boundary or not (used to set internal boundaries)
		/// @param[in] boundary_marker function from face/edge barycenter and a flag if the face/edge is boundary that returns the sideset id
		void set_boundary_side_set(const std::function<int(const RowVectorNd &, bool)> &boundary_marker) { mesh->compute_boundary_ids(boundary_marker); }
		///set the boundary sideset from a lambda that takes the face/edge vertices and a flag if the face/edge is boundary or not (used to set internal boundaries)
		/// @param[in] boundary_marker function from face/edge vertices and a flag if the face/edge is boundary that returns the sideset id
		void set_boundary_side_set(const std::function<int(const std::vector<int> &, bool)> &boundary_marker) { mesh->compute_boundary_ids(boundary_marker); }

		/// Resets the mesh
		void reset_mesh();

		//---------------------------------------------------
		//-----------------IPC-------------------------------
		//---------------------------------------------------

		/// boundary mesh used for collision
		/// boundary_nodes_pos contains the total number of nodes, the internal ones are zero
		/// for high-order fem the faces are triangulated
		/// this is currently supported only for tri and tet meshes
		Eigen::MatrixXd boundary_nodes_pos;
		/// edge indices into full vertices
		Eigen::MatrixXi boundary_edges;
		/// triangle indices into full vertices
		Eigen::MatrixXi boundary_triangles;
		/// ipc collision mesh into surface vertices
		ipc::CollisionMesh collision_mesh;

		Eigen::MatrixXd geom_boundary_nodes_pos;
		Eigen::MatrixXi geom_boundary_edges;
		Eigen::MatrixXi geom_boundary_triangles;
		ipc::CollisionMesh geom_collision_mesh;

		Eigen::MatrixXd boundary_nodes_pos_pressure;
		Eigen::MatrixXi boundary_edges_pressure;
		Eigen::MatrixXi boundary_triangles_pressure;

		/// extracts the boundary mesh
		/// @param[in] bases geom bases
		/// @param[out] boundary_nodes_pos nodes positions
		/// @param[out] boundary_edges edges
		/// @param[out] boundary_triangles triangles
		void extract_boundary_mesh(
			const int n_current_bases,
			const std::vector<ElementBases> &bases,
			Eigen::MatrixXd &boundary_nodes_pos,
			Eigen::MatrixXi &boundary_edges,
			Eigen::MatrixXi &boundary_triangles) const;

		/// checks if vertex is obstacle
		/// @param[in] vi vertex index
		/// @return if vertex is obstalce
		bool is_obstacle_vertex(const size_t vi) const
		{
			return vi >= boundary_nodes_pos.rows() - obstacle.n_vertices();
		}

		// Differentiation functional, only used for transient problems
		void cache_transient_adjoint_quantities(const int current_step);
		struct DiffCachedParts
		{
			StiffnessMatrix gradu_h;
			StiffnessMatrix gradu_h_next;
			Eigen::MatrixXd u;
			double kappa;
			ipc::Constraints contact_set;
			ipc::FrictionConstraints friction_constraint_set;
		};
		std::vector<DiffCachedParts> diff_cached;
		Eigen::MatrixXd initial_velocity_cache;

		// one_form, for export use
		Eigen::VectorXd descent_direction;
		// Aux functions for setting up adjoint equations
		void compute_force_hessian_nonlinear(std::shared_ptr<solver::NLProblem> nl, StiffnessMatrix &hessian, StiffnessMatrix &hessian_prev, const int bdf_order = 1);
		void compute_force_hessian(StiffnessMatrix &hessian, StiffnessMatrix &hessian_prev, const int bdf_order = 1);
		void compute_adjoint_rhs(const std::function<Eigen::MatrixXd(const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, json &params)> &grad_j, const Eigen::MatrixXd &solution, Eigen::VectorXd &b, bool only_surface = false);
		void compute_adjoint_rhs(const SummableFunctional &j, const Eigen::MatrixXd &solution, Eigen::VectorXd &b);
		// Sets up the adjoint problem for static PDE
		void setup_adjoint(const std::function<Eigen::MatrixXd(const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, json &params)> &grad_j, StiffnessMatrix &A, Eigen::VectorXd &b, bool only_surface = false);
		// Solves the adjoint PDE for derivatives
		void solve_adjoint(const IntegrableFunctional &j, Eigen::MatrixXd &adjoint_solution);
		void solve_adjoint(const SummableFunctional &j, Eigen::MatrixXd &adjoint_solution);
		void solve_transient_adjoint(const IntegrableFunctional &j, std::vector<Eigen::MatrixXd> &adjoint_nu, std::vector<Eigen::MatrixXd> &adjoint_p);
		void solve_zero_dirichlet(StiffnessMatrix &A, Eigen::VectorXd &b, const std::vector<int> &indices, Eigen::MatrixXd &adjoint_solution);
		// Discretizes a vector field over the current mesh
		void sample_field(std::function<Eigen::MatrixXd(const Eigen::MatrixXd &)> field, Eigen::MatrixXd &discrete_field, const int order = 1);
		// Change geometric node positions
		void set_v(const Eigen::MatrixXd &vertices);
		void get_vf(Eigen::MatrixXd &vertices, Eigen::MatrixXi &faces, const bool geometric = true);
		// Computes the integral of a given functional J = \int j dx
		double J(const IntegrableFunctional &j)
		{
			// assert((problem->is_time_dependent() && diff_cached.size() > 0) || (!problem->is_time_dependent() && sol.size() > 0));
			return problem->is_time_dependent() ? J_transient(j) : J_static(j);
		}
		double J_static(const IntegrableFunctional &j);
		double J_static(const SummableFunctional &j);
		double J_transient(const IntegrableFunctional &j);
		double J_transient_step(const IntegrableFunctional &j, const int step);
		// Aux functions for computing derivatives of different forces wrt. different parameters
		void compute_shape_derivative_functional_term(const Eigen::MatrixXd &solution, const IntegrableFunctional &j, Eigen::VectorXd &term, const int cur_time_step = 0);
		void compute_shape_derivative_elasticity_term(const Eigen::MatrixXd &solution, const Eigen::MatrixXd &adjoint_sol, Eigen::VectorXd &term);
		void compute_shape_derivative_damping_term(const Eigen::MatrixXd &solution, const Eigen::MatrixXd &prev_solution, const Eigen::MatrixXd &adjoint_sol, Eigen::VectorXd &term);
		void compute_topology_derivative_functional_term(const Eigen::MatrixXd &solution, const IntegrableFunctional &j, Eigen::VectorXd &term);
		void compute_material_derivative_elasticity_term(const Eigen::MatrixXd &solution, const Eigen::MatrixXd &adjoint_sol, Eigen::VectorXd &term);
		void compute_topology_derivative_elasticity_term(const Eigen::MatrixXd &solution, const Eigen::MatrixXd &adjoint_sol, Eigen::VectorXd &term);
		void compute_damping_derivative_damping_term(const Eigen::MatrixXd &solution, const Eigen::MatrixXd &prev_solution, const Eigen::MatrixXd &adjoint_sol, Eigen::VectorXd &term);
		void compute_derivative_contact_term(const ipc::Constraints &contact_set, const Eigen::MatrixXd &solution, const Eigen::MatrixXd &adjoint_sol, Eigen::VectorXd &term);
		void compute_derivative_friction_term(const Eigen::MatrixXd &prev_solution, const Eigen::MatrixXd &solution, const Eigen::MatrixXd &adjoint_sol, const ipc::FrictionConstraints &friction_constraints_set, Eigen::VectorXd &term);
		void compute_mass_derivative_term(const Eigen::MatrixXd &adjoint_sol, const Eigen::MatrixXd &velocity, Eigen::VectorXd &term);
		// Derivatives wrt. an input functional J = \int j dx
		void dJ_shape_static(const IntegrableFunctional &j, Eigen::VectorXd &one_form);
		void dJ_material_static(const IntegrableFunctional &j, Eigen::VectorXd &one_form);
		void dJ_material_static(const SummableFunctional &j, Eigen::VectorXd &one_form);
		void dJ_topology_static(const IntegrableFunctional &j, Eigen::VectorXd &one_form);
		// For transient problems, Derivatives wrt. an input functional J = sum_i J_i, where J_i = \int j dx at time step i
		void dJ_full_material_transient(const IntegrableFunctional &j, Eigen::VectorXd &one_form); // including material, friction, damping
		void dJ_material_transient(const IntegrableFunctional &j, Eigen::VectorXd &one_form);
		void dJ_friction_transient(const IntegrableFunctional &j, double &one_form);
		void dJ_damping_transient(const IntegrableFunctional &j, Eigen::VectorXd &one_form);
		void dJ_shape_transient(const IntegrableFunctional &j, Eigen::VectorXd &one_form);
		void dJ_initial_condition(const IntegrableFunctional &j, Eigen::VectorXd &one_form);
		// More generally, J_i is some function of a vector of \int j dx at time step i
		double J_transient(const std::vector<IntegrableFunctional> &js, const std::function<double(const Eigen::VectorXd &, const json &)> &Ji);
		void solve_transient_adjoint(const std::vector<IntegrableFunctional> &js, const std::function<Eigen::VectorXd(const Eigen::VectorXd &, const json &)> &dJi_dintegrals, std::vector<Eigen::MatrixXd> &adjoint_nu, std::vector<Eigen::MatrixXd> &adjoint_p);
		void dJ_full_material_transient(const std::vector<IntegrableFunctional> &js, const std::function<Eigen::VectorXd(const Eigen::VectorXd &, const json &)> &dJi_dintegrals, Eigen::VectorXd &one_form); // including material, friction, damping
		void dJ_material_transient(const std::vector<IntegrableFunctional> &js, const std::function<Eigen::VectorXd(const Eigen::VectorXd &, const json &)> &dJi_dintegrals, Eigen::VectorXd &one_form);
		void dJ_friction_transient(const std::vector<IntegrableFunctional> &js, const std::function<Eigen::VectorXd(const Eigen::VectorXd &, const json &)> &dJi_dintegrals, double &one_form);
		void dJ_damping_transient(const std::vector<IntegrableFunctional> &js, const std::function<Eigen::VectorXd(const Eigen::VectorXd &, const json &)> &dJi_dintegrals, Eigen::VectorXd &one_form);
		void dJ_initial_condition(const std::vector<IntegrableFunctional> &js, const std::function<Eigen::VectorXd(const Eigen::VectorXd &, const json &)> &dJi_dintegrals, Eigen::VectorXd &one_form);
		void dJ_shape_transient(const std::vector<IntegrableFunctional> &js, const std::function<Eigen::VectorXd(const Eigen::VectorXd &, const json &)> &dJi_dintegrals, Eigen::VectorXd &one_form);
		// unify transient and static
		void dJ_shape(const IntegrableFunctional &j, Eigen::VectorXd &one_form)
		{
			if (problem->is_time_dependent())
				dJ_shape_transient(j, one_form);
			else
				dJ_shape_static(j, one_form);
		}
		void dJ_material(const IntegrableFunctional &j, Eigen::VectorXd &one_form)
		{
			if (problem->is_time_dependent())
				dJ_material_transient(j, one_form);
			else
				dJ_material_static(j, one_form);
		}

		// unify everything
		Eigen::VectorXd sum_gradient(const SummableFunctional &j, const std::string &type)
		{
			assert((problem->is_time_dependent() && diff_cached.size() > 0));

			Eigen::VectorXd grad;
			if (type == "material")
				dJ_material_static(j, grad);
			else
			{
				logger().error("Only static problem and material derivative is supported in sum_gradient!");
				exit(0);
			}
			return grad;
		}
		Eigen::VectorXd integral_gradient(const IntegrableFunctional &j, const std::string &type)
		{
			// assert((problem->is_time_dependent() && diff_cached.size() > 0) || (!problem->is_time_dependent() && sol.size() > 0));

			Eigen::VectorXd grad;
			if (type == "material")
				dJ_material(j, grad);
			else if (type == "shape")
				dJ_shape(j, grad);
			else if (type == "topology")
				dJ_topology_static(j, grad);
			else
			{
				assert(problem->is_time_dependent());
				if (type == "initial-velocity")
				{
					Eigen::VectorXd tmp;
					dJ_initial_condition(j, tmp);
					grad = tmp.tail(tmp.size() / 2);
				}
				else if (type == "initial-position")
				{
					Eigen::VectorXd tmp;
					dJ_initial_condition(j, tmp);
					grad = tmp.head(tmp.size() / 2);
				}
				else if (type == "initial-condition")
					dJ_initial_condition(j, grad);
				else if (type == "friction-coefficient")
				{
					grad.resize(1);
					dJ_friction_transient(j, grad(0));
				}
				else if (type == "damping-parameter")
					dJ_damping_transient(j, grad);
				else if (type == "material-full")
					dJ_full_material_transient(j, grad);
				else
					logger().error("Unknown derivative type!");
			}
			return grad;
		}
		Eigen::VectorXd integral_gradient(const std::vector<IntegrableFunctional> &js, const std::function<Eigen::VectorXd(const Eigen::VectorXd &, const json &)> &dJi_dintegrals, const std::string &type)
		{
			assert(problem->is_time_dependent() && diff_cached.size() > 0);

			Eigen::VectorXd grad;
			if (type == "material")
				dJ_material_transient(js, dJi_dintegrals, grad);
			else if (type == "shape")
				dJ_shape_transient(js, dJi_dintegrals, grad);
			else if (type == "initial-velocity")
			{
				Eigen::VectorXd tmp;
				dJ_initial_condition(js, dJi_dintegrals, tmp);
				grad = tmp.tail(tmp.size() / 2);
			}
			else if (type == "initial-position")
			{
				Eigen::VectorXd tmp;
				dJ_initial_condition(js, dJi_dintegrals, tmp);
				grad = tmp.head(tmp.size() / 2);
			}
			else if (type == "initial-condition")
				dJ_initial_condition(js, dJi_dintegrals, grad);
			else if (type == "friction-coefficient")
			{
				grad.resize(1);
				dJ_friction_transient(js, dJi_dintegrals, grad(0));
			}
			else if (type == "damping-parameter")
				dJ_damping_transient(js, dJi_dintegrals, grad);
			else if (type == "material-full")
				dJ_full_material_transient(js, dJi_dintegrals, grad);
			else
				logger().error("Unknown derivative type!");

			return grad;
		}
		// Alters the mesh for a given discrete perturbation (vector) field over the vertices
		void perturb_mesh(const Eigen::MatrixXd &perturbation);
		void perturb_material(const Eigen::MatrixXd &perturbation);

		// to replace the initial condition
		Eigen::MatrixXd initial_sol_update, initial_vel_update;
		// downsample grad on P2 nodes to grad on P1 nodes, only for P2 contact shape derivative
		StiffnessMatrix down_sampling_mat;

		void project_to_lower_order(Eigen::MatrixXd &y);

		//---------------------------------------------------
		//-----------------OUTPUT----------------------------
		//---------------------------------------------------

		/// boundary visualization mesh vertices
		Eigen::MatrixXd boundary_vis_vertices;
		/// boundary visualization mesh vertices pre image in ref element
		Eigen::MatrixXd boundary_vis_local_vertices;
		/// boundary visualization mesh connectivity
		Eigen::MatrixXi boundary_vis_elements;
		/// boundary visualization mesh elements ids
		Eigen::MatrixXi boundary_vis_elements_ids;
		/// boundary visualization mesh edge/face id
		Eigen::MatrixXi boundary_vis_primitive_ids;
		/// boundary visualization mesh normals
		Eigen::MatrixXd boundary_vis_normals;

		/// grid mesh points to export solution sampled on a grid
		Eigen::MatrixXd grid_points;
		/// grid mesh mapping to fe elements
		Eigen::MatrixXi grid_points_to_elements;
		/// grid mesh boundaries
		Eigen::MatrixXd grid_points_bc;

		/// spectrum of the stiffness matrix, enable only if POLYSOLVE_WITH_SPECTRA is ON (off by default)
		Eigen::Vector4d spectrum;

		/// information of the solver, eg num iteration, time, errors, etc
		/// the informations varies depending on the solver
		json solver_info;

		/// max edge lenght
		double mesh_size;
		/// min edge lenght
		double min_edge_length;
		/// avg edge lenght
		double average_edge_length;

		/// errors, lp_err is in fact an L8 error
		double l2_err, linf_err, lp_err, h1_err, h1_semi_err, grad_max_err;

		/// non zeros and sytem matrix size
		/// num dof is the total dof in the system
		long long nn_zero, mat_size, num_dofs;

		/// time to construct the basis
		double building_basis_time;
		/// time to load the mesh
		double loading_mesh_time;
		/// time to build the polygonal/polyhedral bases
		double computing_poly_basis_time;
		/// time to assembly
		double assembling_stiffness_mat_time;
		/// time to computing the rhs
		double assigning_rhs_time;
		/// time to solve
		double solving_time;
		/// time to compute error
		double computing_errors_time;

		/// statiscs on angle, compute only when using p_ref (false by default)
		double max_angle;
		/// statiscs on tri/tet quality, compute only when using p_ref (false by default)
		double sigma_max, sigma_min, sigma_avg;

		/// number of flipped elements, compute only when using count_flipped_els (false by default)
		int n_flipped;

		/// statiscs on the mesh (simplices)
		int simplex_count;
		/// statiscs on the mesh (regular quad/hex part of the mesh), see Polyspline paper for desciption
		int regular_count;
		/// statiscs on the mesh (regular quad/hex boundary part of the mesh), see Polyspline paper for desciption
		int regular_boundary_count;
		/// statiscs on the mesh (irregular quad/hex part of the mesh), see Polyspline paper for desciption
		int simple_singular_count;
		/// statiscs on the mesh (irregular quad/hex part of the mesh), see Polyspline paper for desciption
		int multi_singular_count;
		/// statiscs on the mesh (boundary quads/hexs), see Polyspline paper for desciption
		int boundary_count;
		/// statiscs on the mesh (irregular boundary quad/hex part of the mesh), see Polyspline paper for desciption
		int non_regular_boundary_count;
		/// statiscs on the mesh (irregular quad/hex part of the mesh), see Polyspline paper for desciption
		int non_regular_count;
		/// statiscs on the mesh (not quad/hex simplex), see Polyspline paper for desciption
		int undefined_count;
		/// statiscs on the mesh (irregular boundary quad/hex part of the mesh), see Polyspline paper for desciption
		int multi_singular_boundary_count;

		/// flag to decide if exporting the time dependent solution to files
		/// or save it in the solution_frames array
		bool solve_export_to_file = true;
		/// saves the frames in a vector instead of VTU
		std::vector<SolutionFrame> solution_frames;

		/// extracts the boundary mesh for visualization, called in build_basis
		void extract_vis_boundary_mesh();
		/// extracts the boundary mesh for collision, called in build_basis
		void build_collision_mesh(
			ipc::CollisionMesh &collision_mesh_, 
			Eigen::MatrixXd &boundary_nodes_pos_,
			Eigen::MatrixXi &boundary_edges_,
			Eigen::MatrixXi &boundary_triangles_,
			const int n_bases_, 
			const std::vector<ElementBases> &bases_) const;

		//add lagrangian multiplier rows for pure neumann/periodic boundary condition, returns the number of rows added
		int remove_pure_neumann_singularity(StiffnessMatrix &A);
		int remove_pure_periodic_singularity(StiffnessMatrix &A);

		/// compute the errors, not part of solve
		void compute_errors();
		/// saves all data on the disk according to the input params
		void export_data();

		/// saves the output statistic to a stream
		/// @param[in] out stream to write output
		void save_json(std::ostream &out);
		/// saves the output statistic to a json object
		/// @param[in] j output json
		void save_json(nlohmann::json &j);
		/// saves the output statistic to disc accoding to params
		void save_json();

		/// evaluates the function fun at the vertices on the mesh
		/// @param[in] actual_dim is the size of the problem (e.g., 1 for Laplace, dim for elasticity)
		/// @param[in] basis basis function
		/// @param[in] fun function to interpolate
		/// @param[out] result output
		void compute_vertex_values(int actual_dim, const std::vector<ElementBases> &basis, const MatrixXd &fun, Eigen::MatrixXd &result);
		/// compute von mises stress at quadrature points for the function fun, also compute the interpolated function
		/// @param[in] fun function to used
		/// @param[out] result output displacement
		/// @param[out] von_mises output von mises
		void compute_stress_at_quadrature_points(const MatrixXd &fun, Eigen::MatrixXd &result, Eigen::VectorXd &von_mises);
		/// interpolate the function fun.
		/// @param[in] n_points is the size of the output.
		/// @param[in] fun function to used
		/// @param[out] result output
		/// @param[in] use_sampler uses the sampler or not
		/// @param[in] boundary_only interpolates only at boundary elements
		void interpolate_function(const int n_points, const Eigen::MatrixXd &fun, Eigen::MatrixXd &result, const bool use_sampler, const bool boundary_only);
		///interpolate the function fun.
		/// @param[in] n_points is the size of the output.
		/// @param[in] actual_dim is the size of the problem (e.g., 1 for Laplace, dim for elasticity)
		/// @param[in] basis basis function
		/// @param[in] fun function to used
		/// @param[out] result output
		/// @param[in] use_sampler uses the sampler or not
		/// @param[in] boundary_only interpolates only at boundary elements
		void interpolate_function(const int n_points, const int actual_dim, const std::vector<ElementBases> &basis, const MatrixXd &fun, MatrixXd &result, const bool use_sampler, const bool boundary_only);

		/// interpolate solution and gradient at element (calls interpolate_at_local_vals with sol)
		/// @param[in] el_index element index
		/// @param[in] local_pts points in the reference element
		/// @param[out] result output
		/// @param[out] result_grad output gradients
		void interpolate_at_local_vals(const int el_index, const MatrixXd &local_pts, MatrixXd &result, MatrixXd &result_grad);
		/// interpolate solution and gradient at element (calls interpolate_at_local_vals with sol)
		/// @param[in] el_index element index
		/// @param[in] local_pts points in the reference element
		/// @param[in] fun function to used
		/// @param[out] result output
		/// @param[out] result_grad output gradients
		void interpolate_at_local_vals(const int el_index, const MatrixXd &local_pts, const MatrixXd &fun, MatrixXd &result, MatrixXd &result_grad);
		/// interpolate the function fun and its gradient at in element el_index for the local_pts in the reference element using bases bases
		/// interpolate solution and gradient at element (calls interpolate_at_local_vals with sol)
		/// @param[in] el_index element index
		/// @param[in] actual_dim is the size of the problem (e.g., 1 for Laplace, dim for elasticity)
		/// @param[in] bases basis function
		/// @param[in] local_pts points in the reference element
		/// @param[in] fun function to used
		/// @param[out] result output
		/// @param[out] result_grad output gradients
		void interpolate_at_local_vals(const int el_index, const int actual_dim, const std::vector<ElementBases> &bases, const MatrixXd &local_pts, const MatrixXd &fun, MatrixXd &result, MatrixXd &result_grad);

		/// checks if mises are not nan
		/// @param[in] fun function to used
		/// @param[in] use_sampler uses the sampler or not
		/// @param[in] boundary_only interpolates only at boundary elements
		/// @return if mises are nan
		bool check_scalar_value(const Eigen::MatrixXd &fun, const bool use_sampler, const bool boundary_only);
		/// computes scalar quantity of funtion (ie von mises for elasticity and norm of velocity for fluid)
		/// @param[in] n_points is the size of the output.
		/// @param[in] fun function to used
		/// @param[out] result scalar value
		/// @param[in] use_sampler uses the sampler or not
		/// @param[in] boundary_only interpolates only at boundary elements
		void compute_scalar_value(const int n_points, const Eigen::MatrixXd &fun, Eigen::MatrixXd &result, const bool use_sampler, const bool boundary_only);
		/// computes scalar quantity of funtion (ie von mises for elasticity and norm of velocity for fluid)
		/// the scalar value is averaged around every node to make it continuos
		/// @param[in] n_points is the size of the output.
		/// @param[in] fun function to used
		/// @param[out] result_scalar scalar value
		/// @param[out] result_tensor tensor value
		/// @param[in] use_sampler uses the sampler or not
		/// @param[in] boundary_only interpolates only at boundary elements
		void average_grad_based_function(const int n_points, const MatrixXd &fun, MatrixXd &result_scalar, MatrixXd &result_tensor, const bool use_sampler, const bool boundary_only);
		/// compute tensor quantity (ie stress tensor or velocy)
		/// @param[in] n_points is the size of the output.
		/// @param[in] fun function to used
		/// @param[out] result resulting tensor
		/// @param[in] use_sampler uses the sampler or not
		/// @param[in] boundary_only interpolates only at boundary elements
		void compute_tensor_value(const int n_points, const Eigen::MatrixXd &fun, Eigen::MatrixXd &result, const bool use_sampler, const bool boundary_only);

		/// computes integrated solution (fun) per surface face. pts and faces are the boundary are the boundary on the rest configuration
		/// @param[in] pts boundary points
		/// @param[in] faces boundary faces
		/// @param[in] fun function to used
		/// @param[in] compute_avg if compute the average across elements
		/// @param[out] result resulting value
		void interpolate_boundary_function(const MatrixXd &pts, const MatrixXi &faces, const MatrixXd &fun, const bool compute_avg, MatrixXd &result);
		/// computes integrated solution (fun) per surface face vertex. pts and faces are the boundary are the boundary on the rest configuration
		/// @param[in] pts boundary points
		/// @param[in] faces boundary faces
		/// @param[in] fun function to used
		/// @param[out] result resulting value
		void interpolate_boundary_function_at_vertices(const MatrixXd &pts, const MatrixXi &faces, const MatrixXd &fun, MatrixXd &result);
		/// computes traction foces for fun (tensor * surface normal) result, stress tensor, and von mises, per surface face. pts and faces are the boundary on the rest configuration.
		/// disp is the displacement of the surface vertices
		/// @param[in] pts boundary points
		/// @param[in] faces boundary faces
		/// @param[in] fun function to used
		/// @param[in] disp displacement to deform mesh
		/// @param[in] compute_avg if compute the average across elements
		/// @param[out] result resulting value
		/// @param[out] stresses resulting stresses
		/// @param[out] mises resulting mises
		/// @param[in] skip_orientation skip reorientation of surface
		void interpolate_boundary_tensor_function(const MatrixXd &pts, const MatrixXi &faces, const MatrixXd &fun, const MatrixXd &disp, const bool compute_avg, MatrixXd &result, MatrixXd &stresses, MatrixXd &mises, const bool skip_orientation = false);
		/// same as interpolate_boundary_tensor_function with disp=0
		/// @param[in] pts boundary points
		/// @param[in] faces boundary faces
		/// @param[in] fun function to used
		/// @param[in] compute_avg if compute the average across elements
		/// @param[out] result resulting value
		/// @param[out] stresses resulting stresses
		/// @param[out] mises resulting mises
		/// @param[in] skip_orientation skip reorientation of surface
		void interpolate_boundary_tensor_function(const MatrixXd &pts, const MatrixXi &faces, const MatrixXd &fun, const bool compute_avg, MatrixXd &result, MatrixXd &stresses, MatrixXd &mises, const bool skip_orientation = false);

		/// returns a triangulated representation of the sideset. sidesets contains integers mapping to faces
		/// @param[in] pts boundary points
		/// @param[in] faces boundary faces
		/// @param[out] sidesets resulting sidesets
		void get_sidesets(Eigen::MatrixXd &pts, Eigen::MatrixXi &faces, Eigen::MatrixXd &sidesets);

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

		/// compute stats (counts els type, mesh lenght, etc), step 1 of solve
		void compute_mesh_stats();

		/// builds visualzation mesh, upsampled mesh used for visualization
		/// the visualization mesh is a dense mesh per element all disconnected
		/// it also retuns the mapping to element id and discretization of every elment
		/// works in 2 and 3d. if the mesh is not simplicial it gets tri/tet halized
		/// @param[out] points mesh points
		/// @param[out] tets mesh cells
		/// @param[out] el_id mapping from points to elements id
		/// @param[out] discr mapping from points to discretization order
		void build_vis_mesh(Eigen::MatrixXd &points, Eigen::MatrixXi &tets, Eigen::MatrixXi &el_id, Eigen::MatrixXd &discr);
		/// builds high-der visualzation mesh per element all disconnected
		/// it also retuns the mapping to element id and discretization of every elment
		/// works in 2 and 3d. if the mesh is not simplicial it gets tri/tet halized
		/// @param[out] points mesh points
		/// @param[out] elements mesh high-order cells
		/// @param[out] el_id mapping from points to elements id
		/// @param[out] discr mapping from points to discretization order
		void build_high_oder_vis_mesh(Eigen::MatrixXd &points, std::vector<std::vector<int>> &elements, Eigen::MatrixXi &el_id, Eigen::MatrixXd &discr);

		/// saves the vtu file for time t
		/// @param[in] name filename
		/// @param[in] t time
		void save_vtu(const std::string &name, const double t);
		/// saves the volume vtu file
		/// @param[in] name filename
		/// @param[in] t time
		void save_volume(const std::string &path, const double t);
		/// saves the surface vtu file for for surface quantites, eg traction forces
		/// @param[in] name filename
		/// @param[in] t time
		void save_surface(const std::string &name);
		///saves the wireframe
		/// @param[in] name filename
		/// @param[in] t time
		void save_wire(const std::string &name, const double t);
		/// save a PVD of a time dependent simulation
		/// @param[in] name filename
		/// @param[in] vtu_names names of the vtu files
		/// @param[in] time_steps total time stesp
		/// @param[in] t0 initial time
		/// @param[in] dt delta t
		/// @param[in] skip_frame every which frame to skip
		void save_pvd(const std::string &name, const std::function<std::string(int)> &vtu_names, int time_steps, double t0, double dt, int skip_frame = 1);
		/// saves a timestep
		/// @param[in] time time in secs
		/// @param[in] t time index
		/// @param[in] t0 initial time
		/// @param[in] dt delta t
		void save_timestep(const double time, const int t, const double t0, const double dt);
		/// saves a subsolve when save_solve_sequence_debug is true
		/// @param[in] i sub solve index
		/// @param[in] t time index
		void save_subsolve(const int i, const int t);

		/// samples to solution on the visualization mesh and return the vis mesh (points and tets) and the interpolated values (fun)
		void get_sampled_solution(Eigen::MatrixXd &points, Eigen::MatrixXi &tets, Eigen::MatrixXd &fun, bool boundary_only = false)
		{
			// TODO: fix me TESEO
			// Eigen::MatrixXd discr;
			// Eigen::MatrixXi el_id;
			// const bool tmp = args["export"]["vis_boundary_only"];
			// args["export"]["vis_boundary_only"] = boundary_only;

			// build_vis_mesh(points, tets, el_id, discr);
			// interpolate_function(points.rows(), sol, fun, false, boundary_only);

			// args["export"]["vis_boundary_only"] = tmp;
		}

		/// samples to stess tensor on the visualization mesh and return them (fun)
		void get_stresses(Eigen::MatrixXd &fun, bool boundary_only = false)
		{
			// TODO: fix me TESEO
			// Eigen::MatrixXd points;
			// Eigen::MatrixXi tets;
			// Eigen::MatrixXi el_id;
			// Eigen::MatrixXd discr;
			// const bool tmp = args["export"]["vis_boundary_only"];
			// args["export"]["vis_boundary_only"] = boundary_only;

			// build_vis_mesh(points, tets, el_id, discr);
			// compute_tensor_value(points.rows(), sol, fun, false, boundary_only);

			// args["export"]["vis_boundary_only"] = tmp;
		}

		/// samples to von mises stesses on the visualization mesh and return them (fun)
		void get_sampled_mises(Eigen::MatrixXd &fun, bool boundary_only = false)
		{
			// TODO: fix me TESEO
			// Eigen::MatrixXd points;
			// Eigen::MatrixXi tets;
			// Eigen::MatrixXi el_id;
			// Eigen::MatrixXd discr;
			// const bool tmp = args["export"]["vis_boundary_only"];
			// args["export"]["vis_boundary_only"] = boundary_only;

			// build_vis_mesh(points, tets, el_id, discr);
			// compute_scalar_value(points.rows(), sol, fun, false, boundary_only);

			// args["export"]["vis_boundary_only"] = tmp;
		}

		/// samples to averaged von mises stesses on the visualization mesh and return them (fun)
		void get_sampled_mises_avg(Eigen::MatrixXd &fun, Eigen::MatrixXd &tfun, bool boundary_only = false)
		{
			// TODO: fix me TESEO
			// Eigen::MatrixXd points;
			// Eigen::MatrixXi tets;
			// Eigen::MatrixXi el_id;
			// Eigen::MatrixXd discr;
			// const bool tmp = args["export"]["vis_boundary_only"];
			// args["export"]["vis_boundary_only"] = boundary_only;

			// build_vis_mesh(points, tets, el_id, discr);
			// average_grad_based_function(points.rows(), sol, fun, tfun, false, boundary_only);

			// args["export"]["vis_boundary_only"] = tmp;
		}

		/// set the multimaterial, this is mean for internal usage.
		void set_materials();

		//homogenization study of unit cell
		void homogenization(Eigen::MatrixXd &C_H)
		{
			compute_mesh_stats();
			build_basis();
			assemble_stiffness_mat();
			assemble_rhs();
			solve_homogenization();
			compute_homogenized_tensor(C_H);
		}
		void homogenize_topology_new(Eigen::MatrixXd &C_H)
		{
			compute_mesh_stats();
			build_basis();
			assemble_stiffness_mat();
		}
		void homogenize_weighted_linear_elasticity(Eigen::MatrixXd &C_H);
		void homogenize_weighted_linear_elasticity_grad(Eigen::MatrixXd &C_H, Eigen::MatrixXd &grad);
		void homogenize_weighted_stokes(Eigen::MatrixXd &K_H);
		void homogenize_weighted_stokes_grad(Eigen::MatrixXd &K_H, Eigen::MatrixXd &grad);

		double min_solid_density = 0;

	private:
		/// splits the solution in solution and pressure for mixed problems
		void sol_to_pressure();
		/// builds bases for polygons, called inside build_basis
		void build_polygonal_basis();

#ifdef POLYFEM_WITH_TBB
		/// limits the number of used threads
		std::shared_ptr<tbb::global_control> thread_limiter;
#endif
	};

} // namespace polyfem
