#pragma once

#include <polyfem/basis/ElementBases.hpp>
#include <polyfem/assembler/ElementAssemblyValues.hpp>
#include <polyfem/assembler/AssemblyValsCache.hpp>
#include <polyfem/assembler/RhsAssembler.hpp>
#include <polyfem/problem/Problem.hpp>
#include <polyfem/mesh/Mesh.hpp>
#include <polyfem/mesh/Obstacle.hpp>
#include <polyfem/utils/RefElementSampler.hpp>
#include <polyfem/mesh/LocalBoundary.hpp>
#include <polyfem/basis/InterfaceData.hpp>
#include <polyfem/assembler/AssemblerUtils.hpp>
#include <polyfem/utils/ElasticityUtils.hpp>
#include <polyfem/Common.hpp>
#include <polyfem/utils/Logger.hpp>

// Instead of including this do a forward declaration
// #include <polyfem/NonlinearSolver.hpp>

#include <polyfem/mesh/mesh2D/NCMesh2D.hpp>
#include <polyfem/mesh/mesh2D/CMesh2D.hpp>
#include <polyfem/mesh/mesh3D/Mesh3D.hpp>
#include <polyfem/utils/StringUtils.hpp>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#ifdef POLYFEM_WITH_TBB
#include <tbb/global_control.h>
#endif

#include <memory>
#include <string>

#include <ipc/collision_mesh.hpp>

#include <ipc/utils/logger.hpp>

// Forward declaration
namespace cppoptlib
{
	template <typename ProblemType>
	class NonlinearSolver;
}

namespace polyfem
{
	//class used to save the solution of time dependent problems in code instead of saving it to the disc
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

	class StepData
	{
	public:
		std::shared_ptr<assembler::RhsAssembler> rhs_assembler;
		std::shared_ptr<solver::NLProblem> nl_problem;
		std::shared_ptr<solver::ALNLProblem> alnl_problem;
	};

	//main class that contains the polyfem solver
	class State
	{
	public:
		~State() = default;

		State(const unsigned int max_threads = std::numeric_limits<unsigned int>::max());

		//initalizing the logger
		//log_file is to write it to a file (use log_file="") to output to consolle
		//log_level 0 all message, 6 no message. 2 is info, 1 is debug
		//quiet the log
		void init_logger(const std::string &log_file, int log_level, const bool is_quiet);

		//initalizing the logger
		//writes to an outputstream
		void init_logger(std::ostream &os, int log_level);

		//initalizing the logger meant for internal usage
		void init_logger(std::vector<spdlog::sink_ptr> &sinks, int log_level);

		//initialize the polyfem solver with a json settings
		void init(const json &args, const std::string &output_dir = "");
		// void init(const std::string &json);

		void init_timesteps();

		//change log level, log_level 0 all message, 6 no message. 2 is info, 1 is debug
		void set_log_level(int log_level)
		{
			spdlog::level::level_enum level =
				static_cast<spdlog::level::level_enum>(std::max(0, std::min(6, log_level)));
			spdlog::set_level(level);
			IPC_LOG(set_level(level));
		}

		//get the output log as json
		//this is *not* what gets printed but more informative
		//information, eg it contains runtimes, errors, etc.
		std::string get_log()
		{
			std::stringstream ss;
			save_json(ss);
			return ss.str();
		}

		//solver settings
		json args;

		// Directory for output files
		std::string output_dir;

		//assembler, it dispatches call to the differnt assembers based on the formulation
		assembler::AssemblerUtils assembler;
		//current problem, it contains rhs and bc
		std::shared_ptr<problem::Problem> problem;

		//density of the input, default=1.
		Density density;

		//FE bases, the size is #elements
		std::vector<ElementBases> bases;
		//FE pressure bases for mixed elements, the size is #elements
		std::vector<ElementBases> pressure_bases;

		// Mapping from input nodes to FE nodes
		std::vector<int> primitive_to_node;

		//Geometric mapping bases, if the elements are isoparametric, this list is empty
		std::vector<ElementBases> geom_bases;

		//used to store assembly values for small problems
		assembler::AssemblyValsCache ass_vals_cache;
		assembler::AssemblyValsCache pressure_ass_vals_cache;

		//list of boundary nodes
		std::vector<int> boundary_nodes;
		//list of neumann boundary nodes
		std::vector<int> pressure_boundary_nodes;
		//mapping from elements to nodes for all mesh
		std::vector<mesh::LocalBoundary> total_local_boundary;
		//mapping from elements to nodes for dirichlet boundary conditions
		std::vector<mesh::LocalBoundary> local_boundary;
		//mapping from elements to nodes for neumann boundary conditions
		std::vector<mesh::LocalBoundary> local_neumann_boundary;
		//nodes on the boundary of polygonal elements, used for harmonic bases
		std::map<int, InterfaceData> poly_edge_to_data;

		//current mesh, it can be a Mesh2D or Mesh3D
		std::unique_ptr<mesh::Mesh> mesh;
		//Obstacles used in collisions
		mesh::Obstacle obstacle;
		//used to sample the solution
		utils::RefElementSampler ref_element_sampler;

		//polygons, used since poly have no geom mapping
		std::map<int, Eigen::MatrixXd> polys;
		//polyhedra, used since poly have no geom mapping
		std::map<int, std::pair<Eigen::MatrixXd, Eigen::MatrixXi>> polys_3d;

		//parent element used to track refinements
		std::vector<int> parent_elements;

		//average system mass, used for contact with IPC
		double avg_mass;

		//stiffness and mass matrix.
		//Stiffness is not compute for non linear problems
		//Mass is computed only for time dependent problems
		StiffnessMatrix stiffness, mass;
		//System righ-hand side.
		//rhs_in is an input that, if not empty, gets copied to rhs
		Eigen::MatrixXd rhs, rhs_in;

		//solution and pressure solution
		//if the problem is not mixed, pressure is empty
		Eigen::MatrixXd sol, pressure;

		//boundary mesh used for collision
		//boundary_nodes_pos contains the total number of nodes, the internal ones are zero
		//for high-order fem the faces are triangulated
		//this is currently supported only for tri and tet meshes
		Eigen::MatrixXd boundary_nodes_pos;
		Eigen::MatrixXi boundary_edges;     // indices into full vertices
		Eigen::MatrixXi boundary_triangles; // indices into full vertices
		ipc::CollisionMesh collision_mesh;  // indices into surface vertices

		Eigen::MatrixXd boundary_nodes_pos_pressure;
		Eigen::MatrixXi boundary_edges_pressure;
		Eigen::MatrixXi boundary_triangles_pressure;

		//boundary visualization mesh
		Eigen::MatrixXd boundary_vis_vertices;
		Eigen::MatrixXd boundary_vis_local_vertices;
		Eigen::MatrixXi boundary_vis_elements;
		Eigen::MatrixXi boundary_vis_elements_ids;
		Eigen::MatrixXi boundary_vis_primitive_ids;
		Eigen::MatrixXd boundary_vis_normals;

		//grid mesh
		Eigen::MatrixXd grid_points;
		Eigen::MatrixXi grid_points_to_elements;
		Eigen::MatrixXd grid_points_bc;

		//spectrum of the stiffness matrix, enable only if POLYSOLVE_WITH_SPECTRA is ON (off by default)
		Eigen::Vector4d spectrum;

		//information of the solver, eg num iteration, time, errors, etc
		//the informations varies depending on the solver
		json solver_info;

		//use average pressure for stokes problem to fix the additional dofs, true by default
		//if false, it will fix one pressure node to zero
		bool use_avg_pressure;

		//stores if input json contains dhat
		bool has_dhat = false;

		//number of bases and pressure bases
		int n_bases, n_pressure_bases;
		//vector of discretization oders, used when not all elements have the same degree
		//one per element
		Eigen::VectorXi disc_orders;

		//max edge lenght
		double mesh_size;
		//min edge lenght
		double min_edge_length;
		//avg edge lenght
		double average_edge_length;

		//errors, lp_err is in fact an L8 error
		double l2_err, linf_err, lp_err, h1_err, h1_semi_err, grad_max_err;

		//non zeros and sytem matrix size
		//num dof is the totdal dof in the system
		long long nn_zero, mat_size, num_dofs;

		//timings
		//construct the basis
		double building_basis_time;
		//load the mesh
		double loading_mesh_time;
		//build the polygonal/polyhedral bases
		double computing_poly_basis_time;
		//assembly
		double assembling_stiffness_mat_time;
		//computing the rhs
		double assigning_rhs_time;
		//solve
		double solving_time;
		//compute error
		double computing_errors_time;

		//statiscs on angle, compute only when using p_ref (false by default)
		double max_angle;
		double sigma_max, sigma_min, sigma_avg;

		//number of flipped elements, compute only when using count_flipped_els (false by default)
		int n_flipped;

		//statiscs on the mesh, see Polyspline paper for desciption
		int simplex_count;
		int regular_count;
		int regular_boundary_count;
		int simple_singular_count;
		int multi_singular_count;
		int boundary_count;
		int non_regular_boundary_count;
		int non_regular_count;
		int undefined_count;
		int multi_singular_boundary_count;

		//flag to decide if exporting the time dependent solution to files
		//or save it in the solution_frames array
		bool solve_export_to_file = true;
		std::vector<SolutionFrame> solution_frames;

		//utility function that gets the problem params (eg material)
		//it adds the problem dimension from the problem and PDE
		json build_json_params();

		//computes the mesh size, it samples every edges n_samples times
		//uses curved_mesh_size (false by default) to compute the size of
		//the linear mesh
		void compute_mesh_size(const mesh::Mesh &mesh, const std::vector<ElementBases> &bases, const int n_samples);

		//loads the mesh from the json arguments
		void load_mesh(bool non_conforming = false,
					   const std::vector<std::string> &names = std::vector<std::string>(),
					   const std::vector<Eigen::MatrixXi> &cells = std::vector<Eigen::MatrixXi>(),
					   const std::vector<Eigen::MatrixXd> &vertices = std::vector<Eigen::MatrixXd>());
		//loads a febio file, uses args_in for default
		void load_febio(const std::string &path, const json &args_in);
		//loads the mesh from a geogram mesh, skip_boundary_sideset = false it uses the lambda boundary_marker to assigm the sideset
		//the input of the lambda is the face barycenter, the output is the sideset id
		void load_mesh(GEO::Mesh &meshin, const std::function<int(const RowVectorNd &)> &boundary_marker, bool non_conforming = false, bool skip_boundary_sideset = false);
		//loads a mesh from a path
		void load_mesh(const std::string &path, bool non_conforming = false)
		{
			//TODO fix Zach
			args["mesh"] = path;
			load_mesh(non_conforming);
		}
		//loads a mesh from a path and uses the bc_tag to assign sideset ids
		//the bc_tag file should contain a list of integers, one per face
		void load_mesh(const std::string &path, const std::string &bc_tag, bool non_conforming = false)
		{
			//TODO fix Zach
			args["mesh"] = path;
			args["bc_tag"] = bc_tag;
			load_mesh(non_conforming);
		}
		//load the mesh from V and F, V is #vertices x dim, F is #elements x size (size = 3 for triangle mesh, size=4 for a quaud mesh if dim is 2)
		void load_mesh(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, bool non_conforming = false)
		{
			if (V.cols() == 2)
				if (non_conforming)
					mesh = std::make_unique<mesh::NCMesh2D>();
				else
					mesh = std::make_unique<mesh::CMesh2D>();
			else
				mesh = std::make_unique<mesh::Mesh3D>();
			mesh->build_from_matrices(V, F);

			load_mesh(non_conforming);
		}

		//set the multimaterial, this is mean for internal usage.
		void set_multimaterial(const std::function<void(const Eigen::MatrixXd &, const Eigen::MatrixXd &, const Eigen::MatrixXd &)> &setter);

		//set the boundary sideset
		//from a lambda that takes the face/edge barycenter
		void set_boundary_side_set(const std::function<int(const RowVectorNd &)> &boundary_marker) { mesh->compute_boundary_ids(boundary_marker); }
		//from a lambda that takes the face/edge barycenter and a flag if the face/edge is boundary or not (used to set internal boundaries)
		void set_boundary_side_set(const std::function<int(const RowVectorNd &, bool)> &boundary_marker) { mesh->compute_boundary_ids(boundary_marker); }
		//from a lambda that takes the face/edge vertices and a flag if the face/edge is boundary or not (used to set internal boundaries)
		void set_boundary_side_set(const std::function<int(const std::vector<int> &, bool)> &boundary_marker) { mesh->compute_boundary_ids(boundary_marker); }

		//solves the problem
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

		//internal methods, they are called from solve

		//builds the bases step 2 of solve
		void build_basis();
		//extracts the boundary mesh for collision, called in build_basis
		void build_collision_mesh();
		//extracts the boundary mesh
		void extract_boundary_mesh(
			const std::vector<ElementBases> &bases,
			Eigen::MatrixXd &boundary_nodes_pos,
			Eigen::MatrixXi &boundary_edges,
			Eigen::MatrixXi &boundary_triangles) const;

		//extracts the boundary mesh for visualization, called in build_basis
		void extract_vis_boundary_mesh();
		//assemble matrices, step 4 of solve
		void assemble_stiffness_mat();
		//compute rhs, step 3 of solve
		void assemble_rhs();
		//solves the proble, step 5
		void solve_problem();

		//timedependent stuff cached
		StepData step_data;
		//Aux solving functions, c_sol=x are necessary since they contain the pressure, while sol dosent
		void init_transient(Eigen::VectorXd &c_sol);
		void solve_transient_navier_stokes_split(const int time_steps, const double dt, const assembler::RhsAssembler &rhs_assembler);
		void solve_transient_navier_stokes(const int time_steps, const double t0, const double dt, const assembler::RhsAssembler &rhs_assembler, Eigen::VectorXd &c_sol);
		void solve_transient_scalar(const int time_steps, const double t0, const double dt, const assembler::RhsAssembler &rhs_assembler, Eigen::VectorXd &x);
		void solve_transient_tensor_linear(const int time_steps, const double t0, const double dt, const assembler::RhsAssembler &rhs_assembler);
		void solve_transient_tensor_non_linear(const int time_steps, const double t0, const double dt, const assembler::RhsAssembler &rhs_assembler);
		void solve_transient_tensor_non_linear_init(const double t0, const double dt, const assembler::RhsAssembler &rhs_assembler);
		void solve_transient_tensor_non_linear_step(const double t0, const double dt, const int t, json &solver_info);
		void solve_linear();
		void solve_navier_stokes();
		void solve_non_linear();

		//compute the errors, not part of solve
		void compute_errors();
		//saves all data on the disk according to the input params
		void export_data();

		//evaluates the function fun at the vertices on the mesh
		//actual dim is the size of the problem (e.g., 1 for Laplace, dim for elasticity)
		void compute_vertex_values(int actual_dim, const std::vector<ElementBases> &basis,
								   const MatrixXd &fun, Eigen::MatrixXd &result);
		//compute von mises stress at quadrature points for the function fun, also compute the interpolated function
		void compute_stress_at_quadrature_points(const MatrixXd &fun, Eigen::MatrixXd &result, Eigen::VectorXd &von_mises);
		//interpolate the function fun. n_points is the size of the output. boundary_only interpolates only at boundary elements
		void interpolate_function(const int n_points, const Eigen::MatrixXd &fun, Eigen::MatrixXd &result, const bool use_sampler, const bool boundary_only);
		//interpolate the function fun. n_points is the size of the output. boundary_only interpolates only at boundary elements
		//actual dim is the size of the problem (e.g., 1 for Laplace, dim for elasticity)
		void interpolate_function(const int n_points, const int actual_dim, const std::vector<ElementBases> &basis, const MatrixXd &fun, MatrixXd &result, const bool use_sampler, const bool boundary_only);

		//interpolate solution and gradient at in element el_index for the local_pts in the reference element (calls interpolate_at_local_vals with sol)
		void interpolate_at_local_vals(const int el_index, const MatrixXd &local_pts, MatrixXd &result, MatrixXd &result_grad);
		//interpolate the function fun and its gradient at in element el_index for the local_pts in the reference element
		void interpolate_at_local_vals(const int el_index, const MatrixXd &local_pts, const MatrixXd &fun, MatrixXd &result, MatrixXd &result_grad);
		//interpolate the function fun and its gradient at in element el_index for the local_pts in the reference element using bases bases
		//actual dim is the size of the problem (e.g., 1 for Laplace, dim for elasticity)
		void interpolate_at_local_vals(const int el_index, const int actual_dim, const std::vector<ElementBases> &bases, const MatrixXd &local_pts, const MatrixXd &fun, MatrixXd &result, MatrixXd &result_grad);

		bool check_scalar_value(const Eigen::MatrixXd &fun, const bool use_sampler, const bool boundary_only);
		//computes scalar quantity of funtion (ie von mises for elasticity and norm of velocity for fluid)
		void compute_scalar_value(const int n_points, const Eigen::MatrixXd &fun, Eigen::MatrixXd &result, const bool use_sampler, const bool boundary_only);
		//computes scalar quantity of funtion (ie von mises for elasticity and norm of velocity for fluid)
		//the scalar value is averaged around every node to make it continuos
		void average_grad_based_function(const int n_points, const MatrixXd &fun, MatrixXd &result_scalar, MatrixXd &result_tensor, const bool use_sampler, const bool boundary_only);
		//compute tensor quantity (ie stress tensor or velocy)
		void compute_tensor_value(const int n_points, const Eigen::MatrixXd &fun, Eigen::MatrixXd &result, const bool use_sampler, const bool boundary_only);

		//computes integrated solution (fun) per surface face. pts and faces are the boundary are the boundary on the rest configuration
		void interpolate_boundary_function(const MatrixXd &pts, const MatrixXi &faces, const MatrixXd &fun, const bool compute_avg, MatrixXd &result);
		//computes integrated solution (fun) per surface face vertex. pts and faces are the boundary are the boundary on the rest configuration
		void interpolate_boundary_function_at_vertices(const MatrixXd &pts, const MatrixXi &faces, const MatrixXd &fun, MatrixXd &result);
		//computes traction foces for fun (tensor * surface normal) result, stress tensor, and von mises, per surface face. pts and faces are the boundary on the rest configuration.
		//disp is the displacement of the surface vertices
		void interpolate_boundary_tensor_function(const MatrixXd &pts, const MatrixXi &faces, const MatrixXd &fun, const MatrixXd &disp, const bool compute_avg, MatrixXd &result, MatrixXd &stresses, MatrixXd &mises, const bool skip_orientation = false);
		//same as above with disp=0
		void interpolate_boundary_tensor_function(const MatrixXd &pts, const MatrixXi &faces, const MatrixXd &fun, const bool compute_avg, MatrixXd &result, MatrixXd &stresses, MatrixXd &mises, const bool skip_orientation = false);

		//returns a triangulated representation of the sideset. sidesets contains integers mapping to faces
		void get_sidesets(Eigen::MatrixXd &pts, Eigen::MatrixXi &faces, Eigen::MatrixXd &sidesets);

		// Resolve path relative to args["root_path"] if the path is not absolute
		std::string resolve_input_path(const std::string &path) const
		{
			return utils::resolve_path(path, args["root_path"]);
		}
		// Resolve path relative to output_dir if the path is not absolute
		std::string resolve_output_path(const std::string &path) const;

		//saves the output statistic to a stream
		void save_json(std::ostream &out);
		//saves the output statistic to a json object
		void save_json(nlohmann::json &j);
		//saves the output statistic to disc accoding to params
		void save_json();

		//compute stats (counts els type, mesh lenght, etc), step 1 of solve
		void compute_mesh_stats();

		//builds visualzation mesh, upsampled mesh used for visualization
		//the visualization mesh is a dense mesh per element all disconnected
		//it also retuns the mapping to element id and discretization of every elment
		//works in 2 and 3d. if the mesh is not simplicial it gets tri/tet halized
		void build_vis_mesh(Eigen::MatrixXd &points, Eigen::MatrixXi &tets, Eigen::MatrixXi &el_id, Eigen::MatrixXd &discr);
		void build_high_oder_vis_mesh(Eigen::MatrixXd &points, std::vector<std::vector<int>> &elements, Eigen::MatrixXi &el_id, Eigen::MatrixXd &discr);

		//saves the vtu file for time t
		void save_vtu(const std::string &name, const double t);
		//saves the volume vtu file
		void save_volume(const std::string &path, const double t);
		//saves the surface vtu file for for surface quantites, eg traction forces
		void save_surface(const std::string &name);
		//saves an obj of the wireframe
		void save_wire(const std::string &name, const double t);
		// save a PVD of a time dependent simulation
		void save_pvd(const std::string &name, const std::function<std::string(int)> &vtu_names, int time_steps, double t0, double dt, int skip_frame = 1);
		// saves a timestep
		void save_timestep(const double time, const int t, const double t0, const double dt);

		//samples to solution on the visualization mesh and return the vis mesh (points and tets) and the interpolated values (fun)
		void get_sampled_solution(Eigen::MatrixXd &points, Eigen::MatrixXi &tets, Eigen::MatrixXd &fun, bool boundary_only = false)
		{
			//TODO fix me TESEO
			// Eigen::MatrixXd discr;
			// Eigen::MatrixXi el_id;
			// const bool tmp = args["export"]["vis_boundary_only"];
			// args["export"]["vis_boundary_only"] = boundary_only;

			// build_vis_mesh(points, tets, el_id, discr);
			// interpolate_function(points.rows(), sol, fun, false, boundary_only);

			// args["export"]["vis_boundary_only"] = tmp;
		}

		//samples to stess tensor on the visualization mesh and return them (fun)
		void get_stresses(Eigen::MatrixXd &fun, bool boundary_only = false)
		{
			//TODO fix me TESEO
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

		//samples to von mises stesses on the visualization mesh and return them (fun)
		void get_sampled_mises(Eigen::MatrixXd &fun, bool boundary_only = false)
		{
			//TODO fix me TESEO
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

		//samples to averaged von mises stesses on the visualization mesh and return them (fun)
		void get_sampled_mises_avg(Eigen::MatrixXd &fun, Eigen::MatrixXd &tfun, bool boundary_only = false)
		{
			//TODO fix me TESEO
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

		//returns the path of the input mesh (wrappers around the arguments)
		//TODO fix me Zach
		inline std::string mesh_path() const { return resolve_input_path(args["mesh"]); }

		inline bool has_mesh() const
		{
			//TODO fix me Zach
			return !mesh_path().empty() || (args.contains("meshes") && !args["meshes"].empty());
		}

		//return the formulation (checks if the problem is scalar or not)
		inline std::string formulation() const { return args["PDE"]["type"]; }

		//check if using iso parametric bases
		inline bool iso_parametric() const
		{
			if (non_regular_count > 0 || non_regular_boundary_count > 0 || undefined_count > 0)
				return true;

			if (args["space"]["advanced"]["use_spline"])
				return true;

			if (mesh->is_rational())
				return false;

			if (args["space"]["use_p_ref"])
				return false;

			if (mesh->orders().size() <= 0)
			{
				if (args["space"]["discr_order"] == 1)
					return true;
				else
					return args["space"]["advanced"]["iso_parametric"];
			}

			if (mesh->orders().minCoeff() != mesh->orders().maxCoeff())
				return false;

			if (args["space"]["discr_order"] == mesh->orders().minCoeff())
				return true;

			//TODO?
			// if (args["space"]["discr_order"] == 1 && args["force_linear_geometry"])
			// 	return true;

			return args["space"]["advanced"]["iso_parametric"];
		}

		template <typename ProblemType>
		std::shared_ptr<cppoptlib::NonlinearSolver<ProblemType>> make_nl_solver() const;

		//returns solver, preconditioner and solver parameters (wrappers around the arguments)
		inline std::string solver_type() const { return args["solver"]["linear"]["solver"]; }
		inline std::string precond_type() const { return args["solver"]["linear"]["precond"]; }
		inline const json &solver_params() const { return args["solver"]["linear"]; }

		//compute a priori prefinement in 2d and 3d, fills disc_orders
		void p_refinement(const mesh::Mesh2D &mesh2d);
		void p_refinement(const mesh::Mesh3D &mesh3d);

		bool is_obstacle_vertex(const size_t vi) const
		{
			return vi >= boundary_nodes_pos.rows() - obstacle.n_vertices();
		}

		int n_boundary_samples() const
		{
			const int n_b_samples_j = args["space"]["advanced"]["n_boundary_samples"];
			const int discr_order = mesh->orders().size() <= 0 ? 1 : mesh->orders().maxCoeff();
			//TODO verify me
			const int n_b_samples = std::max(n_b_samples_j, discr_order * 2 + 1);

			return n_b_samples;
		}

	private:
		//splits the solution in solution and pressure for mixed problems
		void sol_to_pressure();
		//builds bases for polygons, called inside build_basis
		void build_polygonal_basis();

#ifdef POLYFEM_WITH_TBB
		std::shared_ptr<tbb::global_control> thread_limiter;
#endif
	};

} // namespace polyfem
