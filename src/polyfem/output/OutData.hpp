#pragma once

#include <polyfem/Common.hpp>

#include <polyfem/assembler/Problem.hpp>

#include <polyfem/basis/ElementBases.hpp>

#include <polyfem/mesh/Mesh.hpp>

#include <polyfem/utils/RefElementSampler.hpp>

#include <Eigen/Dense>

namespace polyfem
{
	class State;
}

namespace polyfem::output
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

	/// Utilies related to export of geometry
	class OutGeometryData
	{
	public:
		/// @brief different export flags
		struct ExportOptions
		{
			bool volume;
			bool surface;
			bool wire;
			bool contact_forces;
			bool friction_forces;

			bool use_sampler;
			bool boundary_only;
			bool material_params;
			bool body_ids;
			bool sol_on_grid;
			bool velocity;
			bool acceleration;

			bool use_spline;
			bool reorder_output;

			bool solve_export_to_file;

			/// @brief initialize the flags based on the input args
			/// @param args input arguments used to set most of the flags
			/// @param is_mesh_linear if the mesh is linear
			/// @param is_problem_scalar if the problem is scalar
			/// @param solve_export_to_file if export to file or save in the frames
			ExportOptions(const json &args,
						  const bool is_mesh_linear,
						  const bool is_problem_scalar,
						  const bool solve_export_to_file);
		};

		/// extracts the boundary mesh
		/// @param[in] mesh mesh
		/// @param[in] n_bases number of bases
		/// @param[in] bases bases
		/// @param[in] total_local_boundary mesh boundaries
		/// @param[out] boundary_nodes_pos nodes positions
		/// @param[out] boundary_edges edges
		/// @param[out] boundary_triangles triangles
		static void extract_boundary_mesh(
			const mesh::Mesh &mesh,
			const int n_bases,
			const std::vector<basis::ElementBases> &bases,
			const std::vector<mesh::LocalBoundary> &total_local_boundary,
			Eigen::MatrixXd &boundary_nodes_pos,
			Eigen::MatrixXi &boundary_edges,
			Eigen::MatrixXi &boundary_triangles);

		/// @brief unitalize the ref element sampler
		/// @param mesh mesh
		/// @param vismesh_rel_area relative sampling size
		void init_sampler(const polyfem::mesh::Mesh &mesh, const double vismesh_rel_area);

		/// @brief builds the grid to export the solution
		/// @param mesh mesh
		/// @param spacing grid spacing, <=0 mean no grid
		void build_grid(const polyfem::mesh::Mesh &mesh, const double spacing);

		void export_data(
			const State &state,
			const bool is_time_dependent,
			const double tend_in,
			const double dt,
			const ExportOptions &opts,
			const std::string &vis_mesh_path,
			const std::string &nodes_path,
			const std::string &solution_path,
			const std::string &stress_path,
			const std::string &mises_path,
			const bool is_contact_enabled,
			std::vector<output::SolutionFrame> &solution_frames) const;

		/// saves the vtu file for time t
		/// @param[in] name filename
		/// @param[in] t time
		void save_vtu(const std::string &path,
					  const State &state,
					  const double t,
					  const double dt,
					  const ExportOptions &opts,
					  const bool is_contact_enabled,
					  std::vector<output::SolutionFrame> &solution_frames) const;

		/// saves the volume vtu file
		/// @param[in] name filename
		/// @param[in] t time
		void save_volume(const std::string &path,
						 const State &state,
						 const double t,
						 const ExportOptions &opts,
						 std::vector<output::SolutionFrame> &solution_frames) const;

		/// saves the surface vtu file for for surface quantites, eg traction forces
		/// @param[in] name filename
		/// @param[in] t time
		void save_surface(const std::string &export_surface,
						  const State &state,
						  const double dt_in,
						  const ExportOptions &opts,
						  const bool is_contact_enabled,
						  std::vector<output::SolutionFrame> &solution_frames) const;

		/// saves the wireframe
		/// @param[in] name filename
		/// @param[in] t time
		void save_wire(const std::string &name,
					   const State &state,
					   const double t,
					   const ExportOptions &opts,
					   std::vector<output::SolutionFrame> &solution_frames) const;

		/// save a PVD of a time dependent simulation
		/// @param[in] name filename
		/// @param[in] vtu_names names of the vtu files
		/// @param[in] time_steps total time stesp
		/// @param[in] t0 initial time
		/// @param[in] dt delta t
		/// @param[in] skip_frame every which frame to skip
		void save_pvd(const std::string &name, const std::function<std::string(int)> &vtu_names,
					  int time_steps, double t0, double dt, int skip_frame = 1) const;

	private:
		/// used to sample the solution
		utils::RefElementSampler ref_element_sampler;

		/// grid mesh points to export solution sampled on a grid
		Eigen::MatrixXd grid_points;
		/// grid mesh mapping to fe elements
		Eigen::MatrixXi grid_points_to_elements;
		/// grid mesh boundaries
		Eigen::MatrixXd grid_points_bc;

		/// builds the boundary mesh for visualization, called in build_basis
		/// boundary visualization mesh vertices
		/// boundary visualization mesh vertices pre image in ref element
		/// boundary visualization mesh connectivity
		/// boundary visualization mesh elements ids
		/// boundary visualization mesh edge/face id
		/// boundary visualization mesh normals
		void build_vis_boundary_mesh(
			const mesh::Mesh &mesh,
			const std::vector<basis::ElementBases> &bases,
			const std::vector<basis::ElementBases> &gbases,
			const std::vector<mesh::LocalBoundary> &total_local_boundary,
			Eigen::MatrixXd &boundary_vis_vertices,
			Eigen::MatrixXd &boundary_vis_local_vertices,
			Eigen::MatrixXi &boundary_vis_elements,
			Eigen::MatrixXi &boundary_vis_elements_ids,
			Eigen::MatrixXi &boundary_vis_primitive_ids,
			Eigen::MatrixXd &boundary_vis_normals) const;

		/// builds visualzation mesh, upsampled mesh used for visualization
		/// the visualization mesh is a dense mesh per element all disconnected
		/// it also retuns the mapping to element id and discretization of every elment
		/// works in 2 and 3d. if the mesh is not simplicial it gets tri/tet halized
		/// @param[out] points mesh points
		/// @param[out] tets mesh cells
		/// @param[out] el_id mapping from points to elements id
		/// @param[out] discr mapping from points to discretization order
		void build_vis_mesh(
			const mesh::Mesh &mesh,
			const Eigen::VectorXi &disc_orders,
			const std::vector<basis::ElementBases> &gbases,
			const std::map<int, Eigen::MatrixXd> &polys,
			const std::map<int, std::pair<Eigen::MatrixXd, Eigen::MatrixXi>> &polys_3d,
			const bool boundary_only,
			Eigen::MatrixXd &points,
			Eigen::MatrixXi &tets,
			Eigen::MatrixXi &el_id,
			Eigen::MatrixXd &discr) const;

		/// builds high-der visualzation mesh per element all disconnected
		/// it also retuns the mapping to element id and discretization of every elment
		/// works in 2 and 3d. if the mesh is not simplicial it gets tri/tet halized
		/// @param[out] points mesh points
		/// @param[out] elements mesh high-order cells
		/// @param[out] el_id mapping from points to elements id
		/// @param[out] discr mapping from points to discretization order
		void build_high_oder_vis_mesh(
			const mesh::Mesh &mesh,
			const Eigen::VectorXi &disc_orders,
			const std::vector<basis::ElementBases> &bases,
			Eigen::MatrixXd &points,
			std::vector<std::vector<int>> &elements,
			Eigen::MatrixXi &el_id,
			Eigen::MatrixXd &discr) const;
	};

	class OutRuntimeData
	{
	public:
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

		double total_time()
		{
			return building_basis_time + assembling_stiffness_mat_time + solving_time;
		}
	};

	class OutStatsData
	{
	public:
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

		/// compute the errors, not part of solve
		void compute_errors(const int n_bases,
							const std::vector<polyfem::basis::ElementBases> &bases,
							const std::vector<polyfem::basis::ElementBases> &gbases,
							const polyfem::mesh::Mesh &mesh,
							const assembler::Problem &problem,
							const double tend,
							const Eigen::MatrixXd &sol);

		/// compute stats (counts els type, mesh lenght, etc), step 1 of solve
		void compute_mesh_stats(const polyfem::mesh::Mesh &mesh);

		/// computes the mesh size, it samples every edges n_samples times
		/// uses curved_mesh_size (false by default) to compute the size of
		/// the linear mesh
		/// @param[in] mesh to compute stats
		/// @param[in] bases geom bases
		/// @param[in] n_samples used for curved meshes
		void compute_mesh_size(const polyfem::mesh::Mesh &mesh_in, const std::vector<polyfem::basis::ElementBases> &bases_in, const int n_samples, const bool use_curved_mesh_size);

		void reset();

		void count_flipped_elements(const polyfem::mesh::Mesh &mesh, const std::vector<polyfem::basis::ElementBases> &gbases);

		/// saves the output statistic to a json object
		/// @param[in] j output json
		void save_json(const nlohmann::json &args,
					   const int n_bases, const int n_pressure_bases,
					   const Eigen::MatrixXd &sol,
					   const mesh::Mesh &mesh,
					   const Eigen::VectorXi &disc_orders,
					   const assembler::Problem &problem,
					   const output::OutRuntimeData &runtime,
					   const std::string &formulation,
					   const bool isoparametric,
					   const int sol_at_node_id,
					   nlohmann::json &j);
	};
} // namespace polyfem::output