#pragma once

#include <polyfem/Common.hpp>

#include <polyfem/basis/ElementBases.hpp>

#include <polyfem/mesh/Mesh.hpp>

#include <polyfem/io/OutputData.hpp>
#include <polyfem/io/OutStatsData.hpp>

#include <paraviewo/ParaviewWriter.hpp>
#include <paraviewo/VTUWriter.hpp>
#include <paraviewo/HDF5VTUWriter.hpp>

#include <polyfem/utils/RefElementSampler.hpp>

#include <Eigen/Dense>

namespace polyfem::io
{
	/// Utilies related to export of geometry
	class OutGeometryData
	{
	public:
		/// @brief different export flags
		struct ExportOptions
		{
			std::vector<std::string> fields; // fields to export, empty means all

			bool volume;
			bool surface;
			bool wire;
			bool points;
			bool contact_forces;
			bool friction_forces;
			bool normal_adhesion_forces;
			bool tangential_adhesion_forces;

			bool use_sampler;
			bool boundary_only;
			bool sol_on_grid;
			bool discretization_order;
			bool reorder_output;

			bool use_hdf5;

			/// @brief initialize the flags based on the input args
			/// @param[in] args input arguments used to set most of the flags
			/// @param[in] is_mesh_linear if the mesh is linear
			/// @param[in] mesh_has_prisms if the mesh has prisms
			/// @param[in] is_problem_scalar if the problem is scalar
			ExportOptions(const json &args,
						  const bool is_mesh_linear,
						  const bool mesh_has_prisms,
						  const bool is_problem_scalar);

			/// @brief return the extension of the output paraview files depending on use_hdf5
			/// @return either hdf or vtu
			inline std::string file_extension() const
			{
				return use_hdf5 ? ".hdf" : ".vtu";
			}

			bool export_field(const std::string &field) const;
		};

		/// extracts the boundary mesh
		/// @param[in] mesh mesh
		/// @param[in] n_bases number of bases
		/// @param[in] bases bases
		/// @param[in] total_local_boundary mesh boundaries
		/// @param[out] node_positions nodes positions
		/// @param[out] boundary_edges edges
		/// @param[out] boundary_triangles triangles
		/// @param[out] displacement_map map of collision mesh vertices to nodes, empty if identity
		static void extract_boundary_mesh(
			const mesh::Mesh &mesh,
			const int n_bases,
			const std::vector<basis::ElementBases> &bases,
			const std::vector<mesh::LocalBoundary> &total_local_boundary,
			Eigen::MatrixXd &node_positions,
			Eigen::MatrixXi &boundary_edges,
			Eigen::MatrixXi &boundary_triangles,
			std::vector<Eigen::Triplet<double>> &displacement_map_entries);

		/// @brief unitalize the ref element sampler
		/// @param[in] mesh mesh
		/// @param[in] vismesh_rel_area relative sampling size
		void init_sampler(const polyfem::mesh::Mesh &mesh, const double vismesh_rel_area);

		/// @brief builds the grid to export the solution
		/// @param[in] mesh mesh
		/// @param[in] spacing grid spacing, <=0 mean no grid
		void build_grid(const polyfem::mesh::Mesh &mesh, const double spacing);

		/// @brief exports everytihng, txt, vtu, etc
		/// @param[in] space output geometry data
		/// @param[in] output_fields callback appending physics-specific fields
		/// @param[in] is_time_dependent if the sim is time dependent
		/// @param[in] tend_in end time
		/// @param[in] dt delta t
		/// @param[in] opts export options
		/// @param[in] vis_mesh_path vtu path
		void export_data(
			const OutputSpace &space,
			const OutputFieldFunction &output_fields,
			const bool is_time_dependent,
			const double tend_in,
			const double dt,
			const ExportOptions &opts,
			const std::string &vis_mesh_path) const;

		/// saves the vtu file for time t
		/// @param[in] path filename
		/// @param[in] space output geometry data
		/// @param[in] output_fields callback appending physics-specific fields
		/// @param[in] t time
		/// @param[in] dt delta t
		/// @param[in] opts export options
		void save_vtu(const std::string &path,
					  const OutputSpace &space,
					  const OutputFieldFunction &output_fields,
					  const double t,
					  const double dt,
					  const ExportOptions &opts) const;

		/// saves the volume vtu file
		/// @param[in] path filename
		/// @param[in] space output geometry data
		/// @param[in] output_fields callback appending physics-specific fields
		/// @param[in] t time
		/// @param[in] dt delta t
		/// @param[in] opts export options
		void save_volume(const std::string &path,
						 const OutputSpace &space,
						 const OutputFieldFunction &output_fields,
						 const double t,
						 const double dt,
						 const ExportOptions &opts) const;

		/// saves the surface vtu file for for surface quantites, eg traction forces
		/// @param[in] export_surface filename
		/// @param[in] space output geometry data
		/// @param[in] output_fields callback appending physics-specific fields
		/// @param[in] t time
		/// @param[in] dt_in delta_t
		/// @param[in] opts export options
		void save_surface(const std::string &export_surface,
						  const OutputSpace &space,
						  const OutputFieldFunction &output_fields,
						  const double t,
						  const double dt_in,
						  const ExportOptions &opts) const;

		/// saves the  surface vtu file for for constact quantites, eg contact or friction forces
		/// @param[in] export_surface filename
		/// @param[in] space output geometry data
		/// @param[in] output_fields callback appending physics-specific fields
		/// @param[in] t time
		/// @param[in] dt_in delta_t
		/// @param[in] opts export options
		void save_contact_surface(
			const std::string &export_surface,
			const OutputSpace &space,
			const OutputFieldFunction &output_fields,
			const double t,
			const double dt_in,
			const ExportOptions &opts) const;

		/// saves the wireframe
		/// @param[in] name filename
		/// @param[in] space output geometry data
		/// @param[in] output_fields callback appending physics-specific fields
		/// @param[in] t time
		/// @param[in] opts export options
		void save_wire(const std::string &name,
					   const OutputSpace &space,
					   const OutputFieldFunction &output_fields,
					   const double t,
					   const ExportOptions &opts) const;

		/// saves the nodal values
		/// @param[in] path filename
		/// @param[in] space output geometry data
		/// @param[in] output_fields callback appending physics-specific fields
		/// @param[in] opts export options
		void save_points(
			const std::string &path,
			const OutputSpace &space,
			const OutputFieldFunction &output_fields,
			const ExportOptions &opts) const;

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

		/// @brief builds the boundary mesh for visualization
		/// @param[in] mesh mesh
		/// @param[in] gbases geometric bases
		/// @param[in] total_local_boundary boundaries
		/// @param[out] boundary_vis_vertices boundary visualization mesh vertices
		/// @param[out] boundary_vis_local_vertices boundary visualization mesh vertices pre image in ref element
		/// @param[out] boundary_vis_elements boundary visualization mesh connectivity
		/// @param[out] boundary_vis_elements_ids boundary visualization mesh elements ids
		/// @param[out] boundary_vis_primitive_ids boundary visualization mesh edge/face id
		/// @param[out] boundary_vis_normals boundary visualization mesh normals
		void build_vis_boundary_mesh(
			const mesh::Mesh &mesh,
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
		/// @param[in] mesh mesh
		/// @param[in] disc_orders discretization orders
		/// @param[in] gbases geometric bases
		/// @param[in] polys polygons
		/// @param[in] polys_3d polyhedra
		/// @param[in] boundary_only is build only elements touching the boundary
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
			Eigen::MatrixXd &discr,
			Eigen::MatrixXd &local_points) const;

		/// builds high-der visualzation mesh per element all disconnected
		/// it also retuns the mapping to element id and discretization of every elment
		/// works in 2 and 3d. if the mesh is not simplicial it gets tri/tet halized
		/// @param[in] mesh mesh
		/// @param[in] output_orders output cell order per element
		/// @param[in] bases bases used to map output reference nodes
		/// @param[out] points mesh points
		/// @param[out] elements mesh high-order cells
		/// @param[out] el_id mapping from points to elements id
		/// @param[out] discr mapping from points to discretization order
		void build_high_order_vis_mesh(
			const mesh::Mesh &mesh,
			const Eigen::VectorXi &output_orders,
			const std::vector<basis::ElementBases> &bases,
			Eigen::MatrixXd &points,
			std::vector<paraviewo::CellElement> &elements,
			Eigen::MatrixXi &el_id,
			Eigen::MatrixXd &discr,
			Eigen::MatrixXd &local_points) const;
	};

} // namespace polyfem::io
