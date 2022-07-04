#pragma once

#include <polyfem/Common.hpp>
#include <polyfem/utils/Types.hpp>
#include <polyfem/mesh/Mesh.hpp>
#include <polyfem/mesh/Obstacle.hpp>
#include <polyfem/utils/Selection.hpp>

namespace polyfem::mesh
{
	///
	/// @brief      read FEM meshes from a geometry JSON array (or single)
	///
	/// @param[in]  geometry        geometry JSON object(s)
	/// @param[in]  root_path       root path of JSON
	///
	/// @return created Mesh object
	///
	std::unique_ptr<Mesh> read_fem_geometry(
		const json &geometry,
		const std::string &root_path,
		const std::vector<std::string> &names = std::vector<std::string>(),
		const std::vector<Eigen::MatrixXd> &vertices = std::vector<Eigen::MatrixXd>(),
		const std::vector<Eigen::MatrixXi> &cells = std::vector<Eigen::MatrixXi>(),
		const bool non_conforming = false);

	///
	/// @brief      read a FEM mesh from a geometry JSON
	///
	/// @param[in]  geometry        geometry JSON object(s)
	/// @param[in]  displacements   displacements JSON object(s)
	/// @param[in]  root_path       root path of JSON
	///
	/// @return created Obstacle object
	///
	Obstacle read_obstacle_geometry(
		const json &geometry,
		const std::vector<json> &displacements,
		const std::string &root_path,
		const int dim,
		const std::vector<std::string> &names = std::vector<std::string>(),
		const std::vector<Eigen::MatrixXd> &vertices = std::vector<Eigen::MatrixXd>(),
		const std::vector<Eigen::MatrixXi> &cells = std::vector<Eigen::MatrixXi>(),
		const bool non_conforming = false);

	///
	/// @brief      read a FEM mesh from a geometry JSON
	///
	/// @param[in]  jmesh           geometry JSON
	/// @param[in]  root_path       root path of JSON
	/// @param[out] vertices        #V x 3/2 output vertices positions
	/// @param[out] codim_vertices  indicies in vertices for the codimensional vertices
	/// @param[out] codim_edges     indicies in vertices for the codimensional edges
	/// @param[out] faces           indicies in vertices for the surface faces
	///
	void read_fem_mesh(
		const json &jmesh,
		const std::string &root_path,
		Eigen::MatrixXd &vertices,
		Eigen::MatrixXi &cells,
		std::vector<std::vector<int>> &elements,
		std::vector<std::vector<double>> &weights,
		size_t &num_faces,
		std::vector<std::shared_ptr<polyfem::utils::Selection>> &surface_selections,
		std::vector<std::shared_ptr<polyfem::utils::Selection>> &volume_selections);

	///
	/// @brief      read a obstacle mesh from a geometry JSON
	///
	/// @param[in]  jmesh           geometry JSON
	/// @param[in]  root_path       root path of JSON
	/// @param[out] vertices        #V x 3/2 output vertices positions
	/// @param[out] codim_vertices  indicies in vertices for the codimensional vertices
	/// @param[out] codim_edges     indicies in vertices for the codimensional edges
	/// @param[out] faces           indicies in vertices for the surface faces
	///
	void read_obstacle_mesh(
		const json &jmesh,
		const std::string &root_path,
		Eigen::MatrixXd &vertices,
		Eigen::VectorXi &codim_vertices,
		Eigen::MatrixXi &codim_edges,
		Eigen::MatrixXi &faces);

	///
	/// @brief      Fill in missing json geometry parameters with the default values
	///
	/// @param[in]  geometry_in   input json geometry parameters
	/// @param[out] geometry_out  output json geometry parameters
	///
	void apply_default_geometry_parameters(
		const json &geometry_in, json &geometry_out, const std::string &path_prefix = "");

	///
	/// @brief         Transform a mesh inplace using json parameters including scaling, rotation, and translation
	///
	/// @param[in]     transform json object with the mesh data
	/// @param[in,out] vertices  #V x 3/2 input and output vertices positions
	///
	void transform_mesh_from_json(const json &transform, Eigen::MatrixXd &vertices);

	void append_selections(
		const std::string &root_path,
		const json &new_selections,
		const polyfem::utils::Selection::BBox &bbox,
		const size_t &start_element_id,
		const size_t &end_element_id,
		std::vector<std::shared_ptr<polyfem::utils::Selection>> &selections);

} // namespace polyfem::mesh
