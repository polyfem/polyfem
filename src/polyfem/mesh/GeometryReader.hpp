#pragma once

#include <polyfem/Common.hpp>
#include <polyfem/utils/Types.hpp>
#include <polyfem/mesh/Mesh.hpp>
#include <polyfem/mesh/Obstacle.hpp>
#include <polyfem/utils/Selection.hpp>

namespace polyfem::mesh
{
	void read_fem_meshes(
		const json &geometry,
		const std::string &root_path,
		std::unique_ptr<Mesh> &mesh,
		const std::vector<std::string> &names = std::vector<std::string>(),
		const std::vector<Eigen::MatrixXd> &vertices = std::vector<Eigen::MatrixXd>(),
		const std::vector<Eigen::MatrixXi> &cells = std::vector<Eigen::MatrixXi>(),
		const bool non_conforming = false);

	void read_obstacles(
		const json &geometry,
		const std::string &root_path,
		Obstacle &obstacle,
		const std::vector<std::string> &names = std::vector<std::string>(),
		const std::vector<Eigen::MatrixXd> &vertices = std::vector<Eigen::MatrixXd>(),
		const std::vector<Eigen::MatrixXi> &cells = std::vector<Eigen::MatrixXi>(),
		const bool non_conforming = false);

	void load_mesh(
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
	/// @brief      Fill in missing json geometry parameters with the default values
	///
	/// @param[in]  geometry_in  { input json geometry parameters }
	/// @param[out] geometry_out { output json geometry parameters }
	///
	void apply_default_geometry_parameters(
		const json &geometry_in, json &geometry_out, const std::string &path_prefix = "");

	///
	/// @brief         Transform a mesh inplace using json parameters including scaling, rotation, and translation
	///
	/// @param[in]     transform { json object with the mesh data }
	/// @param[in,out] vertices  { #V x 3/2 input and output vertices positions }
	///
	void transform_mesh_from_json(const json &transform, Eigen::MatrixXd &vertices);

	void append_selections(
		const json &new_selections,
		const polyfem::utils::Selection::BBox &bbox,
		const size_t &start_element_id,
		const size_t &end_element_id,
		std::vector<std::shared_ptr<polyfem::utils::Selection>> &selections);

} // namespace polyfem::mesh
