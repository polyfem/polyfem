#pragma once

#include <polyfem/Common.hpp>
#include <polyfem/Types.hpp>

namespace polyfem
{
	///
	/// @brief      Compute the dimension of the mesh's bounding box
	///
	/// @param[in]  vertices { #V x 3/2 vertex positions }
	///
	/// @return N vector of the mesh's dimensions
	///
	VectorNd mesh_dimensions(const Eigen::MatrixXd &vertices);

	///
	/// @brief      Fill in missing json mesh parameters with the default values
	///
	/// @param[in]  mesh_in  { input json mesh parameters }
	/// @param[out] mesh_out { output json mesh parameters }
	///
	void apply_default_mesh_parameters(const json &mesh_in, json &mesh_out, const std::string &path_prefix = "");

	///
	/// @brief      Construct a mesh transformation from json parameters including scaling, rotation, and translation
	///
	/// @param[in]  mesh               { json object with the mesh data }
	/// @param[in]  initial_dimensions { initial dimensions of the mesh }
	/// @param[out] affine_transform   { N×N matrix containg the affine portion of the transform }
	/// @param[out] translation        { N row vector containing the translation portion of the transform }
	///
	void mesh_transform_from_json(const json &mesh, const VectorNd &initial_dimensions, MatrixNd &affine_transform, RowVectorNd &translation);

	///
	/// @brief         Transform a mesh inplace using the given transformation
	///
	/// @param[in]     affine_transform { N×N matrix containg the affine portion of the transform }
	/// @param[in]     translation      { N row vector containing the translation portion of the transform }
	/// @param[in,out] vertices         { #V x 3/2 input and output vertices positions }
	///
	void transform_mesh(const MatrixNd &affine_transform, const RowVectorNd &translation, Eigen::MatrixXd &vertices);

	///
	/// @brief         Transform a mesh inplace using json parameters including scaling, rotation, and translation
	///
	/// @param[in]     mesh     { json object with the mesh data }
	/// @param[in,out] vertices { #V x 3/2 input and output vertices positions }
	///
	void transform_mesh_from_json(const json &mesh, Eigen::MatrixXd &vertices);

	struct MeshParams
	{
		Eigen::MatrixXd vertices;
		Eigen::MatrixXi cells;
		std::vector<std::vector<int>> elements;
		std::vector<std::vector<double>> weights;
		std::vector<int> body_vertices_start;
		std::vector<int> body_faces_start;
		std::vector<int> body_ids;
		std::vector<int> boundary_ids;
		std::vector<std::string> bc_tag_paths;
	};

	void create_from_json(
		const json &jmesh,
		const std::string &root_path,
		MeshParams &mesh);

} // namespace polyfem
