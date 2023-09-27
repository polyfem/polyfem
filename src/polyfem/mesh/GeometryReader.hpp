#pragma once

#include <polyfem/Common.hpp>
#include <polyfem/mesh/Mesh.hpp>
#include <polyfem/mesh/Obstacle.hpp>
#include <polyfem/utils/Selection.hpp>
#include <polyfem/utils/Types.hpp>

namespace polyfem::mesh
{
	///
	/// @brief      read a FEM mesh from a geometry JSON
	///
	/// @param[in]  j_mesh           geometry JSON
	/// @param[in]  root_path       root path of JSON
	/// @param[in]  non_conforming  if true, the mesh will be non-conforming
	///
	/// @return created Mesh object
	///
	std::unique_ptr<Mesh> read_fem_mesh(
		const Units &units,
		const json &j_mesh,
		const std::string &root_path,
		const bool non_conforming = false);

	///
	/// @brief      read FEM meshes from a geometry JSON array (or single)
	///
	/// @param[in]  geometry        geometry JSON object(s)
	/// @param[in]  root_path       root path of JSON
	///
	/// @return created Mesh object
	///
	std::unique_ptr<Mesh> read_fem_geometry(
		const Units &units,
		const json &geometry,
		const std::string &root_path,
		const std::vector<std::string> &names = std::vector<std::string>(),
		const std::vector<Eigen::MatrixXd> &vertices = std::vector<Eigen::MatrixXd>(),
		const std::vector<Eigen::MatrixXi> &cells = std::vector<Eigen::MatrixXi>(),
		const bool non_conforming = false);

	///
	/// @brief      read a obstacle mesh from a geometry JSON
	///
	/// @param[in]  j_mesh           geometry JSON
	/// @param[in]  root_path       root path of JSON
	/// @param[out] vertices        #V x 3/2 output vertices positions
	/// @param[out] codim_vertices  indicies in vertices for the codimensional vertices
	/// @param[out] codim_edges     indicies in vertices for the codimensional edges
	/// @param[out] faces           indicies in vertices for the surface faces
	///
	void read_obstacle_mesh(
		const Units &units,
		const json &j_mesh,
		const std::string &root_path,
		const int dim,
		Eigen::MatrixXd &vertices,
		Eigen::VectorXi &codim_vertices,
		Eigen::MatrixXi &codim_edges,
		Eigen::MatrixXi &faces);

	///
	/// @brief      read a FEM mesh from a geometry JSON
	///
	/// @param[in]  geometry        geometry JSON object(s)
	/// @param[in]  displacements   displacements JSON object(s)
	/// @param[in]  dirichlets    	dirichlet bc JSON object(s)
	/// @param[in]  root_path       root path of JSON
	///
	/// @return created Obstacle object
	///
	Obstacle read_obstacle_geometry(
		const Units &units,
		const json &geometry,
		const std::vector<json> &displacements,
		const std::vector<json> &dirichlets,
		const std::string &root_path,
		const int dim,
		const std::vector<std::string> &names = std::vector<std::string>(),
		const std::vector<Eigen::MatrixXd> &vertices = std::vector<Eigen::MatrixXd>(),
		const std::vector<Eigen::MatrixXi> &cells = std::vector<Eigen::MatrixXi>(),
		const bool non_conforming = false);

	///
	/// @brief Construct an affine transformation \f$Ax+b\f$.
	///
	/// @param[in]  transform       JSON object with the mesh data
	/// @param[in]  mesh_dimensions Dimensions of the mesh (i.e., width, height, depth)
	/// @param[out] A               Multiplicative matrix component of transformation
	/// @param[out] b               Additive translation component of transformation
	///
	void construct_affine_transformation(
		const double unit_scale,
		const json &transform,
		const VectorNd &mesh_dimensions,
		MatrixNd &A,
		VectorNd &b);

} // namespace polyfem::mesh
