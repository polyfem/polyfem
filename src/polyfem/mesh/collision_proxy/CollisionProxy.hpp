#pragma once

#include <polyfem/mesh/Mesh.hpp>
#include <polyfem/mesh/LocalBoundary.hpp>
#include <polyfem/basis/ElementBases.hpp>
#include <polyfem/utils/Types.hpp>

#include <Eigen/Core>

namespace polyfem::mesh
{
	enum class CollisionProxyTessellation
	{
		REGULAR,  ///< @brief Regular tessellation of the mesh
		IRREGULAR ///< @brief Irregular tessellation of the mesh (requires POLYFEM_WITH_TRIANGLE)
	};

	NLOHMANN_JSON_SERIALIZE_ENUM(
		CollisionProxyTessellation,
		{{CollisionProxyTessellation::REGULAR, "regular"},
		 {CollisionProxyTessellation::IRREGULAR, "irregular"}});

	/// @brief Build a collision proxy mesh by upsampling a given mesh.
	/// @param[in] bases Bases for elements
	/// @param[in] geom_bases Geometry bases for elements
	/// @param[in] total_local_boundary Local boundaries for elements
	/// @param[in] n_bases Number of bases (nodes)
	/// @param[in] dim Dimension of the mesh
	/// @param[in] max_edge_length Maximum edge length of the proxy mesh
	/// @param[out] proxy_vertices Output vertices of the proxy mesh
	/// @param[out] proxy_faces Output faces of the proxy mesh
	/// @param[out] displacement_map Output displacement map from proxy mesh to original mesh
	/// @param[in] tessellation Type of tessellation to use
	void build_collision_proxy(
		const std::vector<basis::ElementBases> &bases,
		const std::vector<basis::ElementBases> &geom_bases,
		const std::vector<mesh::LocalBoundary> &total_local_boundary,
		const int n_bases,
		const int dim,
		const double max_edge_length,
		Eigen::MatrixXd &proxy_vertices,
		Eigen::MatrixXi &proxy_faces,
		std::vector<Eigen::Triplet<double>> &displacement_map,
		const CollisionProxyTessellation tessellation = CollisionProxyTessellation::REGULAR);

	/// @brief Build a collision proxy displacement map for a given mesh and proxy mesh.
	/// @param[in] bases Bases for elements
	/// @param[in] geom_bases Geometry bases for elements
	/// @param[in] total_local_boundary Local boundaries for elements
	/// @param[in] n_bases Number of bases (nodes)
	/// @param[in] dim Dimension of the mesh
	/// @param[in] proxy_vertices Vertices of the proxy mesh
	/// @param[out] displacement_map Output displacement map from proxy mesh to original mesh
	void build_collision_proxy_displacement_maps(
		const std::vector<basis::ElementBases> &bases,
		const std::vector<basis::ElementBases> &geom_bases,
		const std::vector<mesh::LocalBoundary> &total_local_boundary,
		const int n_bases,
		const int dim,
		const Eigen::MatrixXd &proxy_vertices,
		// NOTE: no need for proxy_faces
		std::vector<Eigen::Triplet<double>> &displacement_map);

	/// @brief Load a collision proxy mesh and displacement map from files.
	/// @param[in] mesh_filename Mesh filename
	/// @param[in] weights_filename Weights filename
	/// @param[in] in_node_to_node Map from input node IDs to node IDs
	/// @param[in] transformation Transformation to apply to the mesh
	/// @param[out] vertices Output vertices of the proxy mesh
	/// @param[out] codim_vertices Output codimension vertices of the proxy mesh
	/// @param[out] edges Output edges of the proxy mesh
	/// @param[out] faces Output faces of the proxy mesh
	/// @param[out] displacement_map_entries Output displacement map entries
	void load_collision_proxy(
		const std::string &mesh_filename,
		const std::string &weights_filename,
		const Eigen::VectorXi &in_node_to_node,
		const json &transformation,
		Eigen::MatrixXd &vertices,
		Eigen::VectorXi &codim_vertices,
		Eigen::MatrixXi &edges,
		Eigen::MatrixXi &faces,
		std::vector<Eigen::Triplet<double>> &displacement_map_entries);

	/// @brief Load a collision proxy mesh from a file.
	/// @param[in] mesh_filename Mesh filename
	/// @param[in] transformation Transformation to apply to the mesh
	/// @param[out] vertices Output vertices of the proxy mesh
	/// @param[out] codim_vertices Output codimension vertices of the proxy mesh
	/// @param[out] edges Output edges of the proxy mesh
	/// @param[out] faces Output faces of the proxy mesh
	void load_collision_proxy_mesh(
		const std::string &mesh_filename,
		const json &transformation,
		Eigen::MatrixXd &vertices,
		Eigen::VectorXi &codim_vertices,
		Eigen::MatrixXi &edges,
		Eigen::MatrixXi &faces);

	/// @brief Load a collision proxy displacement map from files.
	/// @param[in] weights_filename Weights filename
	/// @param[in] in_node_to_node Map from input node IDs to node IDs
	/// @param[out] displacement_map_entries Output displacement map entries
	void load_collision_proxy_displacement_map(
		const std::string &weights_filename,
		const Eigen::VectorXi &in_node_to_node,
		const size_t num_proxy_vertices,
		std::vector<Eigen::Triplet<double>> &displacement_map_entries);
} // namespace polyfem::mesh