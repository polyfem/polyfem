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
} // namespace polyfem::mesh