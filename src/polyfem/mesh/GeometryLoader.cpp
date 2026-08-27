#include "GeometryLoader.hpp"

#include <polyfem/mesh/GeometryReader.hpp>
#include <polyfem/utils/Logger.hpp>

namespace polyfem::mesh
{
	LoadedGeometry GeometryLoader::load(
		const json &geometry,
		const std::vector<json> &obstacle_displacements,
		const std::vector<json> &dirichlet_conditions,
		const bool non_conforming) const
	{
		LoadedGeometry result;
		result.fem = read_fem_geometry(units_, geometry, resources_, non_conforming);
		if (!result.fem)
			log_and_throw_error("Configured geometry contains no enabled FEM mesh.");
		result.obstacle = load_obstacles(
			geometry, obstacle_displacements, dirichlet_conditions,
			result.fem->dimension(), non_conforming);
		return result;
	}

	Obstacle GeometryLoader::load_obstacles(
		const json &geometry,
		const std::vector<json> &obstacle_displacements,
		const std::vector<json> &dirichlet_conditions,
		const int dimension,
		const bool non_conforming) const
	{
		return read_obstacle_geometry(
			units_, geometry, obstacle_displacements, dirichlet_conditions,
			resources_, dimension, non_conforming);
	}
} // namespace polyfem::mesh
