#pragma once

#include <polyfem/Common.hpp>
#include <polyfem/Units.hpp>
#include <polyfem/io/ResourceIO.hpp>
#include <polyfem/mesh/Mesh.hpp>
#include <polyfem/mesh/Obstacle.hpp>

namespace polyfem::mesh
{
	class LoadedGeometry
	{
	public:
		std::unique_ptr<Mesh> fem;
		Obstacle obstacle;
	};

	/// Applies geometry JSON semantics while delegating all decoding to MeshLoader.
	class GeometryLoader
	{
	public:
		GeometryLoader(Units units, const io::ResourceIO &resources)
			: units_(std::move(units)), resources_(resources) {}

		LoadedGeometry load(
			const json &geometry,
			const std::vector<json> &obstacle_displacements,
			const std::vector<json> &dirichlet_conditions,
			bool non_conforming = false) const;

		Obstacle load_obstacles(
			const json &geometry,
			const std::vector<json> &obstacle_displacements,
			const std::vector<json> &dirichlet_conditions,
			int dimension,
			bool non_conforming = false) const;

	private:
		Units units_;
		const io::ResourceIO &resources_;
	};
} // namespace polyfem::mesh
