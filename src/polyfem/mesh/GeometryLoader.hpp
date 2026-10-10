#pragma once

#include <polyfem/Common.hpp>
#include <polyfem/Units.hpp>
#include <polyfem/io/ResourceIO.hpp>
#include <polyfem/mesh/Mesh.hpp>
#include <polyfem/mesh/MeshLoader.hpp>
#include <polyfem/mesh/Obstacle.hpp>
#include <polyfem/utils/Types.hpp>

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

		/// Load and combine the enabled FEM entries in a geometry configuration.
		std::unique_ptr<Mesh> load_fem(
			const json &geometry,
			bool non_conforming = false) const;
		std::unique_ptr<Mesh> load_fem_entry(
			const json &geometry,
			bool non_conforming = false) const;

		/// Load and prepare one configured surface/codimensional geometry entry.
		SurfaceMesh load_surface(
			const json &geometry,
			int dimension) const;

		Obstacle load_obstacles(
			const json &geometry,
			const std::vector<json> &obstacle_displacements,
			const std::vector<json> &dirichlet_conditions,
			int dimension,
			bool non_conforming = false) const;

		/// Apply configured geometry IDs to an already constructed mesh.
		void apply_geometry_selection(
			Mesh &mesh,
			const json &geometry_selection) const;

		/// Construct the affine map used by geometry and collision-proxy inputs.
		static void construct_affine_transformation(
			double unit_scale,
			const json &transform,
			const VectorNd &mesh_dimensions,
			MatrixNd &A,
			VectorNd &b);

	private:
		Units units_;
		const io::ResourceIO &resources_;
	};
} // namespace polyfem::mesh
