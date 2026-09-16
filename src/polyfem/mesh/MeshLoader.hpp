#pragma once

#include <polyfem/io/ResourceIO.hpp>
#include <polyfem/mesh/Mesh.hpp>
#include <polyfem/mesh/MeshReader.hpp>

namespace polyfem::mesh
{
	/// The canonical typed mesh schema shared by HDF5 bundles and checkpoints.
	inline constexpr long MESH_SCHEMA_VERSION = 1;

	using SurfaceMesh = SurfaceMeshData;

	/// The only non-legacy decoder from a logical resource path to mesh data.
	class MeshLoader
	{
	public:
		explicit MeshLoader(const io::ResourceIO &resources)
			: resources_(resources) {}

		std::unique_ptr<Mesh> load_fem(const std::string &path, bool non_conforming = false) const;
		SurfaceMesh load_surface(const std::string &path) const;

	private:
		void validate_group(const std::string &group, const std::string &expected_type) const;
		const io::ResourceIO &resources_;
	};
} // namespace polyfem::mesh
