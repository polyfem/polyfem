#pragma once

#include <polyfem/mesh/MeshData.hpp>

#include <filesystem>
#include <istream>

namespace GEO
{
	class Mesh;
}

namespace polyfem::mesh
{
	class SurfaceMeshData
	{
	public:
		Eigen::MatrixXd vertices;
		Eigen::VectorXi points;
		Eigen::MatrixXi edges;
		Eigen::MatrixXi faces;
		/// Volumetric cells indexing vertices; only filled when a 3D tet mesh is
		/// read with keep_volume.
		Eigen::MatrixXi tets;
	};

	/// Stateless format decoders used exclusively by MeshLoader and in-memory adapters.
	class MeshReader
	{
	public:
		static MeshData read_msh(const std::filesystem::path &path);
		static MeshData read_hybrid(std::istream &input, const std::string &description);
		static MeshData read_geogram(const std::filesystem::path &path);
		static MeshData from_geogram(GEO::Mesh &mesh);

		/// @param keep_volume For a 3D tet mesh, keep every vertex and the tets,
		///        and index the boundary faces into the full vertex set.
		static SurfaceMeshData read_msh_surface(const std::filesystem::path &path, bool keep_volume = false);
		static SurfaceMeshData read_obj_surface(const std::filesystem::path &path);
		static bool read_triangle_surface(const std::filesystem::path &path, SurfaceMeshData &result);
		static SurfaceMeshData read_geogram_surface(const std::filesystem::path &path);
	};
} // namespace polyfem::mesh
