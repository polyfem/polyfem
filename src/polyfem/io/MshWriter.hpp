#pragma once

#include <polyfem/mesh/Mesh.hpp>

#include <string>

#include <Eigen/Core>

namespace polyfem::io
{
	class MshWriter
	{
	public:
		MshWriter() = delete;

		/// @brief saves the mesh, currently only msh supported
		/// @param[in] path output path
		/// @param[in] binary output binary or not
		static void write(
			const std::string &path,
			const mesh::Mesh &mesh,
			const bool binary = false);
	};
} // namespace polyfem::io
