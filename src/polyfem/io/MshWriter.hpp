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
			const Eigen::MatrixXd &V,
			const Eigen::MatrixXi &F,
			const std::vector<int> &body_ids,
			const bool is_volume,
			const bool binary = false);
	};
} // namespace polyfem::io
