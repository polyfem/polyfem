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

		/// @brief saves the mesh
		/// @param[in] path output path
		/// @param[in] binary output binary or not
		static void write(
			const std::string &path,
			const mesh::Mesh &mesh,
			const bool binary);

		/// @brief saves the mesh
		/// @param[in] path output path
		/// @param[in] binary output binary or not
		static void write(
			const std::string &path,
			const Eigen::MatrixXd &points,
			const Eigen::MatrixXi &cells,
			const std::vector<int> &body_ids,
			const bool is_volume,
			const bool binary = false);

		/// @brief saves the mesh
		/// @param[in] path output path
		/// @param[in] binary output binary or not
		static void write(
			const std::string &path,
			const Eigen::MatrixXd &points,
			const std::vector<std::vector<int>> &cells,
			const std::vector<int> &body_ids,
			const bool is_volume,
			const bool binary = false);
	};
} // namespace polyfem::io
