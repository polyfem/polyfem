// Modified version of read_obj from libigl to include reading polyline elements
// as edges.
//
// Copyright (C) 2013 Alec Jacobson <alecjacobson@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <string>
#include <vector>

#include <Eigen/Core>

namespace polyfem
{
	namespace utils
	{
		class OBJReader
		{
		public:
			OBJReader() = delete;

			/// @brief Read a mesh from an ascii obj file
			///
			/// Fills in vertex positions, normals and texture coordinates. Mesh may
			/// have faces of any number of degree.
			///
			/// @param[in] obj_file_name  path to .obj file
			/// @param[out] V             double matrix of vertex positions
			/// @param[out] TC            double matrix of texture coordinates
			/// @param[out] N             double matrix of corner normals #N by 3
			/// @param[out] F             #F list of face indices into vertex positions
			/// @param[out] FTC           #F list of face indices into vertex texture
			///                           coordinates
			/// @param[out] FN            #F list of face indices into vertex normals
			/// @param[out] L             list of polyline indices into vertex positions
			///
			/// @returns true on success, false on errors
			static bool load(
				const std::string obj_file_name,
				std::vector<std::vector<double>> &V,
				std::vector<std::vector<double>> &TC,
				std::vector<std::vector<double>> &N,
				std::vector<std::vector<int>> &F,
				std::vector<std::vector<int>> &FTC,
				std::vector<std::vector<int>> &FN,
				std::vector<std::vector<int>> &L);

			/// @brief Read a mesh from an already opened ascii obj file
			/// @param[in] obj_file  pointer to already opened .obj file
			static bool load(
				FILE *obj_file,
				std::vector<std::vector<double>> &V,
				std::vector<std::vector<double>> &TC,
				std::vector<std::vector<double>> &N,
				std::vector<std::vector<int>> &F,
				std::vector<std::vector<int>> &FTC,
				std::vector<std::vector<int>> &FN,
				std::vector<std::vector<int>> &L);

			/// @brief Just read V, F, and L from obj file
			static bool load(
				const std::string obj_file_name,
				std::vector<std::vector<double>> &V,
				std::vector<std::vector<int>> &F,
				std::vector<std::vector<int>> &L);

			/// @brief Eigen Wrappers of read_obj.
			/// @retruns These will return true only if the data is perfectly
			///          "rectangular": All faces are the same degree, all have the same
			///          number of textures/normals etc.
			static bool load(
				const std::string str,
				Eigen::MatrixXd &V,
				Eigen::MatrixXi &E,
				Eigen::MatrixXi &F);
		};

		class OBJWriter
		{
		public:
			OBJWriter() = delete;

			static bool save(
				const std::string &path,
				const Eigen::MatrixXd &v,
				const Eigen::MatrixXi &e,
				const Eigen::MatrixXi &f);

			static bool save(
				const std::string &path,
				const Eigen::MatrixXd &v,
				const Eigen::MatrixXi &f)
			{
				return save(path, v, Eigen::MatrixXi(), f);
			}
		};
	} // namespace utils
} // namespace polyfem
