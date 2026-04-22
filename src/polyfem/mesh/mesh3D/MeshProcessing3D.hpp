#pragma once

#include "Mesh3DStorage.hpp"
#include <iostream>
#include <fstream>
#include <array>

namespace polyfem
{
	namespace mesh
	{
		namespace MeshProcessing3D
		{
#define Jacobian_Precision 1.e-7
			const int hex_face_table[6][4] =
				{
					{0, 1, 2, 3},
					{4, 7, 6, 5},
					{0, 4, 5, 1},
					{3, 2, 6, 7},
					{0, 3, 7, 4},
					{1, 5, 6, 2},
			};
			const int hex_tetra_table[8][4] =
				{
					{0, 3, 4, 1},
					{1, 0, 5, 2},
					{2, 1, 6, 3},
					{3, 2, 7, 0},
					{4, 7, 5, 0},
					{5, 4, 6, 1},
					{6, 5, 7, 2},
					{7, 6, 4, 3},
			};
			const int tet_faces[4][3] = {
				{1, 0, 2},
				{3, 2, 0},
				{1, 2, 3},
				{0, 1, 3}};
			const int tet_edges[6][2] = {
				{0, 1},
				{0, 2},
				{0, 3},
				{1, 2},
				{1, 3},
				{2, 3}};

			void build_connectivity(Mesh3DStorage &hmi);
			void reorder_hex_mesh_propogation(Mesh3DStorage &hmi);
			bool scaled_jacobian(Mesh3DStorage &hmi, Mesh_Quality &mq);
			double a_jacobian(Eigen::Vector3d &v0, Eigen::Vector3d &v1, Eigen::Vector3d &v2, Eigen::Vector3d &v3);

			void global_orientation_hexes(Mesh3DStorage &hmi);
			void refine_catmul_clark_polar(Mesh3DStorage &M, int iter, bool reverse, std::vector<int> &Parents);
			void refine_red_refinement_tet(Mesh3DStorage &M, int iter);
			// Mi is a planar surface mesh
			void straight_sweeping(const Mesh3DStorage &Mi, int sweep_coord, double height, int nlayer, Mesh3DStorage &Mo);

			void orient_surface_mesh(Mesh3DStorage &hmi);
			void orient_volume_mesh(Mesh3DStorage &hmi);
			void ele_subdivison_levels(const Mesh3DStorage &hmi, std::vector<int> &Ls);

			// template<typename T>
			void set_intersection_own(const std::vector<uint32_t> &A, const std::vector<uint32_t> &B, std::array<uint32_t, 2> &C, int &num);
		} // namespace MeshProcessing3D
	} // namespace mesh
} // namespace polyfem
