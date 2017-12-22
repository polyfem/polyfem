#pragma once

#include "Mesh3DStorage.hpp"
#include <iostream>
#include <fstream>

namespace poly_fem{
	namespace MeshProcessing3D{
		const int hex_face_table[6][4] =
		{
			{ 0,1,2,3 },
			{ 4,5,6,7 },
			{ 0,1,5,4 },
			{ 0,4,7,3 },
			{ 3,2,6,7 },
			{ 1,5,6,2 },
		};
		void build_connectivity(Mesh3DStorage &hmi);
		void refine_catmul_clark_polar(Mesh3DStorage &M, int iter);

		void  orient_surface_mesh(Mesh3DStorage &hmi);

	} // namespace Navigation3D
} // namespace poly_fem

