#pragma once

#include "Mesh3DStorage.hpp"

namespace polyfem
{
	namespace mesh
	{
		namespace Navigation3D
		{
			// extern double get_index_from_element_face_time;
			// extern double switch_vertex_time;
			// extern double switch_edge_time;
			// extern double switch_face_time;
			// extern double switch_element_time;

			struct Index
			{
				int vertex;
				int edge;
				int face;
				int face_corner;
				int element;
				int element_patch;
			};
			void prepare_mesh(Mesh3DStorage &M);
			// Retrieve the index (v,e,f,h) of one vertex incident to the given face and element
			Index get_index_from_element_face(const Mesh3DStorage &M, int hi);
			Index get_index_from_element_face(const Mesh3DStorage &M, int hi, int lf, int lv = 0);
			Index get_index_from_element_edge(const Mesh3DStorage &M, int hi, int v0, int v1);
			Index get_index_from_element_tri(const Mesh3DStorage &M, int hi, int v0, int v1, int v2);

			// Navigation in a surface Mesh3DStorage
			Index switch_vertex(const Mesh3DStorage &M, Index idx);
			Index switch_edge(const Mesh3DStorage &M, Index idx);
			Index switch_face(const Mesh3DStorage &M, Index idx);
			Index switch_element(const Mesh3DStorage &M, Index idx);

			// Iterate in a Mesh3DStorage
			inline Index next_around_2Dface(const Mesh3DStorage &M, Index idx) { return switch_edge(M, switch_vertex(M, idx)); }
			inline Index next_around_2Dedge(const Mesh3DStorage &M, Index idx) { return switch_vertex(M, switch_face(M, idx)); }
			inline Index next_around_2Dvertex(const Mesh3DStorage &M, Index idx) { return switch_face(M, switch_edge(M, idx)); }

			inline Index next_around_3Dedge(const Mesh3DStorage &M, Index idx) { return switch_element(M, switch_face(M, idx)); }
			// inline Index next_around_3Delement(const Mesh3DStorage &M, Index idx) { idx.element_patch++; return get_index_from_element_face(M, idx.element,idx.element_patch,idx.face_corner); }
		} // namespace Navigation3D
	} // namespace mesh
} // namespace polyfem
