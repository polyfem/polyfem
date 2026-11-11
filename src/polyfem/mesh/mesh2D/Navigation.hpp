#pragma once

#include <polyfem/mesh/mesh2D/PolygonMesh.hpp>

namespace polyfem
{
	namespace mesh
	{
		namespace Navigation
		{

			struct Index
			{
				int vertex;
				int edge;
				int face;
				int face_corner;
			};

			// Computes connectivity information on a surface mesh
			// Marks both boundary edges and vertices
			void prepare_mesh(PolygonMesh &M);

			// Retrieves the index (v,e,f) of one vertex incident to the given face
			Index get_index_from_face(const PolygonMesh &M, int f, int lv);

			// Navigation in a surface mesh
			Index switch_vertex(const PolygonMesh &M, Index idx);
			Index switch_edge(const PolygonMesh &M, Index idx);
			Index switch_face(const PolygonMesh &M, Index idx);

			// Iterate in a mesh
			inline Index next_around_face(const PolygonMesh &M, Index idx) { return switch_edge(M, switch_vertex(M, idx)); }
			inline Index next_around_edge(const PolygonMesh &M, Index idx) { return switch_vertex(M, switch_face(M, idx)); }
			inline Index next_around_vertex(const PolygonMesh &M, Index idx) { return switch_face(M, switch_edge(M, idx)); }

		} // namespace Navigation
	} // namespace mesh
} // namespace polyfem
