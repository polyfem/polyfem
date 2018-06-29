#pragma once

#include <geogram/mesh/mesh.h>


namespace polyfem
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
	void prepare_mesh(GEO::Mesh &M);

	// Retrieves the index (v,e,f) of one vertex incident to the given face
	Index get_index_from_face(const GEO::Mesh &M, int f, int lv);

	// Navigation in a surface mesh
	Index switch_vertex(const GEO::Mesh &M, Index idx);
	Index switch_edge(const GEO::Mesh &M, Index idx);
	Index switch_face(const GEO::Mesh &M, Index idx);

	// Iterate in a mesh
	inline Index next_around_face(const GEO::Mesh &M, Index idx) { return switch_edge(M, switch_vertex(M, idx)); }
	inline Index next_around_edge(const GEO::Mesh &M, Index idx) { return switch_vertex(M, switch_face(M, idx)); }
	inline Index next_around_vertex(const GEO::Mesh &M, Index idx) { return switch_face(M, switch_edge(M, idx)); }

	} // namespace Navigation

} // namespace polyfem
