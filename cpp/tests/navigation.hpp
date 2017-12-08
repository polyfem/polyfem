#pragma once

#include <geogram/mesh/mesh.h>


namespace poly_fem
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

	// Compute connectivity information on a surface mesh
	void prepare_mesh(GEO::Mesh &M);

	// Retrieve the index (v,e,f) of one vertex incident to the given face
	Index get_index_from_face(const GEO::Mesh &M, int f);

	// Navigation in a surface mesh
	Index switch_vertex(const GEO::Mesh &M, Index idx);
	Index switch_edge(const GEO::Mesh &M, Index idx);
	Index switch_face(const GEO::Mesh &M, Index idx);

	} // namespace Navigation

} // namespace poly_fem
