#pragma once

#include <geogram/mesh/mesh.h>


namespace poly_fem
{

	namespace Navigation
	{

	struct Index
	{
		int v;  // vertex
		int e;  // edge
		int f;  // face
		int fc; // face corner
	};

	void prepare_mesh(GEO::Mesh &M);

	Index switch_vertex(const GEO::Mesh &M, Index idx);

	Index switch_edge(const GEO::Mesh &M, Index idx);

	Index switch_face(const GEO::Mesh &M, Index idx);

	} // namespace Navigation

} // namespace poly_fem
