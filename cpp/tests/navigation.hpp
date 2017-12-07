#pragma once

#include <geogram/mesh/mesh.h>


namespace poly_fem
{

	namespace Navigation
	{

	void prepare_mesh(GEO::Mesh &M);

	int switch_vertex(const GEO::Mesh &M, int v, int e, int f);

	int switch_edge(const GEO::Mesh &M, int v, int e, int f);

	int switch_face(const GEO::Mesh &M, int v, int e, int f);

	} // namespace Navigation

} // namespace poly_fem
