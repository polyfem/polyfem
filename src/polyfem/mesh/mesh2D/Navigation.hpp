#pragma once

#include <geogram/mesh/mesh.h>
#include <geogram/basic/attributes.h>

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
			void prepare_mesh(GEO::Mesh &M);

			// Retrieves the index (v,e,f) of one vertex incident to the given face
			Index get_index_from_face(const GEO::Mesh &M, const GEO::Attribute<GEO::index_t> &c2e, int f, int lv);

			// Navigation in a surface mesh
			Index switch_vertex(const GEO::Mesh &M, Index idx);
			Index switch_edge(const GEO::Mesh &M, const GEO::Attribute<GEO::index_t> &c2e, Index idx);
			Index switch_face(const GEO::Mesh &M, const GEO::Attribute<GEO::index_t> &c2e, Index idx);

			// Iterate in a mesh
			inline Index next_around_face(const GEO::Mesh &M, const GEO::Attribute<GEO::index_t> &c2e, Index idx) { return switch_edge(M, c2e, switch_vertex(M, idx)); }
			inline Index next_around_edge(const GEO::Mesh &M, const GEO::Attribute<GEO::index_t> &c2e, Index idx) { return switch_vertex(M, switch_face(M, c2e, idx)); }
			inline Index next_around_vertex(const GEO::Mesh &M, const GEO::Attribute<GEO::index_t> &c2e, Index idx) { return switch_face(M, c2e, switch_edge(M, c2e, idx)); }

		} // namespace Navigation
	} // namespace mesh
} // namespace polyfem
