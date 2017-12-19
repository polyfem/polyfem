#pragma once

#include "Mesh3D.hpp"
using namespace std;

namespace poly_fem{
	namespace Navigation3D{

		struct Index{
			int vertex;
			int edge;
			int face;
			int face_corner;
			int hedra;
			int hedra_patch;
		};
		void prepare_mesh(Mesh &M);
		void build_connectivity(Mesh &M);
		// Retrieve the index (v,e,f,h) of one vertex incident to the given face and polyhedra
		Index get_index_from_hedraface(const Mesh &M, int hi, int lf, int lv = 0);

		// Navigation in a surface mesh
		Index switch_vertex(const Mesh &M, Index idx);
		Index switch_edge(const Mesh &M, Index idx);
		Index switch_face(const Mesh &M, Index idx);
		Index switch_hedra(const Mesh &M, Index idx);

		// Iterate in a mesh
		inline Index next_hf(const Mesh &M, Index idx) { return switch_hedra(M, switch_face(M, idx)); }
		inline Index next_around_face(const Mesh &M, Index idx) { return switch_edge(M, switch_vertex(M, idx)); }
		inline Index next_around_edge(const Mesh &M, Index idx) { return switch_vertex(M, switch_face(M, idx)); }
		inline Index next_around_vertex(const Mesh &M, Index idx) { return switch_face(M, switch_edge(M, idx)); }

	} // namespace Navigation3D
} // namespace poly_fem
