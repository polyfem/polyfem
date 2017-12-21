#include "MeshUtils.hpp"


void poly_fem::compute_element_tags(const GEO::Mesh &M, std::vector<ElementType> &element_tags) {
	using GEO::index_t;
	element_tags.resize(M.facets.nb());

	GEO::Attribute<int> boundary_vertices(M.vertices.attributes(), "boundary_vertex");

	// Step 1: Determine which vertices are regular or not
	//
	// Interior vertices are regular if they are incident to exactly 4 quads
	// Boundary vertices are regular if they are incident to at most 2 quads, and no other facets
	std::vector<int> degree(M.vertices.nb(), 0);
	std::vector<bool> is_regular_vertex(M.vertices.nb());
	for (index_t c = 0; c < M.facet_corners.nb(); ++c) {
		index_t v = M.facet_corners.vertex(c);
		degree[v]++;
	}
	for (index_t v = 0; v < M.vertices.nb(); ++v) {
		assert(degree[v] > 0); // We assume there are no isolated vertices here
		if (boundary_vertices[v]) {
			is_regular_vertex[v] = (degree[v] <= 2);
		} else {
			is_regular_vertex[v] = (degree[v] == 4);
		}
	}
	for (index_t f = 0; f < M.facets.nb(); ++f) {
		if (M.facets.nb_vertices(f) != 4) {
			// Vertices incident to polygonal facets (triangles or > 4 vertices) are marked as singular
			for (index_t lv = 0; lv < M.facets.nb_vertices(f); ++lv) {
				is_regular_vertex[M.facets.vertex(f, lv)] = false;
			}
		}
	}

	// Step 2: Iterate over the facets and determine the type
	for (index_t f =  0; f < M.facets.nb(); ++f) {
		assert(M.facets.nb_vertices(f) > 2);
		if (M.facets.nb_vertices(f) == 4) {
			// Quad facet

			// a) Determine if it is on the mesh boundary
			bool is_boundary = false;
			for (index_t lv = 0; lv < M.facets.nb_vertices(f); ++lv) {
				if (boundary_vertices[M.facets.vertex(f, lv)]) {
					is_boundary = true;
					break;
				}
			}

			// b) Determine if it is regular or not
			if (is_boundary) {
				// A boundary quad is regular iff all its vertices are incident to at most 2 other quads
				// We assume that non-boundary vertices of a boundary quads are always regular
				bool is_singular = false;
				for (index_t lv = 0; lv < M.facets.nb_vertices(f); ++lv) {
					index_t v = M.facets.vertex(f, lv);
					if (boundary_vertices[v]) {
						if (!is_regular_vertex[v]) {
							is_singular = true;
							break;
						}
					} else {
						if (!is_regular_vertex[v]) {
							element_tags[f] = ElementType::Undefined;
							break;
						}
					}
				}

				if (is_singular) {
					element_tags[f] = ElementType::SingularBoundaryCube;
				} else {
					element_tags[f] = ElementType::RegularBoundaryCube;
				}
			} else {
				// An interior quad is regular if all its vertices are singular
				int nb_singulars = 0;
				for (index_t lv = 0; lv < M.facets.nb_vertices(f); ++lv) {
					if (!is_regular_vertex[M.facets.vertex(f, lv)]) {
						++nb_singulars;
					}
				}

				if (nb_singulars == 0) {
					element_tags[f] = ElementType::RegularInteriorCube;
				} else if (nb_singulars == 1) {
					element_tags[f] = ElementType::SimpleSingularInteriorCube;
				} else {
					element_tags[f] = ElementType::MultiSingularInteriorCube;
				}
			}
		} else {
			// Polygonal facet

			// Note: In this function, we consider triangles as polygonal facets
			ElementType tag = ElementType::InteriorPolytope;
			for (index_t lv = 0; lv < M.facets.nb_vertices(f); ++lv) {
				if (boundary_vertices[M.facets.vertex(f, lv)]) {
					tag = ElementType::BoundaryPolytope;
					break;
				}
			}

			element_tags[f] = tag;
		}
	}
}
