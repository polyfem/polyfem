////////////////////////////////////////////////////////////////////////////////
#include "MeshUtils.hpp"
#include <geogram/basic/geometry.h>
#include <geogram/mesh/mesh_preprocessing.h>
#include <geogram/mesh/mesh_topology.h>
#include <geogram/mesh/mesh_geometry.h>
////////////////////////////////////////////////////////////////////////////////

GEO::vec3 poly_fem::mesh_vertex(const GEO::Mesh &M, GEO::index_t v) {
	using GEO::index_t;
	GEO::vec3 p(0, 0, 0);
	for (index_t d = 0; d < std::min(3u, (index_t) M.vertices.dimension()); ++d) {
		if (M.vertices.double_precision()) {
			p[d] = M.vertices.point_ptr(v)[d];
		} else {
			p[d] = M.vertices.single_precision_point_ptr(v)[d];
		}
	}
	return p;
}

// -----------------------------------------------------------------------------

GEO::vec3 poly_fem::facet_barycenter(const GEO::Mesh &M, GEO::index_t f) {
	using GEO::index_t;
	GEO::vec3 p(0, 0, 0);
	for (index_t lv = 0; lv < M.facets.nb_vertices(f); ++lv) {
		p += poly_fem::mesh_vertex(M, M.facets.vertex(f, lv));
	}
	return p / M.facets.nb_vertices(f);
}

// -----------------------------------------------------------------------------

GEO::index_t poly_fem::mesh_create_vertex(GEO::Mesh &M, const GEO::vec3 &p) {
	using GEO::index_t;
	auto v = M.vertices.create_vertex();
	for (index_t d = 0; d < std::min(3u, (index_t) M.vertices.dimension()); ++d) {
		if (M.vertices.double_precision()) {
			M.vertices.point_ptr(v)[d] = p[d];
		} else {
			M.vertices.single_precision_point_ptr(v)[d] = (float) p[d];
		}
	}
	return v;
}

////////////////////////////////////////////////////////////////////////////////

void poly_fem::compute_element_tags(const GEO::Mesh &M, std::vector<ElementType> &element_tags) {
	using GEO::index_t;
	element_tags.resize(M.facets.nb());

	// Step 0: Compute boundary vertices as true boundary + vertices incident to a polygon
	std::vector<bool> is_boundary_vertex(M.vertices.nb(), false);
	{
		GEO::Attribute<int> boundary_vertices(M.vertices.attributes(), "boundary_vertex");
		for (index_t f = 0; f < M.facets.nb(); ++f) {
			if (M.facets.nb_vertices(f) != 4) {
				// Vertices incident to polygonal facets (triangles or > 4 vertices) are marked as boundary
				for (index_t lv = 0; lv < M.facets.nb_vertices(f); ++lv) {
					is_boundary_vertex[M.facets.vertex(f, lv)] = true;
				}
			}
		}
		for (index_t v = 0; v < M.vertices.nb(); ++v) {
			if (boundary_vertices[v]) {
				is_boundary_vertex[v] = true;
			}
		}
	}

	// Step 1: Determine which vertices are regular or not
	//
	// Interior vertices are regular if they are incident to exactly 4 quads
	// Boundary vertices are regular if they are incident to at most 2 quads, and no other facets
	std::vector<int> degree(M.vertices.nb(), 0);
	std::vector<bool> is_regular_vertex(M.vertices.nb());
	for (index_t f =  0; f < M.facets.nb(); ++f) {
		if (M.facets.nb_vertices(f) == 4) {
			// Only count incident quads for the degree
			for (index_t lv = 0; lv < M.facets.nb_vertices(f); ++lv) {
				index_t v = M.facets.vertex(f, lv);
				degree[v]++;
			}
		}
	}
	for (index_t v = 0; v < M.vertices.nb(); ++v) {
		// assert(degree[v] > 0); // We assume there are no isolated vertices here
		if (is_boundary_vertex[v]) {
			is_regular_vertex[v] = (degree[v] <= 2);
		} else {
			is_regular_vertex[v] = (degree[v] == 4);
		}
	}

	// Step 2: Iterate over the facets and determine the type
	for (index_t f =  0; f < M.facets.nb(); ++f) {
		assert(M.facets.nb_vertices(f) > 2);
		if (M.facets.nb_vertices(f) == 4) {
			// Quad facet

			// a) Determine if it is on the mesh boundary
			bool is_boundary_facet = false;
			for (index_t lv = 0; lv < M.facets.nb_vertices(f); ++lv) {
				if (is_boundary_vertex[M.facets.vertex(f, lv)]) {
					is_boundary_facet = true;
					break;
				}
			}

			// b) Determine if it is regular or not
			if (is_boundary_facet) {
				// A boundary quad is regular iff all its vertices are incident to at most 2 other quads
				// We assume that non-boundary vertices of a boundary quads are always regular
				bool is_singular = false;
				for (index_t lv = 0; lv < M.facets.nb_vertices(f); ++lv) {
					index_t v = M.facets.vertex(f, lv);
					if (is_boundary_vertex[v]) {
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
				if (is_boundary_vertex[M.facets.vertex(f, lv)]) {
					tag = ElementType::BoundaryPolytope;
					break;
				}
			}

			element_tags[f] = tag;
		}
	}
}

////////////////////////////////////////////////////////////////////////////////

namespace {

	// Signed area of a polygonal facet
	double signed_area(const GEO::Mesh& M, GEO::index_t f) {
		using namespace GEO;
		double result = 0;
		index_t v0 = M.facet_corners.vertex(M.facets.corners_begin(f));
		const vec3& p0 = Geom::mesh_vertex(M, v0);
		for(index_t c =
			M.facets.corners_begin(f) + 1; c + 1 < M.facets.corners_end(f); c++
		) {
			index_t v1 = M.facet_corners.vertex(c);
			const vec3& p1 = poly_fem::mesh_vertex(M, v1);
			index_t v2 = M.facet_corners.vertex(c + 1);
			const vec3& p2 = poly_fem::mesh_vertex(M, v2);
			result += Geom::triangle_signed_area(vec2(&p0[0]), vec2(&p1[0]), vec2(&p2[0]));
		}
		return result;
	}

} // anonymous namespace

void poly_fem::orient_normals_2d(GEO::Mesh &M) {
	using namespace GEO;
	vector<index_t> component;
	index_t nb_components = get_connected_components(M, component);
	vector<double> comp_signed_volume(nb_components, 0.0);
	for (index_t f = 0; f < M.facets.nb(); ++f) {
		comp_signed_volume[component[f]] += signed_area(M, f);
	}
	for (index_t f = 0; f < M.facets.nb(); ++f) {
		if (comp_signed_volume[component[f]] < 0.0) {
			M.facets.flip(f);
		}
	}
}
