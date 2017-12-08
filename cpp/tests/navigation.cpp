////////////////////////////////////////////////////////////////////////////////
#include "navigation.hpp"
#include <algorithm>
#include <cassert>
////////////////////////////////////////////////////////////////////////////////

using namespace GEO;

namespace {

	typedef std::pair<index_t, index_t> Edge;

	Edge make_edge(index_t v1, index_t v2) {
		return std::make_pair(std::min(v1, v2), std::max(v1, v2));
	}

} // anonymous namespace

// -----------------------------------------------------------------------------

void poly_fem::Navigation::prepare_mesh(GEO::Mesh &M) {
	M.facets.connect();
	M.cells.connect();
	if(M.cells.nb() != 0 && M.facets.nb() == 0) {
		M.cells.compute_borders();
	}

	// Compute a list of all the edges, and store edge index as a corner attribute
	std::vector<std::pair<Edge, index_t>> e2c; // edge to corner id
	for (index_t f = 0; f < M.facets.nb(); ++f) {
		for (index_t c = M.facets.corners_begin(f); c < M.facets.corners_end(f); ++c) {
			index_t v = M.facet_corners.vertex(c);
			index_t c2 = M.facets.next_corner_around_facet(f, c);
			index_t v2 = M.facet_corners.vertex(c2);
			e2c.emplace_back(make_edge(v, v2), c);
		}
	}
	std::sort(e2c.begin(), e2c.end());

	// Assign unique id to edges
	GEO::Attribute<index_t> c2e(M.facet_corners.attributes(), "edge_id");
	M.edges.clear();
	Edge prev_e(-1, -1);
	index_t current_id = -1;
	for (const auto &kv : e2c) {
		Edge e = kv.first;
		index_t c = kv.second;
		if (e != prev_e) {
			M.edges.create_edge(e.first, e.second);
			++current_id;
			prev_e = e;
		}
		c2e[c] = current_id;
	}
}

poly_fem::Navigation::Key poly_fem::Navigation::switch_vertex(const GEO::Mesh &M, Key idx)
{
	index_t c1 = M.facets.next_corner_around_facet(idx.f, idx.fc);
	index_t v1 = M.facet_corners.vertex(c1);
	if (v1 == M.edges.vertex(idx.e, 0) || v1 == M.edges.vertex(idx.e, 1)) {
		idx.fc = c1;
		idx.v = v1;
		return idx;
	} else {
		index_t c2 = M.facets.prev_corner_around_facet(idx.f, idx.fc);
		index_t v2 = M.facet_corners.vertex(c2);
		assert(v1 == M.edges.vertex(idx.e, 0) || v1 == M.edges.vertex(idx.e, 1));
		idx.fc = c2;
		idx.v = v2;
		return idx;
	}
}

poly_fem::Navigation::Key poly_fem::Navigation::switch_edge(const GEO::Mesh &M, Key idx)
{
	index_t v2 = M.edges.vertex(idx.e, 0);
	if (v2 == (index_t) idx.v) {
		v2 = M.edges.vertex(idx.e, 1);
	}
	//std::cout << M.edges.vertex(idx.e, 0) << ' ' << M.edges.vertex(idx.e, 1) << std::endl;
	GEO::Attribute<index_t> c2e(M.facet_corners.attributes(), "edge_id");
	index_t c1 = M.facets.next_corner_around_facet(idx.f, idx.fc);
	index_t v1 = M.facet_corners.vertex(c1);
	// auto show_corner = [&] (int c) {
	// 	index_t v = M.facet_corners.vertex(c);
	// 	index_t e = c2e[c];
	// 	const index_t *ve = M.edges.vertex_index_ptr(e);
	// 	std::cout << "c: " << c << " v: " << v << " e: " << e << " (" << ve[0] << "," << ve[1] << ")" << std::endl;
	// };
	// const index_t *ve = M.edges.vertex_index_ptr(idx.e);
	// std::cout << idx.v << ' ' << idx.e << " (" << ve[0] << ',' << ve[1] << ")" << std::endl;
	// show_corner(idx.fc);
	// show_corner(M.facets.next_corner_around_facet(idx.f, idx.fc));
	// show_corner(M.facets.prev_corner_around_facet(idx.f, idx.fc));
	index_t c3 = M.facets.prev_corner_around_facet(idx.f, idx.fc);
	index_t v3 = M.facet_corners.vertex(c3);
	//std::cout << v1 << ' ' << idx.v << ' ' << v2 << std::endl;
	if (v1 == v2) {
		index_t c2 = M.facets.prev_corner_around_facet(idx.f, idx.fc);
		idx.e = c2e[c2];
		// std::cout << idx.e << std::endl;
		return idx;
	} else {
		idx.e = c2e[idx.fc];
		// std::cout << idx.e << std::endl;
		return idx;
	}
}

poly_fem::Navigation::Key poly_fem::Navigation::switch_face(const GEO::Mesh &M, Key idx)
{
	return {};
}

