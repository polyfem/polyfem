////////////////////////////////////////////////////////////////////////////////
#include <polyfem/mesh/mesh2D/Navigation.hpp>

#include <polyfem/utils/Logger.hpp>

#include <algorithm>
#include <cassert>
////////////////////////////////////////////////////////////////////////////////

using namespace GEO;
using namespace polyfem::mesh::Navigation;

namespace
{

	typedef std::pair<index_t, index_t> Edge;

	Edge make_edge(index_t v1, index_t v2)
	{
		return std::make_pair(std::min(v1, v2), std::max(v1, v2));
	}

} // anonymous namespace

// -----------------------------------------------------------------------------

void polyfem::mesh::Navigation::prepare_mesh(GEO::Mesh &M)
{
	M.facets.connect();
	M.cells.connect();
	if (M.cells.nb() != 0 && M.facets.nb() == 0)
	{
		M.cells.compute_borders();
	}

	// Compute a list of all the edges, and store edge index as a corner attribute
	std::vector<std::pair<Edge, index_t>> e2c; // edge to corner id
	for (index_t f = 0; f < M.facets.nb(); ++f)
	{
		for (index_t c = M.facets.corners_begin(f); c < M.facets.corners_end(f); ++c)
		{
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
	std::vector<bool> boundary_edges;
	for (const auto &kv : e2c)
	{
		Edge e = kv.first;
		index_t c = kv.second;
		if (e != prev_e)
		{
			M.edges.create_edge(e.first, e.second);
			boundary_edges.push_back(true);
			++current_id;
			prev_e = e;
		}
		else
		{
			boundary_edges.back() = false;
		}
		c2e[c] = current_id;
	}

	GEO::Attribute<bool> boundary_edges_attr(M.edges.attributes(), "boundary_edge");
	assert(M.edges.nb() == (index_t)boundary_edges.size());
	for (index_t e = 0; e < M.edges.nb(); ++e)
	{
		boundary_edges_attr[e] = boundary_edges[e] ? 1 : 0;
	}

	GEO::Attribute<bool> boundary_vertices(M.vertices.attributes(), "boundary_vertex");
	boundary_vertices.fill(0);
	for (index_t e = 0; e < M.edges.nb(); ++e)
	{
		if (boundary_edges[e])
		{
			boundary_vertices[M.edges.vertex(e, 0)] = 1;
			boundary_vertices[M.edges.vertex(e, 1)] = 1;
		}
	}
}

// -----------------------------------------------------------------------------

// Retrieve the index (v,e,f) of one vertex incident to the given face
Index polyfem::mesh::Navigation::get_index_from_face(const GEO::Mesh &M, const GEO::Attribute<GEO::index_t> &c2e, int f, int lv)
{
	// GEO::Attribute<index_t> c2e(M.facet_corners.attributes(), "edge_id");
	Index idx;
	idx.face_corner = M.facets.corner(f, lv);
	idx.vertex = M.facet_corners.vertex(idx.face_corner);
	idx.face = f;
	idx.edge = c2e[idx.face_corner];
	int v2 = int(M.facets.vertex(f, (lv + 1) % M.facets.nb_vertices(f)));
	if (switch_vertex(M, idx).vertex != v2)
	{
		assert(false);
		return switch_edge(M, c2e, idx);
	}
	return idx;
}

////////////////////////////////////////////////////////////////////////////////

Index polyfem::mesh::Navigation::switch_vertex(const GEO::Mesh &M, Index idx)
{
	index_t c1 = M.facets.next_corner_around_facet(idx.face, idx.face_corner);
	index_t v1 = M.facet_corners.vertex(c1);
	if (v1 == M.edges.vertex(idx.edge, 0) || v1 == M.edges.vertex(idx.edge, 1))
	{
		idx.face_corner = c1;
		idx.vertex = v1;
		return idx;
	}
	else
	{
		index_t c2 = M.facets.prev_corner_around_facet(idx.face, idx.face_corner);
		index_t v2 = M.facet_corners.vertex(c2);
		assert(v2 == M.edges.vertex(idx.edge, 0) || v2 == M.edges.vertex(idx.edge, 1));
		idx.face_corner = c2;
		idx.vertex = v2;
		return idx;
	}
}

Index polyfem::mesh::Navigation::switch_edge(const GEO::Mesh &M, const GEO::Attribute<GEO::index_t> &c2e, Index idx)
{
	index_t v2 = M.edges.vertex(idx.edge, 0);
	if (v2 == (index_t)idx.vertex)
	{
		v2 = M.edges.vertex(idx.edge, 1);
	}
	// GEO::Attribute<index_t> c2e(M.facet_corners.attributes(), "edge_id");
	index_t c1 = M.facets.next_corner_around_facet(idx.face, idx.face_corner);
	index_t v1 = M.facet_corners.vertex(c1);
	if (v1 == v2)
	{
		index_t c2 = M.facets.prev_corner_around_facet(idx.face, idx.face_corner);
		idx.edge = c2e[c2];
		return idx;
	}
	else
	{
		idx.edge = c2e[idx.face_corner];
		return idx;
	}
}

Index polyfem::mesh::Navigation::switch_face(const GEO::Mesh &M, const GEO::Attribute<GEO::index_t> &c2e, Index idx)
{
	// GEO::Attribute<index_t> c2e(M.facet_corners.attributes(), "edge_id");
	index_t c1 = idx.face_corner;
	if (c2e[c1] != (index_t)idx.edge)
	{
		c1 = M.facets.prev_corner_around_facet(idx.face, c1);
	}
	index_t f2 = M.facet_corners.adjacent_facet(c1);
	if (f2 == NO_FACET)
	{
		idx.face = -1;
		return idx;
	}
	else
	{
		// Iterate over all corners of the new face until we find the vertex we came from.
		// Not ideal but for now it will do the job.
		for (index_t c2 = M.facets.corners_begin(f2); c2 < M.facets.corners_end(f2); ++c2)
		{
			index_t v2 = M.facet_corners.vertex(c2);
			if (v2 == (index_t)idx.vertex)
			{
				idx.face = f2;
				idx.face_corner = c2;
				return idx;
			}
		}
		logger().error("Not found");
		assert(false); // This should not happen
		return idx;
	}
}
