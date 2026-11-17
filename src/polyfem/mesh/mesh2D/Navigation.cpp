////////////////////////////////////////////////////////////////////////////////
#include <polyfem/mesh/mesh2D/Navigation.hpp>

#include <polyfem/utils/Logger.hpp>

#include <algorithm>
#include <cassert>
#include <utility>
#include <vector>
////////////////////////////////////////////////////////////////////////////////

using namespace polyfem::mesh::Navigation;

namespace
{

	typedef std::pair<int, int> Edge;

	Edge make_edge(int v1, int v2)
	{
		return std::make_pair(std::min(v1, v2), std::max(v1, v2));
	}

} // anonymous namespace

// -----------------------------------------------------------------------------

void polyfem::mesh::Navigation::prepare_mesh(PolygonMesh &M)
{
	M.connect_facets();

	// Compute a list of all the edges, and store edge index as a corner attribute
	std::vector<std::pair<Edge, int>> e2c; // edge to corner id
	for (int f = 0; f < M.n_facets(); ++f)
	{
		for (int c = M.corners_begin(f); c < M.corners_end(f); ++c)
		{
			int v = M.corner_vertex(c);
			int c2 = M.next_corner_around_facet(f, c);
			int v2 = M.corner_vertex(c2);
			e2c.emplace_back(make_edge(v, v2), c);
		}
	}
	std::sort(e2c.begin(), e2c.end());

	// Assign unique id to edges
	M.clear_edges();
	Edge prev_e(-1, -1);
	int current_id = -1;
	std::vector<bool> boundary_edges;
	for (const auto &kv : e2c)
	{
		Edge e = kv.first;
		int c = kv.second;
		if (e != prev_e)
		{
			M.create_edge(e.first, e.second);
			boundary_edges.push_back(true);
			++current_id;
			prev_e = e;
		}
		else
		{
			boundary_edges.back() = false;
		}
		M.corner_edge[c] = current_id;
	}

	assert(M.n_edges() == (int)boundary_edges.size());
	for (int e = 0; e < M.n_edges(); ++e)
	{
		M.boundary_edge[e] = boundary_edges[e];
	}

	std::fill(M.boundary_vertex.begin(), M.boundary_vertex.end(), false);
	for (int e = 0; e < M.n_edges(); ++e)
	{
		if (boundary_edges[e])
		{
			M.boundary_vertex[M.edge_vertex(e, 0)] = true;
			M.boundary_vertex[M.edge_vertex(e, 1)] = true;
		}
	}
}

// -----------------------------------------------------------------------------

// Retrieve the index (v,e,f) of one vertex incident to the given face
Index polyfem::mesh::Navigation::get_index_from_face(const PolygonMesh &M, int f, int lv)
{
	Index idx;
	idx.face_corner = M.facet_corner(f, lv);
	idx.vertex = M.corner_vertex(idx.face_corner);
	idx.face = f;
	idx.edge = M.corner_edge[idx.face_corner];
	int v2 = M.facet_vertex(f, (lv + 1) % M.n_facet_vertices(f));
	if (switch_vertex(M, idx).vertex != v2)
	{
		assert(false);
		return switch_edge(M, idx);
	}
	return idx;
}

////////////////////////////////////////////////////////////////////////////////

Index polyfem::mesh::Navigation::switch_vertex(const PolygonMesh &M, Index idx)
{
	int c1 = M.next_corner_around_facet(idx.face, idx.face_corner);
	int v1 = M.corner_vertex(c1);
	if (v1 == M.edge_vertex(idx.edge, 0) || v1 == M.edge_vertex(idx.edge, 1))
	{
		idx.face_corner = c1;
		idx.vertex = v1;
		return idx;
	}
	else
	{
		int c2 = M.prev_corner_around_facet(idx.face, idx.face_corner);
		int v2 = M.corner_vertex(c2);
		assert(v2 == M.edge_vertex(idx.edge, 0) || v2 == M.edge_vertex(idx.edge, 1));
		idx.face_corner = c2;
		idx.vertex = v2;
		return idx;
	}
}

Index polyfem::mesh::Navigation::switch_edge(const PolygonMesh &M, Index idx)
{
	int v2 = M.edge_vertex(idx.edge, 0);
	if (v2 == idx.vertex)
	{
		v2 = M.edge_vertex(idx.edge, 1);
	}
	int c1 = M.next_corner_around_facet(idx.face, idx.face_corner);
	int v1 = M.corner_vertex(c1);
	if (v1 == v2)
	{
		int c2 = M.prev_corner_around_facet(idx.face, idx.face_corner);
		idx.edge = M.corner_edge[c2];
		return idx;
	}
	else
	{
		idx.edge = M.corner_edge[idx.face_corner];
		return idx;
	}
}

Index polyfem::mesh::Navigation::switch_face(const PolygonMesh &M, Index idx)
{
	int c1 = idx.face_corner;
	if (M.corner_edge[c1] != idx.edge)
	{
		c1 = M.prev_corner_around_facet(idx.face, c1);
	}
	int f2 = M.corner_adjacent_facet(c1);
	if (f2 == -1)
	{
		idx.face = -1;
		return idx;
	}
	else
	{
		// Iterate over all corners of the new face until we find the vertex we came from.
		// Not ideal but for now it will do the job.
		for (int c2 = M.corners_begin(f2); c2 < M.corners_end(f2); ++c2)
		{
			int v2 = M.corner_vertex(c2);
			if (v2 == idx.vertex)
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
