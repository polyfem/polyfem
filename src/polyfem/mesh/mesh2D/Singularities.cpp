////////////////////////////////////////////////////////////////////////////////
#include "Singularities.hpp"
#include "Navigation.hpp"
#include <polyfem/mesh/MeshUtils.hpp>
#include <algorithm>
////////////////////////////////////////////////////////////////////////////////

void polyfem::mesh::singular_vertices(
	const GEO::Mesh &M, Eigen::VectorXi &V, int regular_degree, bool ignore_border)
{
	using GEO::index_t;
	std::vector<int> degree(M.vertices.nb(), 0);
	for (index_t f = 0; f < M.facets.nb(); ++f)
	{
		for (index_t lv = 0; lv < M.facets.nb_vertices(f); ++lv)
		{
			index_t v = M.facets.vertex(f, lv);
			degree[v]++;
		}
	}

	// Ignore border vertices if requested
	if (ignore_border)
	{
		GEO::Attribute<bool> boundary_vertices(M.vertices.attributes(), "boundary_vertex");
		for (index_t v = 0; v < M.vertices.nb(); ++v)
		{
			if (boundary_vertices[v])
			{
				degree[v] = regular_degree;
			}
		}
	}

	// Count and create list of irregular vertices
	long nb_reg = std::count(degree.begin(), degree.end(), regular_degree);
	V.resize(M.vertices.nb() - nb_reg);
	for (index_t v = 0, i = 0; v < M.vertices.nb(); ++v)
	{
		if (degree[v] != regular_degree)
		{
			assert(i < V.size());
			V(i++) = v;
		}
	}
}

void polyfem::mesh::singular_edges(const GEO::Mesh &M, const Eigen::VectorXi &V, Eigen::MatrixX2i &E)
{
	using GEO::index_t;
	std::vector<bool> is_singular(M.vertices.nb(), false);
	for (int i = 0; i < V.size(); ++i)
	{
		is_singular[V(i)] = true;
	}
	std::vector<std::pair<int, int>> edges;
	for (index_t e = 0; e < M.edges.nb(); ++e)
	{
		index_t v1 = M.edges.vertex(e, 0);
		index_t v2 = M.edges.vertex(e, 1);
		if (is_singular[v1] && is_singular[v2])
		{
			edges.emplace_back(v1, v2);
		}
	}
	E.resize(edges.size(), 2);
	int cnt = 0;
	for (const auto &kv : edges)
	{
		assert(cnt < E.rows());
		E.row(cnt++) << kv.first, kv.second;
	}
}

void polyfem::mesh::singularity_graph(
	const GEO::Mesh &M, Eigen::VectorXi &V, Eigen::MatrixX2i &E, int regular_degree, bool ignore_border)
{
	singular_vertices(M, V, regular_degree, ignore_border);
	singular_edges(M, V, E);
}

////////////////////////////////////////////////////////////////////////////////

void polyfem::mesh::create_patch_around_singularities(
	GEO::Mesh &M, const Eigen::VectorXi &V, const Eigen::MatrixX2i &E, double t)
{
	using GEO::index_t;
	using Navigation::Index;

	assert(M.vertices.dimension() == 2 || M.vertices.dimension() == 3);

	// Step 0: Make sure all endpoints of E are in V
	std::vector<int> singulars;
	singulars.reserve(V.size() + E.size());
	for (int i = 0; i < V.size(); ++i)
	{
		singulars.push_back(V(i));
	}
	for (int i = 0; i < E.size(); ++i)
	{
		singulars.push_back(E.data()[i]);
	}
	std::sort(singulars.begin(), singulars.end());
	auto it = std::unique(singulars.begin(), singulars.end());
	singulars.resize(std::distance(singulars.begin(), it));

	std::vector<bool> is_singular(M.vertices.nb(), false);
	for (int v : singulars)
	{
		is_singular[v] = true;
	}

	// Step 1: Find all edges around singularities and create new vertices on those edge
	std::vector<int> edge_to_midpoint(M.edges.nb(), -1);
	for (index_t e = 0; e < M.edges.nb(); ++e)
	{
		for (index_t lv = 0; lv < 2; ++lv)
		{
			int v1 = M.edges.vertex(e, lv);
			int v2 = M.edges.vertex(e, (lv + 1) % 2);
			if (is_singular[v1] && !is_singular[v2])
			{
				GEO::vec3 coords = t * mesh_vertex(M, v1) + (1.0 - t) * mesh_vertex(M, v2);
				edge_to_midpoint[e] = mesh_create_vertex(M, coords);
			}
		}
	}

	GEO::Attribute<GEO::index_t> c2e(M.facet_corners.attributes(), "edge_id");

	// Step 2: Iterate over all facets, and subdivide accordingly
	std::vector<index_t> facets_to_delete;
	std::vector<int> next_vertex_around_hole(M.vertices.nb(), -1);
	for (index_t f = 0, old_num_facets = M.facets.nb(); f < old_num_facets; ++f)
	{
		// a. Iterate over edges around the facet, until one marked edge is found
		// b. Place the navigator index so that its vertex points to the unmarked vertex
		// c. Navigate around until another marked edge is found (assert it is different from the previous one)
		// d. Possibly invert the list of vertices / edge midpoints to respect initial facet orientation
		// e. Create new facets given the list + 2 edge midpoints. The rules are:
		//      i. If #v == 1, don't create the midfacet vertex iff the original facet was a triangle
		//     ii. If #v == 2, then don't create the midfacet vertex
		//    iii. If #v >= 3, then create midfacet vertex iff orignal facet was a quad or a tri
		assert(M.facets.nb_vertices(f) > 2);

		// a. Iterate over edges around the facet, until one marked edge is found
		Index idx1 = Navigation::get_index_from_face(M, c2e, f, 0);

		for (index_t lv = 0; lv < M.facets.nb_vertices(f); ++lv)
		{
			if (edge_to_midpoint[idx1.edge] != -1)
			{
				break;
			}
			idx1 = Navigation::next_around_face(M, c2e, idx1);
		}
		if (edge_to_midpoint[idx1.edge] == -1)
		{
			continue;
		}
		facets_to_delete.push_back(f);

		// b. Place the navigator index so that its vertex points to the unmarked vertex
		if (is_singular[idx1.vertex])
		{
			idx1 = Navigation::switch_vertex(M, idx1);
			assert(!is_singular[idx1.vertex]);
		}

		// c. Navigate around until another marked edge is found (assert it is different from the previous one)
		Index idx2 = Navigation::switch_edge(M, c2e, idx1);
		GEO::vector<index_t> unmarked_vertices;
		for (index_t lv = 0; lv < M.facets.nb_vertices(f); ++lv)
		{
			assert(!is_singular[idx2.vertex]);
			unmarked_vertices.push_back(idx2.vertex);
			if (edge_to_midpoint[idx2.edge] != -1)
			{
				break;
			}
			idx2 = Navigation::next_around_face(M, c2e, idx2);
		}
		assert(edge_to_midpoint[idx2.edge] != -1);
		assert(idx1.edge != idx2.edge);
		assert(unmarked_vertices.front() == idx1.vertex);
		assert(unmarked_vertices.back() == idx2.vertex);

		// d. Possibly invert the list of vertices / edge midpoints to respect initial facet orientation
		{
			int v1 = idx2.vertex;
			int v0 = Navigation::switch_vertex(M, idx2).vertex;
			int lv1 = M.facets.find_vertex(f, v1);
			int v2 = M.facets.vertex(f, (lv1 + 1) % M.facets.nb_vertices(f));
			if (v0 != v2)
			{
				// Need to reverse the sequence in `unmarked_vertices`, and swap(idx1, idx2)
				std::reverse(unmarked_vertices.begin(), unmarked_vertices.end());
				std::swap(idx1, idx2);
			}
		}

		// e. Create new facets given the list + 2 edge midpoints. The rules are:
		//      i. If #v == 1, don't create the midfacet vertex iff the original facet was a triangle
		//     ii. If #v == 2, then don't create the midfacet vertex
		//    iii. If #v >= 3, then create midfacet vertex iff orignal facet was a quad or a tri
		int sf = M.facets.nb_vertices(f);
		int nv = (int)unmarked_vertices.size();
		bool create_midfacet_point = false;
		if (nv == 1)
		{
			if (sf == 3)
			{
				create_midfacet_point = false;
			}
			else
			{
				create_midfacet_point = true;
			}
		}
		else if (nv == 2)
		{
			create_midfacet_point = false;
		}
		else
		{
			if (sf == 3 || sf == 4)
			{
				create_midfacet_point = true;
			}
			else
			{
				create_midfacet_point = false;
			}
		}

		int ve1 = edge_to_midpoint[idx1.edge];
		int ve2 = edge_to_midpoint[idx2.edge];
		if (create_midfacet_point)
		{
			GEO::vec3 coords = facet_barycenter(M, f);
			index_t vf = mesh_create_vertex(M, coords);
			next_vertex_around_hole.resize(vf + 1);
			assert(next_vertex_around_hole[ve1] == -1);
			next_vertex_around_hole[ve1] = vf;
			next_vertex_around_hole[vf] = ve2;
			if (nv == 1)
			{
				M.facets.create_quad(unmarked_vertices.front(), ve2, vf, ve1);
			}
			else
			{
				M.facets.create_quad(unmarked_vertices[0], unmarked_vertices[1], vf, ve1);
				M.facets.create_quad(*(unmarked_vertices.rbegin() + 1), unmarked_vertices.back(), ve2, vf);
				for (int i = 1; i < int(unmarked_vertices.size()) - 2; ++i)
				{
					M.facets.create_triangle(unmarked_vertices[i], unmarked_vertices[i + 1], vf);
				}
			}
		}
		else
		{
			assert(next_vertex_around_hole[ve1] == -1);
			next_vertex_around_hole[ve1] = ve2;
			unmarked_vertices.push_back(ve2);
			unmarked_vertices.push_back(ve1);
			M.facets.create_polygon(unmarked_vertices);
		}
	}

	// Step 3: Iterate over new vertices, loop around holes and create polygonal facets
	std::vector<bool> marked(M.vertices.nb(), false);
	for (index_t v = 0; v < M.vertices.nb(); ++v)
	{
		if (next_vertex_around_hole[v] == -1 || marked[v])
		{
			continue;
		}
		GEO::vector<index_t> hole;
		int vi = v;
		do
		{
			vi = next_vertex_around_hole[vi];
			assert(vi != -1);
			hole.push_back(vi);
			marked[vi] = true;
		} while (vi != (int)v);
		M.facets.create_polygon(hole);
	}

	// Step 4: Remove facets that have been split, and delete isolated vertices
	GEO::vector<index_t> to_delete_mask(M.facets.nb(), 0u);
	for (index_t f : facets_to_delete)
	{
		to_delete_mask[f] = 1;
	}
	M.facets.delete_elements(to_delete_mask);
	M.edges.clear();
	M.vertices.remove_isolated();

#if 0
	// Step 2: Find all faces adjacent to a marked edge and place a new vertex at the center of the facet if needed
	std::vector<int> facet_to_midpoint(M.facets.nb(), -1);
	GEO::vector<index_t> facets_to_delete;
	for (index_t f = 0; f < M.facets.nb(); ++f) {
		Index idx = Navigation::get_index_from_face(M, f, 0);
		std::vector<int> marked_edges;
		for (index_t le = 0; le < M.facets.nb_vertices(f); ++le) {
			if (edge_to_midpoint[idx.edge] != -1) {
				marked_edges.push_back(le);
			}
			idx = Navigation::next_around_face(M, idx);
		}
		// There should be exactly 2 split edges per face, otherwise I don't know what to do!
		assert(marked_edges.empty() || marked_edges.size() == 2);
		if (!marked_edges.empty()) {
			int e1 = marked_edges.front();
			int e2 = marked_edges.back();
			if (e1 + 1 == e2 || (e2 + 1) % (int) M.facets.nb_vertices(f) == e1) {
				GEO::vec3 coords = facet_barycenter(f);
				facet_to_midpoint[f] = M.vertices.create_vertex(&coords[0]);
			}
			facets_to_delete.push_back(f);
		}
	}

	// Step 3: Create new facets for each marked facet, based on its configuration
	std::vector<index_t> next_around_hole(M.vertices.nb(), -1);
	std::vector<index_t> prev_around_hole(M.vertices.nb(), -1);

	// Step 4: Remove facets that have been split, and delete isolated vertices
	M.facets.delete_elements(facets_to_delete);
	M.vertices.remove_isolated();
	// + orient facets
#endif
}
