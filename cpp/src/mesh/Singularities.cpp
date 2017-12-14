////////////////////////////////////////////////////////////////////////////////
#include "Singularities.hpp"
#include <algorithm>
////////////////////////////////////////////////////////////////////////////////

void poly_fem::singular_vertices(
	const GEO::Mesh &M, Eigen::VectorXi &V, int regular_degree, bool ignore_border)
{
	using GEO::index_t;
	std::vector<int> degree(M.vertices.nb(), 0);
	for (index_t f = 0; f < M.facets.nb(); ++f) {
		for (index_t lv = 0; lv < M.facets.nb_vertices(f); ++lv) {
			index_t v = M.facets.vertex(f, lv);
			degree[v]++;
		}
	}

	// Ignore border vertices if requested
	if (ignore_border) {
		GEO::Attribute<int> boundary_vertices(M.vertices.attributes(), "boundary_vertex");
		for (index_t v = 0; v < M.vertices.nb(); ++v) {
			if (boundary_vertices[v]) {
				degree[v] = regular_degree;
			}
		}
	}

	// Count and create list of irregular vertices
	long nb_reg = std::count(degree.begin(), degree.end(), regular_degree);
	V.resize(M.vertices.nb() - nb_reg);
	for (index_t v = 0, i = 0; v < M.vertices.nb(); ++v) {
		if (degree[v] != regular_degree) {
			assert(i < V.size());
			V(i++) = v;
		}
	}
}

void poly_fem::singular_edges(const GEO::Mesh &M, const Eigen::VectorXi &V, Eigen::MatrixX2i &E) {
	using GEO::index_t;
	std::vector<bool> is_singular(M.vertices.nb(), false);
	for (int i = 0; i < V.size(); ++i) {
		is_singular[V(i)] = true;
	}
	std::vector<std::pair<int, int>> edges;
	for (index_t e = 0; e < M.edges.nb(); ++e) {
		index_t v1 = M.edges.vertex(e, 0);
		index_t v2 = M.edges.vertex(e, 1);
		if (is_singular[v1] && is_singular[v2]) {
			edges.emplace_back(v1, v2);
		}
	}
	E.resize(edges.size(), 2);
	int cnt = 0;
	for (const auto &kv : edges) {
		assert(cnt < E.rows());
		E.row(cnt++) << kv.first, kv.second;
	}
}

void poly_fem::singularity_graph(
	const GEO::Mesh &M, Eigen::VectorXi &V, Eigen::MatrixX2i &E,int regular_degree, bool ignore_border)
{
	singular_vertices(M, V, regular_degree, ignore_border);
	singular_edges(M, V, E);
}

////////////////////////////////////////////////////////////////////////////////

void poly_fem::create_patch_around_singularities(
	GEO::Mesh &M, const Eigen::VectorXi &V, const Eigen::MatrixX2i &E, double t)
{
	// TODO

	// Step 0: Make sure all endpoints of E are in V
	std::vector<int> v;
	v.reserve(V.size() + E.size());
	for (int i = 0; i < V.size(); ++i) { v.push_back(V(i)); }

	// Step 1: Find all edges around and place a new vertex at the center of the edge
	// Step 2: Find all faces and place a new vertex at the center of the facet if needed
	// Step 3: Create new facets for each existing facet, based on its configuration
}
