#include "Singularities.hpp"
#include <algorithm>

void poly_fem::singular_vertices(const GEO::Mesh &M, Eigen::VectorXi &V, int regular_degree) {
	using GEO::index_t;
	std::vector<int> degree(M.vertices.nb(), 0);
	for (index_t f = 0; f < M.facets.nb(); ++f) {
		for (index_t lv = 0; lv < M.facets.nb_vertices(f); ++lv) {
			index_t v = M.facets.vertex(f, lv);
			degree[v]++;
		}
	}
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

void poly_fem::singularity_graph(const GEO::Mesh &M, Eigen::VectorXi &V, Eigen::MatrixX2i &E, int regular_degree) {
	singular_vertices(M, V, regular_degree);
	singular_edges(M, V, E);
}
