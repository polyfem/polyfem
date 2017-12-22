////////////////////////////////////////////////////////////////////////////////
#include "Refinement.hpp"
#include "Navigation.hpp"
#include "PolygonUtils.hpp"
#include <igl/is_border_vertex.h>
#include <igl/remove_duplicate_vertices.h>
#include <igl/remove_unreferenced.h>
#include <iostream>
#include <vector>
#include <array>
////////////////////////////////////////////////////////////////////////////////

void poly_fem::edge_adjacency_graph(
	const Eigen::MatrixXi &Q, Eigen::MatrixXi &edge_index,
	std::vector<std::vector<int>> &adj,
	std::vector<std::pair<int, int>> *pairs_of_edges,
	std::vector<std::pair<int, int>> *pairs_of_quads,
	Eigen::MatrixXi *quad_index)
{
	assert(Q.cols() == 4); // Assumes quad mesh
	typedef std::pair<int, int> Edge;

	std::vector<std::pair<int, int>> edges;

	edge_index.resizeLike(Q); // (f, lv) -> edge id
	adj.clear();
	edges.clear();
	if (pairs_of_quads) { pairs_of_quads->clear(); }
	if (quad_index) { quad_index->setConstant(Q.rows(), Q.cols(), -1); }

	// Number mesh edges & build dual edge-graph
	std::vector<std::pair<Edge, std::pair<int, int>>> accu;
	accu.reserve((size_t) Q.size());
	for (int f = 0; f < Q.rows(); ++f) {
		for (int lv = 0; lv < 4; ++lv) {
			int x = Q(f, lv);
			int y = Q(f, (lv+1)%4);
			accu.emplace_back(Edge(std::min(x, y), std::max(x, y)), std::make_pair(f, lv));
		}
	}
	std::sort(accu.begin(), accu.end());
	edges.reserve(accu.size() / 2);
	edge_index.setConstant(-1);
	int id = -1;
	int prev_quad = -1;
	int prev_lv = -1;
	Edge prev_side(-1, -1);
	for (const auto &kv : accu) {
		// kv = (v1, v2), (f, lv)
		Edge e = kv.first;
		int f = kv.second.first;
		int lv = kv.second.second;
		// printf("(%d,%d) -> %d, %d\n", e.first, e.second, f, lv);
		if (e != prev_side) {
			++id;
			edges.emplace_back(e);
		} else {
			if (pairs_of_quads) {
				if (quad_index) {
					(*quad_index)(f, lv) = prev_quad;
					(*quad_index)(prev_quad, prev_lv) = f;
				}
				pairs_of_quads->emplace_back(prev_quad, f);
			}
		}
		edge_index(f, lv) = id;
		prev_side = e;
		prev_quad = f;
		prev_lv = lv;
	}
	adj.resize(edges.size());
	for (int f = 0; f < Q.rows(); ++f) {
		for (int lv = 0; lv < 4; ++lv) {
			int e1 = edge_index(f, lv);
			int e2 = edge_index(f, (lv+2)%4);
			assert(e1 != -1);
			if (e1 < e2) {
				adj[e1].push_back(e2);
				adj[e2].push_back(e1);
			}
		}
	}

	if (pairs_of_edges) {
		std::swap(*pairs_of_edges, edges);
	}
}

////////////////////////////////////////////////////////////////////////////////

namespace {

enum class PatternType {
	NOT_PERIODIC,    // Opposite borders do not match
	NOT_SYMMETRIC,   // At least one border is not symmetric along its bisector
	SIMPLE_PERIODIC, // Opposite borders match individually
	DOUBLE_PERIODIC, // All four borders match between each others
};

// -----------------------------------------------------------------------------

// Determine the type of periodicity of the given pattern. This function rely
// only on the input vertex positions, but does not check if the underlying
// topology is consistent.
//
// @param[in]  V                { #V x 3 matrix of vertex positions }
// @param[in]  F                { #F x 3 matrix of triangle indexes }
// @param[out] border_vertices  { List of vertices along each side }
//
// @return     { Type of periodicity of the pattern. }
//
PatternType compute_pattern_type(
	const Eigen::MatrixXd &V, const Eigen::MatrixXi &F,
	std::array<Eigen::VectorXi, 4> &border_vertices)
{
	Eigen::Vector2d lower = V.colwise().minCoeff().head<2>();
	Eigen::Vector2d upper = V.colwise().maxCoeff().head<2>();

	std::array<Eigen::MatrixXd, 2> border;

	// Compute vertices along the border. Sides are numbered as follows:
	//
	//        2
	//   v3 ----- v2
	//   |        |
	// 3 |        | 1
	//   |        |
	//   v0 ----- v1
	//        0
	// ↑y
	// └─→x
	for (int d = 0; d < 2; ++d) {
		std::vector<std::pair<double, int>> low, upp;
		for (int i = 0; i < V.rows(); ++i) {
			if (V(i, 1-d) == lower[1-d]) { low.emplace_back(V(i, d), i); }
			if (V(i, 1-d) == upper[1-d]) { upp.emplace_back(V(i, d), i); }
		}
		std::sort(low.begin(), low.end());
		std::sort(upp.begin(), upp.end());
		border_vertices[d?3:0].resize(low.size());
		for (size_t i = 0; i < low.size(); ++i) {
			border_vertices[d?3:0][i] = low[i].second;
		}
		border_vertices[d?1:2].resize(upp.size());
		for (size_t i = 0; i < upp.size(); ++i) {
			border_vertices[d?1:2][i] = upp[i].second;
		}
	}
	for (int i : {2, 3}) {
		border_vertices[i].reverseInPlace();
	}

	// Check if borders have the same topology and geometry
	for (int d = 0; d < 2; ++d) {
		std::vector<double> low, upp;
		for (int i = 0; i < V.rows(); ++i) {
			if (V(i, d) == lower[d]) { low.push_back(V(i, 1-d)); }
			if (V(i, d) == upper[d]) { upp.push_back(V(i, 1-d)); }
		}
		std::sort(low.begin(), low.end());
		std::sort(upp.begin(), upp.end());
		if (low.size() != upp.size()) {
			return PatternType::NOT_PERIODIC;
		}
		size_t n = low.size();
		border[d].resize(n, 2);
		for (size_t i = 0; i < n; ++i) {
			border[d](i, 0) = low[i];
			border[d](i, 1) = upp[i];
		}
		if (!border[d].col(0).isApprox(border[d].col(1))) {
			return PatternType::NOT_PERIODIC;
		}
	}

	// Check if borders are symmetric
	for (int d = 0; d < 2; ++d) {
		if (!(border[d].col(0).array() - lower[d]).isApprox(
			upper[d] - border[d].col(0).array().reverse()))
		{
			return PatternType::NOT_SYMMETRIC;
		}
	}

	// Check if horizontal and vertical borders are matching
	if (border[0].size() == border[1].size()) {
		if (!border[0].col(0).isApprox(border[1].col(0))) {
			std::cerr << "Warning: pattern boundaries have the same number of vertices, but their position differ slighly." << std::endl;
			Eigen::MatrixXd X(border[0].size(), 2);
			X.col(0) = border[0].col(0);
			X.col(1) = border[1].col(0);
			std::cout << X << std::endl << std::endl;;
		}
		return PatternType::DOUBLE_PERIODIC;
	} else {
		return PatternType::SIMPLE_PERIODIC;
	}

	return PatternType::SIMPLE_PERIODIC;
}

// -----------------------------------------------------------------------------

Eigen::VectorXi vertex_degree(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F) {
	Eigen::VectorXi count = Eigen::VectorXi::Zero(V.rows());

	for (unsigned i=0; i<F.rows();++i) {
		for (unsigned j=0; j<F.cols();++j) {
			// Avoid duplicate edges
			if (F(i,j) < F(i,(j+1)%F.cols())) {
				count(F(i,j  )) += 1;
				count(F(i,(j+1)%F.cols())) += 1;
			}
		}
	}

	return count;
}

} // anonymous namespace

////////////////////////////////////////////////////////////////////////////////

bool poly_fem::instanciate_pattern(
	const Eigen::MatrixXd &IV, const Eigen::MatrixXi &IF,
	const Eigen::MatrixXd &PV, const Eigen::MatrixXi &PF,
	Eigen::MatrixXd &OV, Eigen::MatrixXi &OF)
{
	// Copy input quads (may be reordered)
	Eigen::MatrixXi IQ = IF;

	// List of vertices along each border (from lv to (lv+1)%4)
	std::array<Eigen::VectorXi, 4> border_vertices;

	auto pattern = compute_pattern_type(PV, PF, border_vertices);
	auto valence = vertex_degree(IV, IQ);
	auto border = igl::is_border_vertex(IV, IQ);

	// Check input validity
	switch (pattern) {
	case PatternType::NOT_PERIODIC:
		std::cerr << "Pattern not periodic" << std::endl;
		return false;
	case PatternType::NOT_SYMMETRIC:
		std::cerr << "Pattern not symmetric" << std::endl;
		return false;
	case PatternType::SIMPLE_PERIODIC:
		std::cerr << "Pattern simple periodic" << std::endl;
		return false;
	case PatternType::DOUBLE_PERIODIC:
		// std::cout << "Pattern double periodic" << std::endl;
		// Not implemented
		break;
	default:
		std::cerr << "Unknown patter type" << std::endl;
		return false;
	}

	// Normalized coordinates (between 0 and 1 for barycentric coordinates)
	auto lower = PV.colwise().minCoeff();
	auto upper = PV.colwise().maxCoeff();
	Eigen::MatrixXd PVN = (PV.rowwise() - lower).array().rowwise() / (upper - lower).cwiseMax(1e-5).array();

	// Number edges of the input quad mesh
	Eigen::MatrixXi edge_index;
	std::vector<std::vector<int>> adj;
	std::vector<std::pair<int, int>> pairs_of_edges;
	std::vector<std::pair<int, int>> pairs_of_quads;
	edge_adjacency_graph(IQ, edge_index, adj, &pairs_of_edges, &pairs_of_quads);

	// Instantiate (duplicating vertices)
	Eigen::MatrixXd V(IQ.rows() * PV.rows(), 3);
	Eigen::MatrixXi F(IQ.rows() * PF.rows(), PF.cols());

	// Remapped vertex id (after duplicate removal)
	Eigen::VectorXi remap = Eigen::VectorXi::LinSpaced((int) V.rows(), 0, (int) V.rows()-1);

	for (int q = 0; q < IQ.rows(); ++q) {
		const auto & u = PVN.col(0).array();
		const auto & v = PVN.col(1).array();
		Eigen::RowVector3d a = IV.row(IQ(q, 0));
		Eigen::RowVector3d b = IV.row(IQ(q, 1));
		Eigen::RowVector3d c = IV.row(IQ(q, 2));
		Eigen::RowVector3d d = IV.row(IQ(q, 3));
		int v0 = (int) PVN.rows() * q;
		V.middleRows(v0, PVN.rows()) = ((1-u)*(1-v)).matrix()*a
			+ (u*(1-v)).matrix()*b
			+ (u*v).matrix()*c
			+ ((1-u)*v).matrix()*d;
		int f0 = (int) PF.rows() * q;
		F.middleRows(f0, PF.rows()) = PF.array() + v0;
	}

	auto min_max = [](int x, int y) {
		return std::make_pair(std::min(x, y), std::max(x, y));
	};

	// Stitch vertices from adjacent quads
	for (const auto &qq : pairs_of_quads) {
		int q1 = qq.first;
		int q2 = qq.second;
		if (q2 > q1) { std::swap(q1, q2); }
		const int v1 = (int) PVN.rows() * q1;
		const int v2 = (int) PVN.rows() * q2;
		int lv1 = 0;
		int lv2 = 0;
		for (; lv1 < 4; ++lv1) {
			int x1 = IQ(q1, lv1);
			int y1 = IQ(q1, (lv1+1)%4);
			for (lv2 = 0; lv2 < 4; ++lv2) {
				int x2 = IQ(q2, lv2);
				int y2 = IQ(q2, (lv2+1)%4);
				if (min_max(x1, y1) == min_max(x2, y2)) {
					int e = edge_index(q1, lv1);
					// tfm::printf("quads: (%s, %s) and (%s, %s)\n", q1, lv1, q2, lv2);
					// tfm::printf("└ edge: %s-%s (id: %e)\n", x1, y1, e);
					assert(edge_index(q2, lv2) == e);
					Eigen::VectorXi side1 = border_vertices[lv1].array() + v1;
					Eigen::VectorXi side2 = border_vertices[lv2].array() + v2;
					if (x1 > y1) { side1.reverseInPlace(); }
					if (x2 > y2) { side2.reverseInPlace(); }
					for (int ii = 0; ii < (int) side1.size(); ++ii) {
						remap(side2[ii]) = side1[ii];
					}
					lv1 = 4; lv2 = 4;
				}
			}
		}
	}

	// Remap vertices according to 'remap' + remove unreferenced vertices
	// We also average remapped vertices positions, in case they mismatch too much
	for (int f = 0; f < F.rows(); ++f) {
		for (int lv = 0; lv < F.cols(); ++lv) {
			int ov = F(f, lv);
			int nv = remap(ov);
			Eigen::RowVector3d p_avg = 0.5 * (V.row(ov) + V.row(nv));
			F(f, lv) = nv;
			V.row(nv) = p_avg;
		}
	}
	Eigen::VectorXi I;
	igl::remove_unreferenced(V, F, OV, OF, I);

	// Remove duplicate vertices
	// Eigen::MatrixXi OVI, OVJ;
	// igl::remove_duplicate_vertices(V, F, 0.0, OV, OVI, OVJ, OF);

	return true;
}

////////////////////////////////////////////////////////////////////////////////

void poly_fem::refine_quad_mesh(
	const Eigen::MatrixXd &IV, const Eigen::MatrixXi &IF,
	Eigen::MatrixXd &OV, Eigen::MatrixXi &OF)
{
	Eigen::MatrixXd PV(9, 3);
	Eigen::MatrixXi PF(4, 4);
	PV <<
		-1, 0, 0,
		0, 0, 0,
		0, 1, 0,
		-1, 1, 0,
		1, 1, 0,
		1, 0, 0,
		1, -1, 0,
		0, -1, 0,
		-1, -1, 0;
	PF <<
		1, 2, 3, 4,
		3, 2, 6, 5,
		6, 2, 8, 7,
		2, 1, 9, 8;
	PF = PF.array() - 1;
	// std::cout << PF << std::endl;
	bool res = instanciate_pattern(IV, IF, PV, PF, OV, OF);
	assert(res);
}

////////////////////////////////////////////////////////////////////////////////

void poly_fem::polar_split(const Eigen::MatrixXd &IV, Eigen::MatrixXd &OV, std::vector<std::vector<int>> &OF, double t) {
	assert(IV.cols() == 2 || IV.cols() == 3);
	Eigen::RowVector3d bary;
	if (is_star_shaped(IV, bary)) {
		// Create 1-ring around the central point
		OV.resize(2 * IV.rows(), IV.cols());
		OV.topRows(IV.rows()) = IV;
		OV.bottomRows(IV.rows()) = (t * IV).rowwise() + (1.0 - t) * bary.head(IV.cols());
		int n = (int) IV.rows();
		OF.push_back({});
		for (int i = 0; i < IV.rows(); ++i) {
			OF.front().push_back(i + n);
			OF.push_back({i, (i+1)%n, (i+1)%n + n, i + n});
		}
	} else {
		throw std::invalid_argument("Non star-shaped input polygon.");
	}
}

////////////////////////////////////////////////////////////////////////////////

void poly_fem::refine_polygonal_mesh(const GEO::Mesh &M_in, GEO::Mesh &M_out, bool refine_polygons, double t) {
	using GEO::index_t;
	using Navigation::Index;

	// [Helper] Retrieve vertex position
	auto mesh_point = [] (const GEO::Mesh &M, index_t v) {
		GEO::vec3 p(0, 0, 0);
		for (index_t d = 0; d < std::min(3u, (index_t) M.vertices.dimension()); ++d) {
			if (M.vertices.double_precision()) {
				p[d] = M.vertices.point_ptr(v)[d];
			} else {
				p[d] = M.vertices.single_precision_point_ptr(v)[d];
			}
		}
		return p;
	};

	// [Helper] Create a new vertex and set vertex coordinates
	auto mesh_create_vertex = [] (GEO::Mesh &M, const GEO::vec3 &p) {
		auto v = M.vertices.create_vertex();
		for (index_t d = 0; d < std::min(3u, (index_t) M.vertices.dimension()); ++d) {
			if (M.vertices.double_precision()) {
				M.vertices.point_ptr(v)[d] = p[d];
			} else {
				M.vertices.single_precision_point_ptr(v)[d] = (float) p[d];
			}
		}
		return v;
	};

	// [Helper] Retrieve facet barycenter
	auto facet_barycenter = [&mesh_point](const GEO::Mesh &M, index_t f) {
		GEO::vec3 p(0, 0, 0);
		for (index_t lv = 0; lv < M.facets.nb_vertices(f); ++lv) {
			p += mesh_point(M, M.facets.vertex(f, lv));
		}
		return p / M.facets.nb_vertices(f);
	};

	// Step 0: Clear output mesh, and fill it with M_in's vertices
	assert(&M_in != &M_out);
	M_out.copy(M_in);
	M_out.edges.clear();
	M_out.facets.clear();

	// Step 1: Chain vertices around polygonal facets
	std::vector<int> next_vertex_around_hole(M_in.vertices.nb(), -1);
	for (index_t f = 0; f < M_in.facets.nb(); ++f) {
		if (M_in.facets.nb_vertices(f) > 4) {
			for (index_t lv = 0; lv < M_in.facets.nb_vertices(f); ++lv) {
				index_t v1 = M_in.facets.vertex(f, lv);
				index_t v2 = M_in.facets.vertex(f, (lv+1)%M_in.facets.nb_vertices(f));
				next_vertex_around_hole[v1] = v2;
			}
		}
	}

	// Step 2: Iterate over facets and refine triangles and quads
	std::vector<int> edge_to_midpoint(M_in.edges.nb(), -1);
	for (index_t f = 0; f < M_in.facets.nb(); ++f) {
		index_t nv = M_in.facets.nb_vertices(f);
		assert(nv > 2);
		if (nv == 3 || nv == 4) {
			Index idx = Navigation::get_index_from_face(M_in, f, 0);
			assert(Navigation::switch_vertex(M_in, idx).vertex == (int) M_in.facets.vertex(f, 1));
			// Create mid-edge vertices
			for (index_t lv = 0; lv < M_in.facets.nb_vertices(f); ++lv, idx = Navigation::next_around_face(M_in, idx)) {
				assert(idx.vertex == (int) M_in.facets.vertex(f, lv));
				if (edge_to_midpoint[idx.edge] == -1) {
					GEO::vec3 coords = 0.5 * (mesh_point(M_in, idx.vertex) + mesh_point(M_in, Navigation::switch_vertex(M_in, idx).vertex));
					edge_to_midpoint[idx.edge] = mesh_create_vertex(M_out, coords);
					assert(edge_to_midpoint[idx.edge] + 1 == (int) M_out.vertices.nb());
					next_vertex_around_hole.push_back(-1);
				}
				// Chain vertices around holes
				int v1 = idx.vertex;
				int v2 = Navigation::switch_vertex(M_in, idx).vertex;
				if (next_vertex_around_hole[v2] == v1) {
					// printf("v1: %d, v2: %d, v12: %d, n(v1): %d, n(v2): %d\n",
					// 	v1, v2, edge_to_midpoint[idx.edge], next_vertex_around_hole[v1], next_vertex_around_hole[v2]);
					int v12 = edge_to_midpoint[idx.edge];
					next_vertex_around_hole[v2] = v12;
					next_vertex_around_hole[v12] = v1;
				}
			}
			// Create mid-face vertex
			index_t vf = mesh_create_vertex(M_out, facet_barycenter(M_in, f));
			assert(vf + 1 == M_out.vertices.nb());
			next_vertex_around_hole.push_back(-1);
			// Create quads
			for (index_t lv = 0; lv < M_in.facets.nb_vertices(f); ++lv) {
				idx = Navigation::get_index_from_face(M_in, f, lv);
				assert(Navigation::switch_vertex(M_in, idx).vertex == (int) M_in.facets.vertex(f, (lv+1)%M_in.facets.nb_vertices(f)));
				int v1 = idx.vertex;
				int v12 = edge_to_midpoint[idx.edge];
				int v01 = edge_to_midpoint[Navigation::switch_edge(M_in, idx).edge];
				assert(v12 != -1 && v01 != -1);
				M_out.facets.create_quad(v1, v12, vf, v01);
			}
		}
	}

	// Step 3: Create polygonal faces following vertices around holes
	std::vector<bool> marked(M_out.vertices.nb(), false);
	for (index_t v = 0, vmax = M_out.vertices.nb(); v < vmax; ++v) {
		if (next_vertex_around_hole[v] == -1 || marked[v]) {
			continue;
		}
		GEO::vector<index_t> hole;
		int vi = v;
		do {
			// std::cout << vi << ';';
			vi = next_vertex_around_hole[vi];
			assert(vi != -1);
			hole.push_back(vi);
			marked[vi] = true;
		} while (vi != (int) v);
		// std::cout << std::endl;

		// Record facet
		if (refine_polygons) {
			if (M_in.vertices.dimension() != 2) {
				std::cerr << "WARNING: Input mesh has dimension > 2, but polygonal facets will be split considering their XY coordinates only." << std::endl;
			}

			// Subdivide the hole using polar refinement
			index_t n = hole.size();
			Eigen::MatrixXd P(n, 2), V;
			std::vector<std::vector<int>> F;
			for (index_t k = 0; k < n; ++k) {
				GEO::vec3 pk = mesh_point(M_out, hole[k]);
				P.row(k) << pk[0], pk[1];
			}
			polar_split(P, V, F, t);
			std::vector<int> remap(n);
			assert(V.rows() == 2*n);
			for (index_t k = 0; k < n; ++k) {
				GEO::vec3 qk = GEO::vec3(V(k+n, 0), V(k+n, 1), 0);
				remap[k] = mesh_create_vertex(M_out, qk);
			}
			for (const auto &poly : F) {
				GEO::vector<index_t> vertices;
				for (int vk : poly) {
					if (vk < (int) n) { vertices.push_back(hole[vk]); }
					else { vertices.push_back(remap[vk - n]); }
					// std::cout << vertices.back() << ',';
				}
				// std::cout << std::endl;
				M_out.facets.create_polygon(vertices);
			}

		} else {
			// Keep the hole as-is
			M_out.facets.create_polygon(hole);
		}
	}
}

