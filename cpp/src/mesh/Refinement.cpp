////////////////////////////////////////////////////////////////////////////////
#include "Refinement.hpp"
#include <igl/is_border_vertex.h>
#include <igl/remove_duplicate_vertices.h>
#include <igl/remove_unreferenced.h>
#include <iostream>
#include <vector>
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
