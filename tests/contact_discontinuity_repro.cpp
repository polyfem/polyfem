// Standalone repro for an IPC ESP gradient discontinuity.
// Loads a dump produced by ESPContactForm's ESP_CONTACT_DUMP_DIR
// hook (rest_V.bin, rest_E.bin, rest_F.bin, displaced_{lo,hi}.bin,
// params_{lo,hi}.json), rebuilds the collision set + potential at α_lo
// and α_hi, computes value+gradient, prints diff per-collision.
//
// Usage:
//   contact_discontinuity_repro <dump_dir>
//
// Notes:
//   * params.json's barrier_type is informational; this binary uses
//     ipc-toolkit's default barrier (set by ESPParameters).
//     If the discontinuity reproduces, the barrier choice is irrelevant
//     to localization.

#include <Eigen/Dense>
#include <ipc/collision_mesh.hpp>
#include <ipc/esp/esp_collisions.hpp>
#include <ipc/esp/esp_potential.hpp>
#include <ipc/esp/esp_parameters.hpp>
#include <ipc/esp/collisions/esp_collision.hpp>
#include <ipc/esp/collisions/vertex_matrix_view.hpp>
#include <ipc/distance/distance_type.hpp>
#include <ipc/distance/point_triangle.hpp>
#include <ipc/distance/point_edge.hpp>
#include <ipc/distance/edge_edge.hpp>
#include <ipc/barrier/barrier.hpp>
#include <algorithm>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <unordered_set>
#include <utility>
#include <polyfem/quadrature/TriQuadrature.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <string>

namespace
{

	template <typename Scalar>
	Eigen::Matrix<Scalar, -1, -1> read_mat_bin(const std::string &path)
	{
		std::ifstream f(path, std::ios::binary);
		if (!f)
			throw std::runtime_error("cannot open " + path);
		int32_t r, c;
		f.read((char *)&r, sizeof(int32_t));
		f.read((char *)&c, sizeof(int32_t));
		Eigen::Matrix<Scalar, -1, -1> M(r, c);
		for (int i = 0; i < r; i++)
			for (int j = 0; j < c; j++)
			{
				Scalar v;
				f.read((char *)&v, sizeof(Scalar));
				M(i, j) = v;
			}
		return M;
	}

	// Lightweight scalar extraction from the params JSON without pulling in
	// nlohmann/json's parser (we control the writer's format).
	double get_num(const std::string &body, const std::string &key, double def = 0.0)
	{
		std::regex r("\"" + key + "\"\\s*:\\s*([^,\\n}]+)");
		std::smatch m;
		if (std::regex_search(body, m, r))
			return std::stod(m[1].str());
		return def;
	}

	bool get_bool(const std::string &body, const std::string &key, bool def = false)
	{
		std::regex r("\"" + key + "\"\\s*:\\s*(true|false)");
		std::smatch m;
		if (std::regex_search(body, m, r))
			return m[1].str() == "true";
		return def;
	}

	ipc::FaceQuadRule build_quad_rule(int order)
	{
		if (order <= 0)
			return {};
		polyfem::quadrature::Quadrature q;
		polyfem::quadrature::TriQuadrature(true).get_quadrature(order, q);
		ipc::FaceQuadRule rule;
		rule.reserve(q.size());
		for (int i = 0; i < q.size(); i++)
		{
			double x = q.points(i, 0), y = q.points(i, 1), w = q.weights(i) * 2.0;
			rule.push_back({{{1.0 - x - y, x, y}}, w});
		}
		return rule;
	}

	struct DumpState
	{
		Eigen::MatrixXd displaced;
		double dhat = 0, dbar_factor = 1, barrier_stiffness = 1;
		int quad_order = 1, integration_type = 1;
		bool ogc_collisions = false, area_weights = true;
	};

	DumpState load_state(const std::string &dir, const std::string &tag)
	{
		DumpState s;
		s.displaced = read_mat_bin<double>(dir + "/displaced_" + tag + ".bin");
		std::ifstream pf(dir + "/params_" + tag + ".json");
		std::stringstream ss;
		ss << pf.rdbuf();
		std::string body = ss.str();
		s.dhat = get_num(body, "dhat");
		s.dbar_factor = get_num(body, "dbar_factor", 1.0);
		s.quad_order = (int)get_num(body, "quad_order", 1);
		s.integration_type = (int)get_num(body, "integration_type", 1);
		s.barrier_stiffness = get_num(body, "barrier_stiffness", 1.0);
		s.ogc_collisions = get_bool(body, "ogc_collisions", false);
		s.area_weights = get_bool(body, "area_weights", true);
		return s;
	}

	ipc::ESPParameters make_params(const DumpState &s)
	{
		// NOTE: ogc_collisions was removed from ESPParameters upstream; the field is
		// still parsed from the dump for format compatibility but no longer passed.
		ipc::ESPParameters p(
			s.dhat, s.dbar_factor, s.quad_order, s.area_weights,
			static_cast<ipc::ESPParameters::IntegrationType>(s.integration_type));
		p.face_quad_rule = build_quad_rule(s.quad_order);
		// Barrier: leave default (NormalizedClampedLogBarrier per the struct).
		// Caller can edit here to match production runs more precisely.
		return p;
	}

	struct Side
	{
		ipc::ESPParameters params;
		ipc::ESPCollisions collisions;
		std::unique_ptr<ipc::ESPPotential> pot;
		double value = 0;
		Eigen::VectorXd grad;
		Side(const DumpState &s, const ipc::CollisionMesh &mesh)
			: params(make_params(s))
		{
			collisions.build(mesh, s.displaced, params);
			pot = std::make_unique<ipc::ESPPotential>(params, /*use_near_far=*/true);
			value = (*pot)(collisions, mesh, s.displaced);
			grad = pot->gradient(collisions, mesh, s.displaced);
		}
	};

} // namespace

int main(int argc, char **argv)
{
#ifndef POLYFEM_TEST_DIR
#define POLYFEM_TEST_DIR "."
#endif
	const std::string dir = (argc >= 2)
								? std::string(argv[1])
								: std::string(POLYFEM_TEST_DIR) + "/data/esp_contact_discontinuity";
	std::cout << "[dump_dir] " << dir << "\n";

	Eigen::MatrixXd V = read_mat_bin<double>(dir + "/rest_V.bin");
	Eigen::MatrixXi E = read_mat_bin<int>(dir + "/rest_E.bin");
	Eigen::MatrixXi F = read_mat_bin<int>(dir + "/rest_F.bin");
	std::cout << "[rest] V=" << V.rows() << "x" << V.cols()
			  << " E=" << E.rows() << " F=" << F.rows() << "\n";

	ipc::CollisionMesh mesh(V, E, F);

	DumpState lo = load_state(dir, "lo");
	DumpState hi = load_state(dir, "hi");

	std::cout << std::scientific;
	std::cout.precision(10);

	Side L(lo, mesh), H(hi, mesh);
	std::cout << "[lo] ncollisions=" << L.collisions.size()
			  << " value=" << L.value << " ||grad||=" << L.grad.norm() << "\n";
	std::cout << "[hi] ncollisions=" << H.collisions.size()
			  << " value=" << H.value << " ||grad||=" << H.grad.norm() << "\n";

	const double ddisp = (hi.displaced - lo.displaced).norm();
	std::cout << "||Δdisplaced|| = " << ddisp << "\n";
	if (L.grad.size() != H.grad.size())
	{
		std::cerr << "Gradient sizes differ; cannot diff.\n";
		return 1;
	}
	const Eigen::VectorXd dg = H.grad - L.grad;
	std::cout << "||Δgrad||      = " << dg.norm() << "\n";

	// (1) Per-vertex Δgrad: identify suspect vertices.
	const int dim = (int)mesh.dim();
	const int nv = (int)mesh.num_vertices();
	Eigen::VectorXd dg_per_v(nv);
	for (int v = 0; v < nv; ++v)
		dg_per_v(v) = dg.segment(v * dim, dim).norm();
	std::vector<int> order(nv);
	std::iota(order.begin(), order.end(), 0);
	std::partial_sort(order.begin(), order.begin() + std::min(20, nv), order.end(),
					  [&](int a, int b) { return dg_per_v(a) > dg_per_v(b); });

	std::cout << "\n=== Top vertices by ||Δg_v|| ===\n";
	const int K = std::min(15, nv);
	std::unordered_set<int> suspect;
	const double cutoff = std::max(1e-12, dg_per_v(order[0]) * 1e-3);
	for (int k = 0; k < K; ++k)
	{
		int v = order[k];
		std::cout << "  v=" << v << "  ||Δg_v||=" << dg_per_v(v) << "\n";
		if (dg_per_v(v) >= cutoff)
			suspect.insert(v);
	}

	// (2) Iterate dicts in ESPCollisions and pair lo↔hi by their key
	//     (vertex id / edge-edge pair / face id+qp). For each dict that
	//     touches a suspect vertex (via its primary_vertex_ids), report
	//     size and per-sub-collision typed hashes + distances at lo & hi.
	using HashT = std::array<ipc::index_t, 3>;
	auto hash_str = [](const HashT &h) {
		return "(" + std::to_string(h[0]) + "," + std::to_string(h[1]) + "," + std::to_string(h[2]) + ")";
	};

	auto touches_suspect_dict = [&](const auto &dict) {
		for (auto vid : dict.primary_vertex_ids())
		{
			if (vid >= 0 && (int)vid < nv && suspect.count((int)vid))
				return true;
		}
		return false;
	};

	// Returns sub-collision hashes for a dict (in iteration order).
	auto dict_hashes = [](const auto &dict) {
		std::vector<HashT> out;
		out.reserve(dict.size());
		for (int i = 0; i < dict.size(); ++i)
			out.push_back(dict[i].get_typed_hash());
		return out;
	};

	auto print_subcollisions = [&](const auto &dict, const Eigen::MatrixXd &X_lo, const Eigen::MatrixXd &X_hi) {
		for (int i = 0; i < dict.size(); ++i)
		{
			const auto &c = dict[i];
			HashT h = c.get_typed_hash();
			double d_lo = c.compute_distance(X_lo);
			double d_hi = c.compute_distance(X_hi);
			std::cout << "      [" << i << "] " << c.name() << " hash=" << hash_str(h)
					  << " d_lo=" << d_lo << " d_hi=" << d_hi
					  << "  vid=[";
			for (int j = 0; j < c.num_vertices(); ++j)
			{
				if (j)
					std::cout << ",";
				std::cout << c.vertex_id(j);
			}
			std::cout << "]\n";
		}
	};

	// Generic dict-pair printer (works for any ESPCollisionDict<...>).
	auto report_dict_pair = [&](const std::string &key,
								const auto *dict_lo, const auto *dict_hi) {
		const int s_lo = dict_lo ? dict_lo->size() : 0;
		const int s_hi = dict_hi ? dict_hi->size() : 0;
		std::vector<HashT> h_lo = dict_lo ? dict_hashes(*dict_lo) : std::vector<HashT>();
		std::vector<HashT> h_hi = dict_hi ? dict_hashes(*dict_hi) : std::vector<HashT>();
		std::set<HashT> set_lo(h_lo.begin(), h_lo.end()), set_hi(h_hi.begin(), h_hi.end());
		const bool member_diff = set_lo != set_hi;
		std::cout << "  " << key << "  size_lo=" << s_lo << " size_hi=" << s_hi
				  << (member_diff ? "  ** MEMBERSHIP DIFF **" : "")
				  << "  presence=" << (dict_lo ? 'L' : '-') << (dict_hi ? 'H' : '-') << "\n";
		if (dict_lo)
		{
			std::cout << "    [lo subs]\n";
			print_subcollisions(*dict_lo, lo.displaced, hi.displaced);
		}
		if (dict_hi)
		{
			std::cout << "    [hi subs]\n";
			print_subcollisions(*dict_hi, lo.displaced, hi.displaced);
		}
	};

	std::cout << "\n=== Dicts touching suspect vertices ===\n";

	// vertex_collisions: keyed by vertex id (3D)
	std::cout << "-- vertex_collisions --\n";
	{
		std::set<ipc::index_t> keys;
		for (auto &[k, _] : L.collisions.vertex_collisions)
			keys.insert(k);
		for (auto &[k, _] : H.collisions.vertex_collisions)
			keys.insert(k);
		for (auto k : keys)
		{
			auto it_l = L.collisions.vertex_collisions.find(k);
			auto it_h = H.collisions.vertex_collisions.find(k);
			const auto *dl = it_l != L.collisions.vertex_collisions.end() ? it_l->second.get() : nullptr;
			const auto *dh = it_h != H.collisions.vertex_collisions.end() ? it_h->second.get() : nullptr;
			if (!((dl && touches_suspect_dict(*dl)) || (dh && touches_suspect_dict(*dh))))
				continue;
			report_dict_pair("V(" + std::to_string(k) + ")", dl, dh);
		}
	}

	// edge_edge_collisions: keyed by ordered edge pair
	std::cout << "-- edge_edge_collisions --\n";
	{
		std::set<std::pair<ipc::index_t, ipc::index_t>> keys;
		for (auto &[k, _] : L.collisions.edge_edge_collisions)
			keys.insert(k);
		for (auto &[k, _] : H.collisions.edge_edge_collisions)
			keys.insert(k);
		for (auto k : keys)
		{
			auto it_l = L.collisions.edge_edge_collisions.find(k);
			auto it_h = H.collisions.edge_edge_collisions.find(k);
			const auto *dl = it_l != L.collisions.edge_edge_collisions.end() ? it_l->second.get() : nullptr;
			const auto *dh = it_h != H.collisions.edge_edge_collisions.end() ? it_h->second.get() : nullptr;
			if (!((dl && touches_suspect_dict(*dl)) || (dh && touches_suspect_dict(*dh))))
				continue;
			report_dict_pair("EE(" + std::to_string(k.first) + "," + std::to_string(k.second) + ")", dl, dh);
		}
	}

	// face_collisions: keyed by face id, value is vector<unique_ptr<dict>> over qp.
	// For each face dict, compute the virtual vertex (face quadrature point) at
	// lo and hi using barycentric weights, then evaluate per-sub-collision value
	// and gradient. The discontinuity term will show |Δval| or ||Δgrad|| >> 0
	// for an essentially-zero displacement step.
	std::cout << "-- face_collisions (with per-sub val/grad) --\n";
	auto face_q_pos = [&](const Eigen::MatrixXd &X, int fi, int qi,
						  const ipc::ESPParameters &p) {
		Eigen::Matrix<double, 1, 3> q = Eigen::Matrix<double, 1, 3>::Zero();
		const auto &qp = p.face_quad_rule[qi];
		for (int j = 0; j < 3; ++j)
			q += qp.lambda[j] * X.row(F(fi, j));
		return q;
	};
	struct FaceRow
	{
		std::string key;
		std::string name;
		HashT hash;
		double val_lo = 0, val_hi = 0;
		double gnorm_lo = 0, gnorm_hi = 0, dgnorm = 0;
		double dist_lo = 0, dist_hi = 0;
		int dtype_lo = -1, dtype_hi = -1;
		bool touches_suspect = false;
	};
	auto pt_dtype_str = [](ipc::PointTriangleDistanceType t) {
		switch (t)
		{
		case ipc::PointTriangleDistanceType::P_T0:
			return "P_T0";
		case ipc::PointTriangleDistanceType::P_T1:
			return "P_T1";
		case ipc::PointTriangleDistanceType::P_T2:
			return "P_T2";
		case ipc::PointTriangleDistanceType::P_E0:
			return "P_E0";
		case ipc::PointTriangleDistanceType::P_E1:
			return "P_E1";
		case ipc::PointTriangleDistanceType::P_E2:
			return "P_E2";
		case ipc::PointTriangleDistanceType::P_T:
			return "P_T";
		default:
			return "AUTO";
		}
	};
	auto pe_dtype_str = [](ipc::PointEdgeDistanceType t) {
		switch (t)
		{
		case ipc::PointEdgeDistanceType::P_E0:
			return "P_E0";
		case ipc::PointEdgeDistanceType::P_E1:
			return "P_E1";
		case ipc::PointEdgeDistanceType::P_E:
			return "P_E";
		default:
			return "AUTO";
		}
	};
	std::vector<FaceRow> frows;
	{
		std::set<ipc::index_t> keys;
		for (auto &[k, _] : L.collisions.face_collisions)
			keys.insert(k);
		for (auto &[k, _] : H.collisions.face_collisions)
			keys.insert(k);
		for (auto k : keys)
		{
			auto it_l = L.collisions.face_collisions.find(k);
			auto it_h = H.collisions.face_collisions.find(k);
			size_t n_lo_qp = it_l != L.collisions.face_collisions.end() ? it_l->second.size() : 0;
			size_t n_hi_qp = it_h != H.collisions.face_collisions.end() ? it_h->second.size() : 0;
			size_t nq = std::max(n_lo_qp, n_hi_qp);
			for (size_t qi = 0; qi < nq; ++qi)
			{
				const auto *dl = (qi < n_lo_qp) ? it_l->second[qi].get() : nullptr;
				const auto *dh = (qi < n_hi_qp) ? it_h->second[qi].get() : nullptr;
				if (!dl && !dh)
					continue;
				const bool susp =
					(dl && touches_suspect_dict(*dl)) || (dh && touches_suspect_dict(*dh));
				const int s_lo = dl ? dl->size() : 0;
				const int s_hi = dh ? dh->size() : 0;
				const int ns = std::max(s_lo, s_hi);
				for (int i = 0; i < ns; ++i)
				{
					FaceRow r;
					r.key = "F(" + std::to_string(k) + ",qp" + std::to_string(qi) + ")[" + std::to_string(i) + "]";
					r.touches_suspect = susp;
					auto sample = [&](const ipc::ESPCollision &c,
									  const Eigen::MatrixXd &X,
									  const ipc::ESPParameters &p,
									  double &val, double &gnorm, int &dtype, double &d2) {
						auto qpos = face_q_pos(X, k, qi, p);
						ipc::VertexMatrixView<3> view(X, qpos);
						Eigen::VectorXd pos = c.dof(view);
						val = c(pos, p, nullptr);
						Eigen::VectorXd g = c.gradient(pos, p, nullptr);
						gnorm = g.norm();
						const std::string nm = c.name();
						if (nm == "fv_3d" && pos.size() == 12)
						{
							auto t = ipc::point_triangle_distance_type(
								pos.segment<3>(9), pos.segment<3>(0), pos.segment<3>(3), pos.segment<3>(6));
							dtype = (int)t;
							d2 = ipc::point_triangle_distance(
								pos.segment<3>(9), pos.segment<3>(0), pos.segment<3>(3), pos.segment<3>(6), t);
						}
						else if (nm == "ev_3d" && pos.size() == 9)
						{
							auto t = ipc::point_edge_distance_type(
								pos.segment<3>(6), pos.segment<3>(0), pos.segment<3>(3));
							dtype = (int)t;
							d2 = ipc::point_edge_distance(
								pos.segment<3>(6), pos.segment<3>(0), pos.segment<3>(3), t);
						}
						else if (nm == "vv_3d" && pos.size() == 6)
						{
							dtype = -1;
							d2 = (pos.segment<3>(0) - pos.segment<3>(3)).squaredNorm();
						}
						else
						{
							dtype = -1;
							d2 = 0;
						}
					};
					if (i < s_lo)
					{
						const auto &c = (*dl)[i];
						r.name = c.name();
						r.hash = c.get_typed_hash();
						sample(c, lo.displaced, L.params, r.val_lo, r.gnorm_lo, r.dtype_lo, r.dist_lo);
					}
					if (i < s_hi)
					{
						const auto &c = (*dh)[i];
						if (r.name.empty())
						{
							r.name = c.name();
							r.hash = c.get_typed_hash();
						}
						sample(c, hi.displaced, H.params, r.val_hi, r.gnorm_hi, r.dtype_hi, r.dist_hi);
					}
					r.dgnorm = std::abs(r.gnorm_hi - r.gnorm_lo);
					frows.push_back(std::move(r));
				}
			}
		}
	}
	// First: list any sub-collision whose distance type FLIPPED between lo and hi.
	std::cout << "\n=== Sub-collisions with distance-type flip ===\n";
	int n_flipped = 0;
	for (auto &r : frows)
	{
		if (r.dtype_lo != r.dtype_hi && r.dtype_lo >= 0 && r.dtype_hi >= 0)
		{
			n_flipped++;
			const char *slo = (r.name == "fv_3d")   ? pt_dtype_str((ipc::PointTriangleDistanceType)r.dtype_lo)
							  : (r.name == "ev_3d") ? pe_dtype_str((ipc::PointEdgeDistanceType)r.dtype_lo)
													: "?";
			const char *shi = (r.name == "fv_3d")   ? pt_dtype_str((ipc::PointTriangleDistanceType)r.dtype_hi)
							  : (r.name == "ev_3d") ? pe_dtype_str((ipc::PointEdgeDistanceType)r.dtype_hi)
													: "?";
			std::cout << "  " << r.key << " " << r.name
					  << " hash=(" << r.hash[0] << "," << r.hash[1] << "," << r.hash[2] << ")"
					  << "  dtype: " << slo << " → " << shi
					  << "  d²_lo=" << r.dist_lo << " d²_hi=" << r.dist_hi
					  << "  susp=" << (r.touches_suspect ? "yes" : "no") << "\n";
		}
	}
	std::cout << "(" << n_flipped << " sub-collisions flipped distance-type)\n";

	std::sort(frows.begin(), frows.end(), [](const FaceRow &a, const FaceRow &b) {
		const bool a_flip = (a.dtype_lo != a.dtype_hi);
		const bool b_flip = (b.dtype_lo != b.dtype_hi);
		if (a_flip != b_flip)
			return a_flip;
		return std::max(std::abs(a.val_hi - a.val_lo), a.dgnorm) > std::max(std::abs(b.val_hi - b.val_lo), b.dgnorm);
	});
	std::cout << "\nkey                          name      hash                 val_lo         val_hi         |Δval|          ||g_lo||       ||g_hi||       Δ||g||         dtype_lo dtype_hi susp\n";
	const int M = std::min<int>(30, (int)frows.size());
	for (int i = 0; i < M; ++i)
	{
		const FaceRow &r = frows[i];
		std::cout << r.key;
		for (int p = (int)r.key.size(); p < 30; ++p)
			std::cout << ' ';
		std::cout << r.name;
		for (int p = (int)r.name.size(); p < 10; ++p)
			std::cout << ' ';
		std::cout << "(" << r.hash[0] << "," << r.hash[1] << "," << r.hash[2] << ")";
		std::cout << "  " << r.val_lo << "  " << r.val_hi
				  << "  " << std::abs(r.val_hi - r.val_lo)
				  << "  " << r.gnorm_lo << "  " << r.gnorm_hi
				  << "  " << r.dgnorm
				  << "  " << r.dtype_lo << "       " << r.dtype_hi
				  << "       " << (r.touches_suspect ? "yes" : "no") << "\n";
	}

	// Summary: if every per-sub Δ is FP-noise but the global ||Δg|| isn't, the
	// discontinuity is in the dict-level aggregation, not per-sub stencil.
	double max_dval = 0, max_dg = 0;
	for (auto &r : frows)
	{
		max_dval = std::max(max_dval, std::abs(r.val_hi - r.val_lo));
		max_dg = std::max(max_dg, r.dgnorm);
	}
	std::cout << "\n=== Summary ===\n";
	std::cout << "global ||Δgrad||             = " << dg.norm() << "\n";
	std::cout << "max per-sub |Δval|           = " << max_dval << "\n";
	std::cout << "max per-sub |Δ||grad||_local|= " << max_dg << "\n";
	if (n_flipped > 0)
	{
		std::cout << "→ " << n_flipped << " sub-collision(s) flipped distance-type. The\n"
										  "  per-region distance gradient formula switches at region\n"
										  "  boundaries; this is the discontinuity source. Look in\n"
										  "  ESPCollisionTemplate<Face3P1, Vertex3>::gradient (and\n"
										  "  ::gradient_nearfar) at high_order_collision_template.cpp:721\n"
										  "  — point_triangle_distance_type / point_triangle_distance_gradient.\n";
	}
	else if (dg.norm() > 1e-10 && max_dval < 1e-10 && max_dg < 1e-10)
	{
		std::cout << "→ Per-sub stencils are STABLE and no dtype flips; check the\n"
					 "  near/far aggregation (gradient_nearfar) inside\n"
					 "  ESPPotential::gradient.\n";
	}
	else
	{
		std::cout << "→ Look at top rows above with the largest |Δval| / Δ||g||;\n"
					 "  those sub-collisions are the discontinuity source.\n";
	}
	return 0;
}
