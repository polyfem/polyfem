#ifdef POLYFEM_WITH_MISO

////////////////////////////////////////////////////////////////////////////////
#include <polyfem/utils/Jacobian.hpp>
#include <polyfem/utils/par_for.hpp>
#include "EigenAdapters.hpp"
#include <polyfem/autogen/auto_p_bases.hpp>

#include <iomanip>
#include <iostream>
#include <cmath>
#include <limits>
#include <random>
#include <algorithm>

#include <Eigen/Dense>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <polyfem/State.hpp>
#include <polyfem/varforms/VarForm.hpp>

#include "VarFormTestAccess.hpp"
////////////////////////////////////////////////////////////////////////////////

using namespace polyfem;
using namespace polyfem::utils;
using namespace polyfem::autogen;
namespace
{
	std::shared_ptr<State> get_state(int dim, const std::string &material_type = "NeoHookean", const int discr_order = 1)
	{
		const std::string path = POLYFEM_DATA_DIR;

		json material;
		material = R"(
        {
            "type": "NeoHookean",
            "E": 20000,
            "nu": 0.3,
            "rho": 1000,
            "phi": 1,
            "psi": 1
        }
        )"_json;

		json in_args = R"(
		{
			"time": {
				"dt": 0.001,
				"tend": 1.0
			},

			"output": {
				"log": {
					"level": "error"
				}
			}

		})"_json;
		in_args["materials"] = material;
		if (discr_order != 1)
			in_args["space"]["discr_order"] = discr_order;
		if (dim == 2)
		{
			in_args["geometry"] = R"([{
				"enabled": true,
				"surface_selection": 7
			}])"_json;
			in_args["geometry"][0]["mesh"] = path + "/contact/meshes/2D/simple/circle/circle36.obj";
			in_args["boundary_conditions"] = R"({
				"dirichlet_boundary": [{
					"id": "all",
					"value": [0, 0]
				}],
				"rhs": [10, 10]
			})"_json;
		}
		else
		{
			in_args["geometry"] = R"([{
				"transformation": {
					"scale": [0.1, 1, 1]
				},
				"surface_selection": [
					{
						"id": 1,
						"axis": "z",
						"position": 0.8,
						"relative": true
					},
					{
						"id": 2,
						"axis": "-z",
						"position": 0.2,
						"relative": true
					},
					{
						"id": 3,
						"box": [[0, 0, 0.2], [1, 1, 0.8]],
						"relative": true
					}
				],
				"n_refs": 1
			}])"_json;
			in_args["geometry"][0]["mesh"] = path + "/contact/meshes/3D/simple/bar/bar-6.msh";
			in_args["boundary_conditions"] = R"({
				"neumann_boundary": [{
					"id": 1,
					"value": [1000, 1000, 1000]
				}],
				"pressure_boundary": [{
					"id": 1,
					"value": -2000
				},
				{
					"id": 2,
					"value": -2000
				},
				{
					"id": 3,
					"value": -2000
				}],
				"rhs": [0, 0, 0]
			})"_json;
		}

		auto state = std::make_shared<State>();
		state->init(in_args, true);
		state->set_max_threads(1);

		state->load_mesh();

		test::VarFormTestAccess::prepare(*state->variational_formulation);

		return state;
	}

	template <int N>
	constexpr Eigen::Matrix<double, ((N + 1) * (N + 2)) / 2, 3> upsample_triangle()
	{
		constexpr int num = ((N + 1) * (N + 2)) / 2;

		Eigen::Matrix<double, num, 3> out;
		for (int i = 0, k = 0; i <= N; i++)
			for (int j = 0; i + j <= N; j++, k++)
			{
				std::array<int, 3> arr = {{i, j, N - i - j}};
				std::sort(arr.begin(), arr.end());

				out.row(k) << i, j, N - i - j;
			}

		out.template leftCols<3>() /= N;
		return out;
	}

	template <int N>
	constexpr Eigen::Matrix<double, ((N + 1) * (N + 2) * (N + 3)) / 6, 4> upsample_tetrahedron()
	{
		constexpr int num = ((N + 1) * (N + 2) * (N + 3)) / 6;

		Eigen::Matrix<double, num, 4> out;
		for (int i = 0, k = 0; i <= N; i++)
			for (int j = 0; j <= N; j++)
				for (int l = 0; i + j + l <= N; l++, k++)
				{
					std::array<int, 4> arr = {{i, j, l, N - i - j - l}};
					std::sort(arr.begin(), arr.end());

					out.row(k) << i, j, l, N - i - j - l;
				}

		out.template leftCols<4>() /= N;
		return out;
	}
} // namespace

TEST_CASE("jacobian-evaluate", "[jacobian]")
{
	const double tol = 1e-8;
	constexpr int N = 7;
	for (int dim = 2; dim <= 3; dim++)
	{
		for (int order = 1; order < 4; order++)
		{
			Eigen::MatrixXd cp;
			if (dim == 2)
				autogen::p_nodes_2d(order, cp);
			else
				autogen::p_nodes_3d(order, cp);
			cp += Eigen::MatrixXd::Random(cp.rows(), cp.cols()) * 0.2;

			Eigen::MatrixXd uv;
			if (dim == 2)
				uv = upsample_triangle<N>().leftCols<2>();
			else
				uv = upsample_tetrahedron<N>().leftCols<3>();

			Eigen::VectorXd jac1 = robust_evaluate_jacobian(order, cp, uv);

			std::vector<Eigen::MatrixXd> grads(cp.rows(), Eigen::MatrixXd::Zero(uv.rows(), dim));
			for (int bid = 0; bid < cp.rows(); bid++)
				if (dim == 2)
					p_grad_basis_value_2d(false, order, bid, uv, grads[bid]);
				else
					p_grad_basis_value_3d(false, order, bid, uv, grads[bid]);

			Eigen::VectorXd jac2 = jac1;
			for (int k = 0; k < uv.rows(); k++)
			{
				Eigen::MatrixXd jac_mat;
				jac_mat.setZero(dim, dim);
				for (int bid = 0; bid < cp.rows(); bid++)
					jac_mat += cp.row(bid).transpose() * grads[bid].row(k);

				jac2(k) = jac_mat.determinant();
			}

			Eigen::VectorXd denominator = jac1.array().abs().cwiseMax(jac2.array().abs()).cwiseMax(tol);
			REQUIRE(((jac2 - jac1).array() / denominator.array()).abs().maxCoeff() / tol < 1);
		}
	}
}

TEST_CASE("jacobian-tree", "[jacobian]")
{
	// Leaf node
	Tree t;
	REQUIRE(!t.has_children());
	REQUIRE(t.depth() == 0);
	REQUIRE(t.n_leaves() == 1);

	// One level of children
	t.add_children(4);
	REQUIRE(t.has_children());
	REQUIRE(t.n_children() == 4);
	REQUIRE(t.depth() == 1);
	REQUIRE(t.n_leaves() == 4);

	// merge a two-level source: child(0) has 4 grandchildren
	Tree src;
	src.add_children(4);
	src.child(0).add_children(4);

	const bool changed = t.merge(src, 3);
	REQUIRE(changed);
	// child(0) expanded to depth 2; children 1-3 remain leaves
	REQUIRE(t.depth() == 2);
	REQUIRE(t.n_leaves() == 7); // 4 from child(0) + 1 each from 1,2,3

	// Second merge: no new structure → no change
	const bool changed2 = t.merge(src, 3);
	REQUIRE(!changed2);
}

// Helper: invert one element by dragging a single node far in +x.
// Uses local node 1 of element 1; among the 3 nodes, partition-of-unity
// guarantees at least one has a positive n_x, and node 1 does for this mesh.
static Eigen::VectorXd make_x_flip(const test::VarFormDebugData &debug, const int dim)
{
	Eigen::VectorXd u = Eigen::VectorXd::Zero(debug.n_bases * dim);
	const int dof = (*debug.bases)[1].bases[1].global()[0].index;
	u(dof * dim) = 10.0;
	return u;
}

TEST_CASE("jacobian-is-valid", "[jacobian]")
{
	for (const int dim : {2, 3})
	{
		auto state = get_state(dim);
		const auto debug = test::VarFormTestAccess::debug_data(*state->variational_formulation);
		const int ndof = debug.n_bases * dim;

		// Zero displacement → all elements valid
		{
			const Eigen::VectorXd u = Eigen::VectorXd::Zero(ndof);
			auto [valid, inv_id, tree] = is_valid(dim, *debug.bases, *debug.geometry_bases, u);
			REQUIRE(valid);
			REQUIRE(inv_id == -1);
		}

		// x-flip → det F = -1 everywhere → all elements invalid
		{
			const Eigen::VectorXd u_inv = make_x_flip(debug, dim);
			auto [valid, inv_id, tree] = is_valid(dim, *debug.bases, *debug.geometry_bases, u_inv);
			REQUIRE(!valid);
			REQUIRE(inv_id >= 0);
		}
	}
}

TEST_CASE("jacobian-count-invalid", "[jacobian]")
{
	for (const int dim : {2, 3})
	{
		auto state = get_state(dim);
		const auto debug = test::VarFormTestAccess::debug_data(*state->variational_formulation);
		const int ndof = debug.n_bases * dim;

		// Zero displacement → no invalid elements
		const Eigen::VectorXd u_valid = Eigen::VectorXd::Zero(ndof);
		REQUIRE(count_invalid(dim, *debug.bases, *debug.geometry_bases, u_valid).empty());

		// x-flip → all elements invalid
		const Eigen::VectorXd u_inv = make_x_flip(debug, dim);
		REQUIRE(!count_invalid(dim, *debug.bases, *debug.geometry_bases, u_inv).empty());
	}
}

TEST_CASE("jacobian-max-time-step", "[jacobian]")
{
	for (const int dim : {2, 3})
	{
		auto state = get_state(dim);
		const auto debug = test::VarFormTestAccess::debug_data(*state->variational_formulation);
		const int ndof = debug.n_bases * dim;
		const Eigen::VectorXd u_zero = Eigen::VectorXd::Zero(ndof);
		const Eigen::VectorXd u_inv = make_x_flip(debug, dim);

		// Same start and end (valid → valid) → step = 1
		{
			auto [step, id, inv_s, tree] = max_time_step(dim, *debug.bases, *debug.geometry_bases, u_zero, u_zero);
			REQUIRE(step == Catch::Approx(1.0));
		}

		// Valid → inverted → step ∈ (0, 1)
		auto [step, id, inv_s, tree] = max_time_step(dim, *debug.bases, *debug.geometry_bases, u_zero, u_inv);
		REQUIRE(step > 0.0);
		REQUIRE(step < 1.0);

		// Certified endpoint must be valid: u0 + step*(u_inv - u0) = step*u_inv
		const Eigen::VectorXd u_cert = step * u_inv;
		auto [cert_valid, cert_id, cert_tree] = is_valid(dim, *debug.bases, *debug.geometry_bases, u_cert);
		REQUIRE(cert_valid);
	}
}

// Refinement only lands in the right place if miso's child q is the same
// sub-simplex as polyfem's child q -- with the same vertex order, since each
// child's own children are defined relative to it. That is an assumption about
// generated code, so check it: split real miso subdomains along random paths,
// read each child's corners (its coordinate functions are affine, so their
// control points are the vertex positions), and compare them with what
// extract_subelement() produces for the same path.
namespace
{
	struct IntervalGuard
	{
		IntervalGuard() { miso::RealInterval::init(); }
		~IntervalGuard() { miso::RealInterval::deinit(); }
	};

	// Spatial child indices of a path -> the leaf simplex polyfem refines to.
	Eigen::MatrixXd polyfem_leaf(const Eigen::MatrixXd &ref, const std::vector<int> &spatial_path, const int n_children)
	{
		utils::Tree tree;
		utils::Tree *node = &tree;
		for (const int q : spatial_path)
		{
			node->add_children(n_children);
			node = &node->child(q);
		}
		const auto [leaves, levels] = extract_subelement(ref, tree);
		const int dim = ref.cols();
		// The deepest leaves are the last split's children, contiguous and in
		// child order, so the path's leaf is the one at its final index.
		int seen = 0;
		for (int l = 0; l < static_cast<int>(levels.size()); ++l)
			if (levels[l] == static_cast<int>(spatial_path.size()) && seen++ == spatial_path.back())
				return leaves.middleRows(l * (dim + 1), dim + 1);
		return Eigen::MatrixXd();
	}

	// Corners of a miso subdomain, as polyfem-ordered rows. Space-time
	// subdomains have each spatial corner twice (t0, t1): vertex 2k is corner k.
	template <typename Sub>
	Eigen::MatrixXd miso_corners(const Sub &s, const int dim, const int *perm, const bool space_time)
	{
		Eigen::MatrixXd out(dim + 1, dim);
		for (int j = 0; j <= dim; ++j)
		{
			const auto v = s.getVertex(space_time ? 2 * j : j);
			for (int d = 0; d < dim; ++d)
				out(perm[j], d) = lower(v[d]);
			if (space_time)
			{
				// Tensor split: both time copies of a corner share its position.
				const auto w = s.getVertex(2 * j + 1);
				for (int d = 0; d < dim; ++d)
					REQUIRE(lower(w[d]) == lower(v[d]));
			}
		}
		return out;
	}

	template <typename Sub>
	void check_child_numbering(Sub root, const int dim, const int *perm, const bool space_time)
	{
		const int n_spatial = dim == 2 ? 4 : 8;
		const int n_children = space_time ? 2 * n_spatial : n_spatial;
		Eigen::MatrixXd ref(dim + 1, dim);
		if (dim == 2)
			ref << 0, 0, 1, 0, 0, 1;
		else
			ref << 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1;

		std::mt19937 rng(42);
		for (int trial = 0; trial < 20; ++trial)
		{
			Sub cur = root;
			std::vector<int> spatial_path;
			for (int level = 0; level < 5; ++level)
			{
				const unsigned q = rng() % n_children;
				Sub next;
				// Scheme 0 is the spatial (Val) or space-time (CGV) split;
				// the time-only scheme never moves the spatial corners.
				cur.split(0, [&](Sub &&c) {
					if (c.history->index == q)
						next = std::move(c);
				});
				REQUIRE(next.history);
				REQUIRE(next.history->index == q);
				cur = std::move(next);
				// build_tree() maps space-time child i to spatial child i/2.
				spatial_path.push_back(space_time ? q / 2 : q);
			}
			CAPTURE(dim, space_time, trial, spatial_path);
			const Eigen::MatrixXd expected = polyfem_leaf(ref, spatial_path, n_spatial);
			REQUIRE(expected.size() > 0);
			CHECK(miso_corners(cur, dim, perm, space_time) == expected);
		}
	}
} // namespace

TEST_CASE("jacobian-subdivision-matches-miso", "[jacobian]")
{
	const IntervalGuard guard;

	// The coordinate functions do not depend on the element's nodes, so any
	// node matrix of the right size will do.
	Eigen::MatrixXd tri, tet;
	p_nodes_2d(2, tri);
	p_nodes_3d(2, tet);

	SECTION("static triangle") { check_child_numbering(miso::make_p2tri_val(tri, 0), 2, miso::tri1_perm, false); }
	SECTION("static tetrahedron") { check_child_numbering(miso::make_p2tet_val(tet, 0), 3, miso::tet1_perm, false); }
	SECTION("continuous triangle") { check_child_numbering(miso::make_p2tri_cgv(tri, tri, 0), 2, miso::tri1_perm, true); }
	SECTION("continuous tetrahedron") { check_child_numbering(miso::make_p2tet_cgv(tet, tet, 0), 3, miso::tet1_perm, true); }
}

// is_valid() reports *where* inside the element it found the violation, as a
// subdivision path in the same child numbering the quadrature refinement uses.
// Walking that path with the refinement's own geometry (extract_subelement)
// must land on a simplex with a vertex where the Jacobian is not positive;
// otherwise refining towards it would sample the wrong place.
TEST_CASE("jacobian-is-valid-witness-tree", "[jacobian]")
{
	for (const int dim : {2, 3})
	{
		// P3: the triangle's interior node and the tet's face nodes have basis
		// functions ~ l_i l_j l_k, whose gradient vanishes at every vertex.
		// Moving one leaves the Jacobian at the corners untouched, so any
		// inversion it causes is strictly inside the element and can only be
		// found by subdividing -- which is what the tree has to record.
		auto state = get_state(dim, "NeoHookean", /*discr_order=*/3);
		const auto debug = test::VarFormTestAccess::debug_data(*state->variational_formulation);
		const auto &bases = *debug.bases;
		const auto &gbases = *debug.geometry_bases;
		const int order = std::max(bases[0].bases.front().order(), gbases[0].bases.front().order());
		const Eigen::VectorXd zero = Eigen::VectorXd::Zero(debug.n_bases * dim);

		Eigen::MatrixXd ref(dim + 1, dim);
		if (dim == 2)
			ref << 0, 0, 1, 0, 0, 1;
		else
			ref << 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1;

		// A valid configuration has nothing to point at.
		{
			const auto [valid, id, tree] = is_valid(dim, bases, gbases, zero);
			REQUIRE(valid);
			REQUIRE(!tree.has_children());
		}

		// Find such a node on element 0 by the property itself rather than by
		// assuming a local node ordering. Its basis gradient must vanish at the
		// corners, so the corners must not move for a push in *any* direction:
		// checking one direction is not enough, since on this axis-aligned
		// mesh an edge node can leave them unchanged along a single axis.
		const auto corner_jac = [&](const Eigen::VectorXd &u) {
			return robust_evaluate_jacobian(order, extract_nodes(dim, bases[0], gbases[0], u, order), ref);
		};
		const Eigen::VectorXd j_rest = corner_jac(zero);
		const double tol = 1e-12 * j_rest.cwiseAbs().maxCoeff();
		Eigen::VectorXd dir;
		for (const auto &b : bases[0].bases)
		{
			const int node = b.global()[0].index;
			bool corners_fixed = true;
			for (int d = 0; d < dim && corners_fixed; ++d)
			{
				Eigen::VectorXd u = zero;
				u(node * dim + d) = 1;
				corners_fixed = (corner_jac(u) - j_rest).cwiseAbs().maxCoeff() < tol;
			}
			if (corners_fixed)
			{
				dir = zero;
				dir(node * dim) = 1;
				break;
			}
		}
		REQUIRE(dir.size() == zero.size());

		// The Jacobian is affine in the displacement of that node, and its
		// gradient changes sign across the element, so a large enough push
		// inverts the interior. Bracket, then bisect to the onset.
		double hi = 1;
		while (std::get<0>(is_valid(dim, bases, gbases, hi * dir)))
			hi *= 2;
		double lo = 0;
		for (int it = 0; it < 30; ++it)
		{
			const double mid = 0.5 * (lo + hi);
			(std::get<0>(is_valid(dim, bases, gbases, mid * dir)) ? lo : hi) = mid;
		}

		for (const double s : {1.001, 1.1, 2.0})
		{
			const Eigen::VectorXd u = s * hi * dir;
			const auto [valid, id, tree] = is_valid(dim, bases, gbases, u);
			REQUIRE(!valid);
			REQUIRE(id >= 0);

			// Corners are still valid, so the violation cannot have been
			// found at the root.
			REQUIRE(tree.has_children());

			const auto [leaves, levels] = extract_subelement(ref, tree);
			const int depth = *std::max_element(levels.begin(), levels.end());

			// The witness is one of the deepest leaves (its siblings are added
			// alongside it), so the smallest Jacobian over their vertices must
			// be non-positive. The scale is the rest Jacobian, for a relative
			// tolerance that absorbs double rounding of an interval certificate.
			const Eigen::MatrixXd cp = extract_nodes(dim, bases[id], gbases[id], u, order);
			const Eigen::MatrixXd rest_cp = extract_nodes(dim, bases[id], gbases[id], zero, order);
			double min_j = std::numeric_limits<double>::infinity();
			double scale = 0;
			for (int l = 0; l < static_cast<int>(levels.size()); ++l)
			{
				const Eigen::MatrixXd v = leaves.middleRows(l * (dim + 1), dim + 1);
				scale = std::max(scale, robust_evaluate_jacobian(order, rest_cp, v).cwiseAbs().maxCoeff());
				if (levels[l] == depth)
					min_j = std::min(min_j, robust_evaluate_jacobian(order, cp, v).minCoeff());
			}
			CAPTURE(dim, s, id, depth, min_j, scale);
			CHECK(min_j <= 1e-10 * scale);
		}
	}
}

// is_valid()/max_time_step() flip the FPU into round-up mode for interval
// arithmetic and must restore it on every exit path, including the throw for an
// unsupported order. A leak here is invisible locally but silently changes the
// rounding of every later floating-point operation on this thread.
TEST_CASE("jacobian-fpu-rounding-restored", "[jacobian]")
{
	// 0.1 is not representable: under round-to-nearest the sum lands just below
	// 1, under round-up strictly above it. Cheap way to observe the mode.
	const auto sum_tenths = []() {
		volatile double acc = 0.0;
		for (int i = 0; i < 10; ++i)
			acc = acc + 0.1;
		return static_cast<double>(acc);
	};

	const double before = sum_tenths();

	auto state = get_state(3);
	const auto debug = test::VarFormTestAccess::debug_data(*state->variational_formulation);
	const Eigen::VectorXd u = Eigen::VectorXd::Zero(debug.n_bases * 3);

	// Normal (non-throwing) path.
	{
		auto [valid, id, tree] = is_valid(3, *debug.bases, *debug.geometry_bases, u);
		REQUIRE(valid);
		REQUIRE(sum_tenths() == before);
	}

	// Throwing path: an order the generated solvers do not cover must still
	// leave the rounding mode as it found it. Tetrahedra stop at P3, so a P4
	// tet basis reaches the unsupported-order branch inside the guard's scope.
	{
		auto p4_state = get_state(3, "NeoHookean", /*discr_order=*/4);
		const auto p4 = test::VarFormTestAccess::debug_data(*p4_state->variational_formulation);
		const Eigen::VectorXd u4 = Eigen::VectorXd::Zero(p4.n_bases * 3);

		REQUIRE_THROWS(is_valid(3, *p4.bases, *p4.geometry_bases, u4));
		REQUIRE(sum_tenths() == before);
	}
}

// The static check must not depend on how many threads it runs across: the
// rounding mode it relies on is per-thread state, so a parallel implementation
// that established it only on the calling thread would silently lose the
// conservative guarantee on every worker.
TEST_CASE("jacobian-is-valid-thread-independent", "[jacobian]")
{
	const size_t n_threads_before = utils::NThread::get().num_threads();

	for (const int dim : {2, 3})
	{
		auto state = get_state(dim);
		const auto debug = test::VarFormTestAccess::debug_data(*state->variational_formulation);
		const int ndof = debug.n_bases * dim;

		for (const Eigen::VectorXd &u : {Eigen::VectorXd(Eigen::VectorXd::Zero(ndof)),
										 Eigen::VectorXd(make_x_flip(debug, dim))})
		{
			utils::NThread::get().set_num_threads(1);
			const auto [serial_valid, serial_id, serial_tree] =
				is_valid(dim, *debug.bases, *debug.geometry_bases, u);

			utils::NThread::get().set_num_threads(8);
			const auto [par_valid, par_id, par_tree] =
				is_valid(dim, *debug.bases, *debug.geometry_bases, u);

			CHECK(serial_valid == par_valid);
			// Which element is reported must be deterministic too, otherwise
			// logs and quadrature refinement drift run to run.
			CHECK(serial_id == par_id);
		}
	}

	utils::NThread::get().set_num_threads(static_cast<int>(n_threads_before));
}

#endif

#include <polyfem/utils/Jacobian.hpp>
#include <polyfem/autogen/auto_p_bases.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

namespace
{
	polyfem::basis::ElementBases make_linear_simplex_basis(
		const int dim,
		const int global_offset = 0,
		const Eigen::RowVectorXd &translation = Eigen::RowVectorXd())
	{
		Eigen::MatrixXd nodes;
		if (dim == 2)
			polyfem::autogen::p_nodes_2d(1, nodes);
		else
			polyfem::autogen::p_nodes_3d(1, nodes);

		if (translation.size() == dim)
			nodes.rowwise() += translation;

		polyfem::basis::ElementBases element;
		element.bases.resize(nodes.rows());
		for (int i = 0; i < nodes.rows(); ++i)
		{
			polyfem::RowVectorNd node = nodes.row(i);
			element.bases[i].init(1, global_offset + i, i, node);
			element.bases[i].set_basis([dim, i](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) {
				if (dim == 2)
					polyfem::autogen::p_basis_value_2d(false, 1, i, uv, val);
				else
					polyfem::autogen::p_basis_value_3d(false, 1, i, uv, val);
			});
			element.bases[i].set_grad([dim, i](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) {
				if (dim == 2)
					polyfem::autogen::p_grad_basis_value_2d(false, 1, i, uv, val);
				else
					polyfem::autogen::p_grad_basis_value_3d(false, 1, i, uv, val);
			});
		}

		return element;
	}

	Eigen::MatrixXd nodal_displacements_to_matrix(const Eigen::VectorXd &u, const int dim)
	{
		Eigen::MatrixXd U(u.size() / dim, dim);
		for (int i = 0; i < U.rows(); ++i)
			U.row(i) = u.segment(i * dim, dim).transpose();
		return U;
	}
} // namespace

TEST_CASE("jacobian extract nodes on linear simplexes", "[jacobian][utils]")
{
	for (const int dim : {2, 3})
	{
		const auto basis = make_linear_simplex_basis(dim);
		const int n_nodes = dim + 1;

		Eigen::VectorXd u(n_nodes * dim);
		for (int i = 0; i < u.size(); ++i)
			u(i) = 0.05 * (i + 1);

		Eigen::MatrixXd reference_nodes;
		if (dim == 2)
			polyfem::autogen::p_nodes_2d(1, reference_nodes);
		else
			polyfem::autogen::p_nodes_3d(1, reference_nodes);

		const Eigen::MatrixXd cp = polyfem::utils::extract_nodes(dim, basis, basis, u, 1);
		const Eigen::MatrixXd expected = reference_nodes + nodal_displacements_to_matrix(u, dim);

		REQUIRE(cp.rows() == expected.rows());
		REQUIRE(cp.cols() == expected.cols());
		for (int r = 0; r < cp.rows(); ++r)
			for (int c = 0; c < cp.cols(); ++c)
				REQUIRE(cp(r, c) == Catch::Approx(expected(r, c)).margin(1e-12));
	}
}

TEST_CASE("jacobian extract nodes over element ranges", "[jacobian][utils]")
{
	const int dim = 2;
	const int n_nodes_per_element = dim + 1;
	const auto basis0 = make_linear_simplex_basis(dim, 0);
	const auto basis1 = make_linear_simplex_basis(dim, n_nodes_per_element, (Eigen::RowVector2d() << 2.0, -1.0).finished());

	const std::vector<polyfem::basis::ElementBases> bases = {basis0, basis1};
	const std::vector<polyfem::basis::ElementBases> gbases = bases;

	Eigen::VectorXd u(bases.size() * n_nodes_per_element * dim);
	for (int i = 0; i < u.size(); ++i)
		u(i) = 0.01 * (i + 1);

	const Eigen::MatrixXd all_cp = polyfem::utils::extract_nodes(dim, bases, gbases, u, 1);
	const Eigen::MatrixXd first_cp = polyfem::utils::extract_nodes(dim, bases, gbases, u, 1, 1);

	REQUIRE(all_cp.rows() == 2 * n_nodes_per_element);
	REQUIRE(first_cp.rows() == n_nodes_per_element);
	REQUIRE(first_cp.cols() == dim);

	for (int r = 0; r < first_cp.rows(); ++r)
		for (int c = 0; c < first_cp.cols(); ++c)
			REQUIRE(first_cp(r, c) == Catch::Approx(all_cp(r, c)).margin(1e-12));
}
