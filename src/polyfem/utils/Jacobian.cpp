#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/par_for.hpp>
#include "Jacobian.hpp"
#include <polyfem/autogen/auto_p_bases.hpp>
#include <polyfem/io/Evaluator.hpp>

#ifdef POLYFEM_WITH_MISO
#include "EigenAdapters.hpp"
#include <algorithms/solve.hpp>
#include <algorithms/minimize.hpp>
#include <algorithms/batch-minimize.hpp>
using namespace miso;
#endif

using namespace polyfem::assembler;

namespace polyfem::utils
{
	Eigen::MatrixXd extract_nodes(const int dim, const std::vector<basis::ElementBases> &bases, const std::vector<basis::ElementBases> &gbases, const Eigen::VectorXd &u, int order, int n_elem)
	{
		if (n_elem < 0)
			n_elem = bases.size();
		Eigen::MatrixXd local_pts;
		if (dim == 3)
			autogen::p_nodes_3d(order, local_pts);
		else
			autogen::p_nodes_2d(order, local_pts);
		const int n_basis_per_cell = local_pts.rows();
		Eigen::MatrixXd cp = Eigen::MatrixXd::Zero(n_elem * n_basis_per_cell, dim);
		for (int e = 0; e < n_elem; ++e)
		{
			ElementAssemblyValues vals;
			vals.compute(e, dim == 3, local_pts, bases[e], gbases[e]);

			for (std::size_t j = 0; j < vals.basis_values.size(); ++j)
				for (const auto &g : vals.basis_values[j].global)
					cp.middleRows(e * n_basis_per_cell, n_basis_per_cell) += g.val * vals.basis_values[j].val * u.segment(g.index * dim, dim).transpose();

			Eigen::MatrixXd mapped;
			gbases[e].eval_geom_mapping(local_pts, mapped);
			cp.middleRows(e * n_basis_per_cell, n_basis_per_cell) += mapped;
		}
		return cp;
	}

	namespace
	{
		int control_point_order(const std::vector<basis::ElementBases> &bases, const std::vector<basis::ElementBases> &gbases)
		{
			return std::max(bases[0].bases.front().order(), gbases[0].bases.front().order());
		}

		Eigen::MatrixXd control_point_nodes(const int dim, const int order)
		{
			Eigen::MatrixXd local_pts;
			if (dim == 3)
				autogen::p_nodes_3d(order, local_pts);
			else
				autogen::p_nodes_2d(order, local_pts);
			return local_pts;
		}
	} // namespace

	Eigen::MatrixXd rest_nodes(const int dim, const std::vector<basis::ElementBases> &bases, const std::vector<basis::ElementBases> &gbases)
	{
		const Eigen::MatrixXd local_pts = control_point_nodes(dim, control_point_order(bases, gbases));
		const int n_per = local_pts.rows();
		Eigen::MatrixXd rest(bases.size() * n_per, dim);
		for (int e = 0; e < (int)bases.size(); ++e)
		{
			Eigen::MatrixXd mapped;
			gbases[e].eval_geom_mapping(local_pts, mapped);
			rest.middleRows(e * n_per, n_per) = mapped;
		}
		return rest;
	}

	Eigen::MatrixXd displacement_nodes(const int dim, const std::vector<basis::ElementBases> &bases, const std::vector<basis::ElementBases> &gbases, const Eigen::VectorXd &u, bool *exact)
	{
		const Eigen::MatrixXd local_pts = control_point_nodes(dim, control_point_order(bases, gbases));
		const int n_per = local_pts.rows();
		constexpr double tol = 1e-10;

		Eigen::MatrixXd disp = Eigen::MatrixXd::Zero(bases.size() * n_per, dim);
		bool all_exact = true;
		for (int e = 0; e < (int)bases.size(); ++e)
		{
			ElementAssemblyValues vals;
			vals.compute(e, dim == 3, local_pts, bases[e], gbases[e]);

			// A conforming Lagrange basis of the control-point order is 1 at
			// its own node and 0 at the others (up to the rounding of nodes
			// like 1/3), with a single global of weight 1: each control point
			// is then one entry of u, copied rather than computed.
			std::vector<int> owner(n_per, -1);
			bool nodal = true;
			for (int j = 0; j < (int)vals.basis_values.size(); ++j)
			{
				const auto &global = vals.basis_values[j].global;
				if (global.size() != 1 || global.front().val != 1)
					nodal = false;
				const Eigen::VectorXd &val = vals.basis_values[j].val;
				for (int k = 0; k < n_per; ++k)
				{
					if (std::abs(val(k) - 1) <= tol)
					{
						nodal = nodal && owner[k] < 0;
						owner[k] = j;
					}
					else if (std::abs(val(k)) > tol)
						nodal = false;
				}
			}
			nodal = nodal && std::find(owner.begin(), owner.end(), -1) == owner.end();

			if (nodal)
			{
				for (int k = 0; k < n_per; ++k)
				{
					const int g = vals.basis_values[owner[k]].global.front().index;
					disp.row(e * n_per + k) = u.segment(g * dim, dim).transpose();
				}
				continue;
			}

			// Geometry of higher order than the displacement, or a
			// non-conforming basis: evaluate, which rounds.
			all_exact = false;
			for (std::size_t j = 0; j < vals.basis_values.size(); ++j)
				for (const auto &g : vals.basis_values[j].global)
					disp.middleRows(e * n_per, n_per) += g.val * vals.basis_values[j].val * u.segment(g.index * dim, dim).transpose();
		}
		if (exact)
			*exact = all_exact;
		return disp;
	}

	Eigen::MatrixXd extract_nodes(const int dim, const basis::ElementBases &basis, const basis::ElementBases &gbasis, const Eigen::VectorXd &u, int order)
	{
		Eigen::MatrixXd local_pts;
		if (dim == 3)
			autogen::p_nodes_3d(order, local_pts);
		else
			autogen::p_nodes_2d(order, local_pts);

		Eigen::MatrixXd cp;
		gbasis.eval_geom_mapping(local_pts, cp);

		ElementAssemblyValues vals;
		vals.compute(0, dim == 3, local_pts, basis, gbasis);
		for (std::size_t j = 0; j < vals.basis_values.size(); ++j)
			for (const auto &g : vals.basis_values[j].global)
				cp += g.val * vals.basis_values[j].val * u.segment(g.index * dim, dim).transpose();

		return cp;
	}

	Eigen::MatrixXd refined_nodes(const int dim, const int i)
	{
		Eigen::MatrixXd A(dim + 1, dim);
		if (dim == 2)
		{
			A << 0., 0.,
				1., 0.,
				0., 1.;
			switch (i)
			{
			case 0:
				break;
			case 1:
				A.col(0).array() += 1;
				break;
			case 2:
				A.col(1).array() += 1;
				break;
			case 3:
				A.array() -= 1;
				A *= -1;
				break;
			default:
				throw std::runtime_error("Invalid node index");
			}
		}
		else
		{
			A << 0, 0, 0,
				1, 0, 0,
				0, 1, 0,
				0, 0, 1;
			switch (i)
			{
			case 0:
				break;
			case 1:
				A.col(0).array() += 1;
				break;
			case 2:
				A.col(1).array() += 1;
				break;
			case 3:
				A.col(2).array() += 1;
				break;
			case 4:
			{
				Eigen::VectorXd tmp = 1. - A.col(0).array() - A.col(1).array();
				A.col(2) += A.col(1);
				A.col(1) = A.col(0);
				A.col(0) = tmp;
				break;
			}
			case 5:
			{
				Eigen::VectorXd tmp = 1. - A.col(0).array();
				A.col(1) += A.col(0);
				A.col(0) = tmp;
				break;
			}
			case 6:
			{
				Eigen::VectorXd tmp0 = A.col(0);
				Eigen::VectorXd tmp1 = A.col(1);
				A.col(0) = tmp1;
				A.col(1) = 1. - tmp0.array() - tmp1.array();
				A.col(2) += tmp0 + tmp1;
				break;
			}
			case 7:
			{
				Eigen::VectorXd tmp = A.col(1);
				A.col(0) += tmp;
				A.col(1) = 1. - tmp.array();
				A.col(2) += tmp;
				break;
			}
			default:
				throw std::runtime_error("Invalid node index");
			}
		}
		return A / 2;
	}

	/// @brief given the position of the vertices of a triangle, extract the subtriangle vertices based on the tree
	/// @param pts vertices of the triangle
	/// @param tree refinement hiararchy
	/// @return vertices of the subtriangles, and the refinement level of each subtriangle
	std::tuple<Eigen::MatrixXd, std::vector<int>> extract_subelement(const Eigen::MatrixXd &pts, const Tree &tree)
	{
		if (!tree.has_children())
			return {pts, std::vector<int>{0}};

		const int dim = pts.cols();
		Eigen::MatrixXd out;
		std::vector<int> levels;
		for (int i = 0; i < tree.n_children(); i++)
		{
			Eigen::MatrixXd uv;
			uv.setZero(dim + 1, dim + 1);
			uv.rightCols(dim) = refined_nodes(dim, i);
			if (dim == 2)
				uv.col(0) = 1. - uv.col(2).array() - uv.col(1).array();
			else
				uv.col(0) = 1. - uv.col(3).array() - uv.col(1).array() - uv.col(2).array();

			Eigen::MatrixXd pts_ = uv * pts;

			auto [tmp, L] = extract_subelement(pts_, tree.child(i));
			if (out.size() == 0)
				out = tmp;
			else
			{
				out.conservativeResize(out.rows() + tmp.rows(), Eigen::NoChange);
				out.bottomRows(tmp.rows()) = tmp;
			}
			for (int &i : L)
				++i;
			levels.insert(levels.end(), L.begin(), L.end());
		}
		return {out, levels};
	}

	Eigen::VectorXd robust_evaluate_jacobian(
		const int order,
		const Eigen::MatrixXd &cp,
		const Eigen::MatrixXd &uv)
	{
#ifdef POLYFEM_WITH_MISO
		const int dim = cp.cols();
		Eigen::VectorXd result(uv.rows());

		// TODO: replace with constant function evaluator (same as used for P=1 static validity)
		std::vector<Eigen::MatrixXd> grads(cp.rows(), Eigen::MatrixXd::Zero(uv.rows(), dim));
		for (int bid = 0; bid < cp.rows(); bid++)
		{
			if (dim == 2)
				autogen::p_grad_basis_value_2d(false, order, bid, uv, grads[bid]);
			else
				autogen::p_grad_basis_value_3d(false, order, bid, uv, grads[bid]);
		}
		for (int k = 0; k < uv.rows(); k++)
		{
			Eigen::MatrixXd jac_mat = Eigen::MatrixXd::Zero(dim, dim);
			for (int bid = 0; bid < cp.rows(); bid++)
				jac_mat += cp.row(bid).transpose() * grads[bid].row(k);
			result(k) = jac_mat.determinant();
		}
		return result;
#else
		log_and_throw_error("Enable Bezier or Miso library to allow robust Jacobian evaluation!");
		return Eigen::VectorXd::Zero(uv.rows());
#endif
	}

#ifdef POLYFEM_WITH_MISO
	namespace
	{
		void build_tree(Tree &tree, const std::vector<unsigned> &path, unsigned n_spatial)
		{
			// Space-time child i maps to spatial child i/2 (even=lower t, odd=upper t).
			// Children beyond 2*n_spatial are time-only subdivisions and are skipped.
			Tree *dst = &tree;
			for (const auto idx : path)
			{
				if (idx < 2 * n_spatial)
				{
					dst->add_children(n_spatial);
					dst = &(dst->child(idx / 2));
				}
				// else: time-only subdivision, skip
			}
		}

		/// @brief Build a refinement tree from a *spatial* subdivision path.
		///
		/// The static (*Val) problems split space only, with a single scheme,
		/// so each path entry is already the spatial child index -- unlike the
		/// space-time path handled by build_tree(). The child numbering matches
		/// the one refine_quadrature() uses in ElasticForm (checked for both
		/// triangles and tetrahedra, including vertex order, so it holds at
		/// every depth).
		void build_spatial_tree(Tree &tree, const std::vector<unsigned> &path, const unsigned n_children)
		{
			Tree *dst = &tree;
			for (const auto idx : path)
			{
				assert(idx < n_children);
				dst->add_children(n_children);
				dst = &(dst->child(idx));
			}
		}
		/// @brief RAII guard for miso's interval arithmetic FPU state.
		///
		/// RealInterval::init()/deinit() switch the FPU rounding mode to
		/// round-up and back to round-to-nearest. That is *per-thread* CPU
		/// state, so it must be paired on every exit path -- including the
		/// throws below, which previously left the whole thread stuck in
		/// round-up mode -- and re-established inside each worker when these
		/// routines are parallelized.
		struct FPUGuard
		{
			FPUGuard() { RealInterval::init(); }
			~FPUGuard() { RealInterval::deinit(); }
			FPUGuard(const FPUGuard &) = delete;
			FPUGuard &operator=(const FPUGuard &) = delete;
		};

		/// @brief Report an element type/order the conservative check cannot handle.
		///
		/// The miso-generated solvers only cover simplices: triangles up to
		/// P4 and tetrahedra up to P3. Anything else (quads, hexahedra,
		/// prisms, pyramids, or a higher order) lands here.
		[[noreturn]] void throw_unsupported(const int dim, const int order)
		{
			log_and_throw_error(
				"Conservative Jacobian check does not support {} of order {}; "
				"supported: triangles (P1-P4) and tetrahedra (P1-P3). Use "
				"\"check_inversion\": \"Discrete\" for this mesh.",
				dim == 2 ? "2D elements" : "3D elements", order);
		}
	} // anonymous namespace
#endif // POLYFEM_WITH_MISO

	// Debug utility: counts elements with invalid Jacobian in a given configuration.
	// Not used in the solve loop; called for diagnostics and visualization.
	std::vector<int> count_invalid(
		const int dim,
		const std::vector<basis::ElementBases> &bases,
		const std::vector<basis::ElementBases> &gbases,
		const Eigen::VectorXd &u,
		const unsigned max_iter,
		const Eigen::MatrixXd *rest)
	{
		std::vector<int> invalidList;
#ifdef POLYFEM_WITH_MISO
		const int order = control_point_order(bases, gbases);
		const int n_per = std::max(bases[0].bases.size(), gbases[0].bases.size());
		const int n_elem = static_cast<int>(bases.size());
		// Rest and displacement go to miso separately, and are read before
		// the interval guard switches the FPU to round-up, so a rest computed
		// here matches the cached one.
		const Eigen::MatrixXd rest_here = rest ? Eigen::MatrixXd() : rest_nodes(dim, bases, gbases);
		const Eigen::MatrixXd &r = rest ? *rest : rest_here;
		const Eigen::MatrixXd d = displacement_nodes(dim, bases, gbases, u);
		const FPUGuard fpu_guard;

		auto run_solve = [&](auto problem, int e) {
			Parameters params;
			params.constraintEpsilon.assign(1, 0.0);
			params.maxIter = max_iter;
			params.findOne = true;
			Info info;
			auto sols = solve(std::move(problem), params, &info);
			if (!info.success())
				logger().warn("Jacobian solve gave up at element {} after {} iterations", e, info.numIterations);
			return !sols.empty();
		};

		for (int e = 0; e < n_elem; ++e)
		{
			const int o = e * n_per;
			bool invalid = false;
			if (dim == 2)
			{
				switch (order)
				{
				case 1:
					invalid = eval_p1tri_jac(r, d, o) <= 0;
					break;
				case 2:
					invalid = run_solve(make_p2tri_val(r, d, o), e);
					break;
				case 3:
					invalid = run_solve(make_p3tri_val(r, d, o), e);
					break;
				case 4:
					invalid = run_solve(make_p4tri_val(r, d, o), e);
					break;
				default:
					throw_unsupported(dim, order);
				}
			}
			else
			{
				switch (order)
				{
				case 1:
					invalid = eval_p1tet_jac(r, d, o) <= 0;
					break;
				case 2:
					invalid = run_solve(make_p2tet_val(r, d, o), e);
					break;
				case 3:
					invalid = run_solve(make_p3tet_val(r, d, o), e);
					break;
				default:
					throw_unsupported(dim, order);
				}
			}
			if (invalid)
				invalidList.push_back(e);
		}
#else
		log_and_throw_error("Enable Bezier or Miso library to allow robust Jacobian check!");
#endif
		return invalidList;
	}

	std::tuple<bool, int, Tree>
	is_valid(
		const int dim,
		const std::vector<basis::ElementBases> &bases,
		const std::vector<basis::ElementBases> &gbases,
		const Eigen::VectorXd &u,
		const double threshold,
		const unsigned max_iter,
		const Eigen::MatrixXd *rest)
	{
#ifdef POLYFEM_WITH_MISO
		const int order = control_point_order(bases, gbases);
		const int n_per = std::max(bases[0].bases.size(), gbases[0].bases.size());
		const int n_elem = static_cast<int>(bases.size());
		// Rest and displacement go to miso separately, and are read before
		// the interval guard switches the FPU to round-up, so a rest computed
		// here matches the cached one.
		const Eigen::MatrixXd rest_here = rest ? Eigen::MatrixXd() : rest_nodes(dim, bases, gbases);
		const Eigen::MatrixXd &r = rest ? *rest : rest_here;
		const Eigen::MatrixXd d = displacement_nodes(dim, bases, gbases, u);
		const FPUGuard fpu_guard;

		// Spatial split: 2^dim children per level.
		const unsigned n_children = (dim == 2) ? 4u : 8u;
		Tree witness;

		// The witness subdomain solve() returns carries its own subdivision
		// history, so the location of the violation comes for free; it is
		// only a matter of not discarding it.
		auto run_solve = [&](auto problem, int e) {
			Parameters params;
			params.constraintEpsilon.assign(1, threshold);
			params.maxIter = max_iter;
			params.findOne = true;
			Info info;
			auto sols = solve(std::move(problem), params, &info);
			if (!info.success())
			{
				// Ambiguous counts as invalid, but there is no certified
				// location to refine towards, so leave the tree empty.
				logger().warn("Jacobian solve gave up at element {} after {} iterations", e, info.numIterations);
				return true;
			}
			if (sols.empty())
				return false;
			if (sols.front().history)
				build_spatial_tree(witness, sols.front().history->path(), n_children);
			return true;
		};

		for (int e = 0; e < n_elem; ++e)
		{
			const int o = e * n_per;
			bool invalid = false;
			if (dim == 2)
			{
				switch (order)
				{
				case 1:
					invalid = eval_p1tri_jac(r, d, o) <= 0;
					break;
				case 2:
					invalid = run_solve(make_p2tri_val(r, d, o), e);
					break;
				case 3:
					invalid = run_solve(make_p3tri_val(r, d, o), e);
					break;
				case 4:
					invalid = run_solve(make_p4tri_val(r, d, o), e);
					break;
				default:
					throw_unsupported(dim, order);
				}
			}
			else
			{
				switch (order)
				{
				case 1:
					invalid = eval_p1tet_jac(r, d, o) <= 0;
					break;
				case 2:
					invalid = run_solve(make_p2tet_val(r, d, o), e);
					break;
				case 3:
					invalid = run_solve(make_p3tet_val(r, d, o), e);
					break;
				default:
					throw_unsupported(dim, order);
				}
			}
			// P1 has a constant Jacobian per element, so there is nothing to
			// localize and the tree stays empty, as for the continuous check.
			if (invalid)
				return {false, e, std::move(witness)};
		}
		return {true, -1, Tree{}};
#else
		log_and_throw_error("Enable Bezier or Miso library to allow robust Jacobian check!");
		return {false, -1, Tree{}};
#endif
	}

	std::tuple<double, int, double, Tree> max_time_step(
		const int dim,
		const std::vector<basis::ElementBases> &bases,
		const std::vector<basis::ElementBases> &gbases,
		const Eigen::VectorXd &u1,
		const Eigen::VectorXd &u2,
		double precision,
		double threshold,
		const unsigned max_iter,
		const Eigen::MatrixXd *rest)
	{
#ifdef POLYFEM_WITH_MISO
		const int order = control_point_order(bases, gbases);
		const int n_per = std::max(bases[0].bases.size(), gbases[0].bases.size());
		const int n_elem = static_cast<int>(bases.size());
		// As in is_valid(): rest and displacement separately, before the guard.
		const Eigen::MatrixXd rest_here = rest ? Eigen::MatrixXd() : rest_nodes(dim, bases, gbases);
		const Eigen::MatrixXd &r = rest ? *rest : rest_here;
		const Eigen::MatrixXd d1 = displacement_nodes(dim, bases, gbases, u1);
		const Eigen::MatrixXd d2 = displacement_nodes(dim, bases, gbases, u2);
		const FPUGuard fpu_guard;

		double step = 1.0;
		int invalid_id = -1;
		double invalid_step = 1.0;
		Tree tree;

		// Spatial split: 2^dim children (ignore time subdivisions)
		const unsigned n_children = (dim == 2) ? 4u : 8u;

		// Build a vector of all per-element problems (same type for all elements) and
		// run a single batch_minimize over the entire mesh.
		auto run_batch = [&](auto factory) {
			using ProblemType = decltype(factory(0));
			std::vector<ProblemType> problems;
			problems.reserve(n_elem);
			for (int e = 0; e < n_elem; ++e)
				problems.push_back(factory(e * n_per));

			Parameters params;
			params.targetPrecision = precision;
			params.constraintEpsilon.assign(1, threshold);
			params.maxIter = max_iter;
			params.requiredMinimum = 0.;
			params.minBoxWidth = 0.;
			Info info;
			auto result = batch_minimize(std::move(problems), params, &info);
			const double t_lo = lower(result);
			if (!info.success())
			{
				if (t_lo <= 0)
					logger().error("Jacobian batch_minimize gave up with step size 0 after {} iterations", info.numIterations);
				else
					logger().warn("Jacobian batch_minimize gave up after {} iterations (step={})", info.numIterations, t_lo);
			}
			if (t_lo < step)
			{
				step = t_lo;
				invalid_id = info.pathToFeasibleId ? static_cast<int>(*info.pathToFeasibleId) : -1;
				invalid_step = upper(result);
				tree = Tree{};
				// P1 is a problem in t only: its subdivisions carry no
				// spatial location to refine towards.
				if (order > 1)
					build_tree(tree, info.pathToFeasible, n_children);
			}
		};

		if (dim == 2)
		{
			switch (order)
			{
			case 1:
				run_batch([&](int o) { return make_p1tri_cgv(r, d1, d2, o); });
				break;
			case 2:
				run_batch([&](int o) { return make_p2tri_cgv(r, d1, d2, o); });
				break;
			case 3:
				run_batch([&](int o) { return make_p3tri_cgv(r, d1, d2, o); });
				break;
			case 4:
				run_batch([&](int o) { return make_p4tri_cgv(r, d1, d2, o); });
				break;
			default:
				throw_unsupported(dim, order);
			}
		}
		else
		{
			switch (order)
			{
			case 1:
				run_batch([&](int o) { return make_p1tet_cgv(r, d1, d2, o); });
				break;
			case 2:
				run_batch([&](int o) { return make_p2tet_cgv(r, d1, d2, o); });
				break;
			case 3:
				run_batch([&](int o) { return make_p3tet_cgv(r, d1, d2, o); });
				break;
			default:
				throw_unsupported(dim, order);
			}
		}

		return {step, invalid_id, invalid_step, tree};
#else
		log_and_throw_error("Enable Bezier or Miso library to allow robust Jacobian check!");
		return {1.0, -1, 1.0, Tree{}};
#endif
	}
} // namespace polyfem::utils
