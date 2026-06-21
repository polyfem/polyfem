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
		void build_tree(Tree &tree, const std::vector<unsigned> &path, unsigned n_children)
		{
			Tree *dst = &tree;
			for (const auto idx : path)
			{
				dst->add_children(n_children);
				dst = &(dst->child(idx));
			}
		}
	} // anonymous namespace
#endif // POLYFEM_WITH_MISO

	std::vector<int> count_invalid(
		const int dim,
		const std::vector<basis::ElementBases> &bases,
		const std::vector<basis::ElementBases> &gbases,
		const Eigen::VectorXd &u)
	{
		std::vector<int> invalidList;
#ifdef POLYFEM_WITH_MISO
		const int order = std::max(bases[0].bases.front().order(), gbases[0].bases.front().order());
		const int n_per = std::max(bases[0].bases.size(), gbases[0].bases.size());
		const Eigen::MatrixXd cp = extract_nodes(dim, bases, gbases, u, order);
		const int n_elem = static_cast<int>(bases.size());

		for (int e = 0; e < n_elem; ++e)
		{
			const int o = e * n_per;
			bool invalid = false;
			if (dim == 2)
			{
				switch (order)
				{
				case 1: invalid = JacEval_P1Tri(rv<3>(cp,o,0), rv<3>(cp,o,1)) <= 0; break;
				case 2: invalid = !solve(make_p2tri_val(cp, o), {0.0}, 0, true).empty(); break;
				case 3: invalid = !solve(make_p3tri_val(cp, o), {0.0}, 0, true).empty(); break;
				case 4: invalid = !solve(make_p4tri_val(cp, o), {0.0}, 0, true).empty(); break;
				default: throw std::invalid_argument("Order not supported");
				}
			}
			else
			{
				switch (order)
				{
				case 1: invalid = JacEval_P1Tet(rv<4>(cp,o,0), rv<4>(cp,o,1), rv<4>(cp,o,2)) <= 0; break;
				case 2: invalid = !solve(make_p2tet_val(cp, o), {0.0}, 0, true).empty(); break;
				case 3: invalid = !solve(make_p3tet_val(cp, o), {0.0}, 0, true).empty(); break;
				default: throw std::invalid_argument("Order not supported");
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
		const double threshold)
	{
#ifdef POLYFEM_WITH_MISO
		const int order = std::max(bases[0].bases.front().order(), gbases[0].bases.front().order());
		const int n_per = std::max(bases[0].bases.size(), gbases[0].bases.size());
		const Eigen::MatrixXd cp = extract_nodes(dim, bases, gbases, u, order);
		const int n_elem = static_cast<int>(bases.size());

		// Tree reconstruction from static validators is best-effort:
		// not all generated Val classes carry subdivision history.
		for (int e = 0; e < n_elem; ++e)
		{
			const int o = e * n_per;
			bool invalid = false;
			if (dim == 2)
			{
				switch (order)
				{
				case 1: invalid = JacEval_P1Tri(rv<3>(cp,o,0), rv<3>(cp,o,1)) <= 0; break;
				case 2: invalid = !solve(make_p2tri_val(cp, o), {0.0}, 0, true).empty(); break;
				case 3: invalid = !solve(make_p3tri_val(cp, o), {0.0}, 0, true).empty(); break;
				case 4: invalid = !solve(make_p4tri_val(cp, o), {0.0}, 0, true).empty(); break;
				default: throw std::invalid_argument("Order not supported");
				}
			}
			else
			{
				switch (order)
				{
				case 1: invalid = JacEval_P1Tet(rv<4>(cp,o,0), rv<4>(cp,o,1), rv<4>(cp,o,2)) <= 0; break;
				case 2: invalid = !solve(make_p2tet_val(cp, o), {0.0}, 0, true).empty(); break;
				case 3: invalid = !solve(make_p3tet_val(cp, o), {0.0}, 0, true).empty(); break;
				default: throw std::invalid_argument("Order not supported");
				}
			}
			if (invalid)
				return {false, e, Tree{}};
		}
		return {true, -1, Tree{}};
#else
		log_and_throw_error("Enable Bezier or Miso library to allow robust Jacobian check!");
		return {false, -1, Tree{}};
#endif
	}

	bool is_valid(
		const int dim,
		const std::vector<basis::ElementBases> &bases,
		const std::vector<basis::ElementBases> &gbases,
		const Eigen::VectorXd &u1,
		const Eigen::VectorXd &u2,
		const double threshold)
	{
#ifdef POLYFEM_WITH_MISO
		const int order = std::max(bases[0].bases.front().order(), gbases[0].bases.front().order());
		const int n_per = std::max(bases[0].bases.size(), gbases[0].bases.size());
		const Eigen::MatrixXd cp1 = extract_nodes(dim, bases, gbases, u1, order);
		const Eigen::MatrixXd cp2 = extract_nodes(dim, bases, gbases, u2, order);
		const int n_elem = static_cast<int>(bases.size());

		for (int e = 0; e < n_elem; ++e)
		{
			const int o = e * n_per;
			bool valid;
			if (dim == 2)
			{
				switch (order)
				{
				case 1: valid = lower(minimize(make_p1tri_cgv(cp1, cp2, o), 1e-6, {0.0})) >= 1.0; break;
				case 2: valid = lower(minimize(make_p2tri_cgv(cp1, cp2, o), 1e-6, {0.0})) >= 1.0; break;
				case 3: valid = lower(minimize(make_p3tri_cgv(cp1, cp2, o), 1e-6, {0.0})) >= 1.0; break;
				case 4: valid = lower(minimize(make_p4tri_cgv(cp1, cp2, o), 1e-6, {0.0})) >= 1.0; break;
				default: throw std::invalid_argument("Order not supported");
				}
			}
			else
			{
				switch (order)
				{
				case 1: valid = lower(minimize(make_p1tet_cgv(cp1, cp2, o), 1e-6, {0.0})) >= 1.0; break;
				case 2: valid = lower(minimize(make_p2tet_cgv(cp1, cp2, o), 1e-6, {0.0})) >= 1.0; break;
				case 3: valid = lower(minimize(make_p3tet_cgv(cp1, cp2, o), 1e-6, {0.0})) >= 1.0; break;
				default: throw std::invalid_argument("Order not supported");
				}
			}
			if (!valid)
				return false;
		}
		return true;
#else
		log_and_throw_error("Enable Bezier or Miso library to allow robust Jacobian check!");
		return false;
#endif
	}

	std::tuple<double, int, double, Tree> max_time_step(
		const int dim,
		const std::vector<basis::ElementBases> &bases,
		const std::vector<basis::ElementBases> &gbases,
		const Eigen::VectorXd &u1,
		const Eigen::VectorXd &u2,
		double precision)
	{
#ifdef POLYFEM_WITH_MISO
		const int order = std::max(bases[0].bases.front().order(), gbases[0].bases.front().order());
		const int n_per = std::max(bases[0].bases.size(), gbases[0].bases.size());
		const Eigen::MatrixXd cp1 = extract_nodes(dim, bases, gbases, u1, order);
		const Eigen::MatrixXd cp2 = extract_nodes(dim, bases, gbases, u2, order);
		const int n_elem = static_cast<int>(bases.size());

		double step = 1.0;
		int invalid_id = -1;
		double invalid_step = 1.0;
		Tree tree;

		// CGV scheme[0] subdivides all variables: 2^(dim+1) children
		const unsigned n_children = (dim == 2) ? 8u : 16u;

		auto run = [&](auto problem, int e) {
			Info info;
			auto result = minimize(std::move(problem), precision, {0.0}, 0, -infinity, 0., &info);
			const double t_lo = lower(result);
			if (t_lo < step)
			{
				step = t_lo;
				invalid_id = e;
				invalid_step = upper(result);
				tree = Tree{};
				build_tree(tree, info.pathToFeasible, n_children);
			}
		};

		for (int e = 0; e < n_elem; ++e)
		{
			const int o = e * n_per;
			if (dim == 2)
			{
				switch (order)
				{
				case 1: run(make_p1tri_cgv(cp1, cp2, o), e); break;
				case 2: run(make_p2tri_cgv(cp1, cp2, o), e); break;
				case 3: run(make_p3tri_cgv(cp1, cp2, o), e); break;
				case 4: run(make_p4tri_cgv(cp1, cp2, o), e); break;
				default: throw std::invalid_argument("Order not supported");
				}
			}
			else
			{
				switch (order)
				{
				case 1: run(make_p1tet_cgv(cp1, cp2, o), e); break;
				case 2: run(make_p2tet_cgv(cp1, cp2, o), e); break;
				case 3: run(make_p3tet_cgv(cp1, cp2, o), e); break;
				default: throw std::invalid_argument("Order not supported");
				}
			}
		}

		return {step, invalid_id, invalid_step, tree};
#else
		log_and_throw_error("Enable Bezier or Miso library to allow robust Jacobian check!");
		return {1.0, -1, 1.0, Tree{}};
#endif
	}
} // namespace polyfem::utils
