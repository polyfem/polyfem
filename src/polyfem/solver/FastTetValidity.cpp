#include "FastTetValidity.hpp"

#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/MaybeParallelFor.hpp>

#include <Eigen/Dense>
#include <Eigen/SVD>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <limits>

namespace polyfem::solver
{
	namespace
	{
		constexpr double INF = std::numeric_limits<double>::infinity();

		/// @brief For each of the K entries of f(i), the minimum over i in [0, n) as the serial fold m = std::min(m, f(i))
		/// from +inf gives it: a NaN term is skipped, and of two equal terms the earlier one is kept. Blocks of 1024
		/// indices are folded in parallel and their minima are folded in block order, so the result does not depend on
		/// the number of threads.
		template <int K, typename F>
		std::array<double, K> par_min(const int n, const F &f)
		{
			constexpr int block = 1024;
			const int n_blocks = (n + block - 1) / block;
			std::array<double, K> init;
			init.fill(INF);
			std::vector<std::array<double, K>> block_min(n_blocks, init);
			utils::maybe_parallel_for(n_blocks, [&](int start, int end, int /*thread_id*/) {
				for (int b = start; b < end; ++b)
					for (int i = b * block; i < std::min(n, (b + 1) * block); ++i)
					{
						const std::array<double, K> v = f(i);
						for (int k = 0; k < K; ++k)
							block_min[b][k] = std::min(block_min[b][k], v[k]);
					}
			});
			std::array<double, K> m = init;
			for (const auto &bm : block_min)
				for (int k = 0; k < K; ++k)
					m[k] = std::min(m[k], bm[k]);
			return m;
		}

		/// @brief the i in [0, n) with keep(i), in increasing order (the tests in parallel, then a serial pass)
		template <typename F>
		std::vector<int> par_select(const int n, const F &keep)
		{
			std::vector<char> flag(n);
			utils::maybe_parallel_for(n, [&](int start, int end, int /*thread_id*/) {
				for (int i = start; i < end; ++i)
					flag[i] = keep(i) ? 1 : 0;
			});
			std::vector<int> out;
			for (int i = 0; i < n; ++i)
				if (flag[i])
					out.push_back(i);
			return out;
		}
	} // namespace

	std::shared_ptr<FastTetValidity> FastTetValidity::build(const std::vector<basis::ElementBases> &bases,
															const std::vector<basis::ElementBases> &geom_bases,
															const int n_bases, const json &materials)
	{
		bool neo = true;
		const json mats = materials.is_array() ? materials : json::array({materials});
		for (const auto &m : mats)
			neo = neo && m.contains("type") && m["type"] == "NeoHookean";
		bool p1 = n_bases > 0;
		for (const auto &bs : bases)
			p1 = p1 && bs.bases.size() == 4;
		// the corner tets are the elements only for a linear geometry map (no curved / higher-order geometry)
		p1 = p1 && geom_bases.size() == bases.size();
		for (const auto &gs : geom_bases)
			p1 = p1 && gs.bases.size() == 4;
		if (!(neo && p1))
		{
			logger().info("[NonsmoothNewton] fast validity off (native check used): NeoHookean {}, P1 tets {}", neo, p1);
			return nullptr;
		}

		auto v = std::make_shared<FastTetValidity>();
		v->tets_.resize(bases.size(), 4);
		v->nodes_.setZero(n_bases, 3);
		for (size_t e = 0; e < bases.size(); ++e)
			for (int c = 0; c < 4; ++c)
			{
				const auto &g = bases[e].bases[c].global();
				p1 = p1 && g.size() == 1;
				v->tets_(e, c) = g[0].index;
				v->nodes_.row(g[0].index) = g[0].node;
			}
		v->tet_sign_.resize(v->tets_.rows());
		for (int e = 0; e < v->tets_.rows(); ++e)
		{
			Eigen::Matrix3d D;
			for (int c = 1; c < 4; ++c)
				D.col(c - 1) = (v->nodes_.row(v->tets_(e, c)) - v->nodes_.row(v->tets_(e, 0))).transpose();
			v->tet_sign_[e] = D.determinant() > 0 ? 1.0 : -1.0;
		}
		logger().info("[NonsmoothNewton] fast validity {}: NeoHookean {}, P1 tets {} ({} tets)", p1 ? "on" : "off (native check used)", neo, p1, v->tets_.rows());
		if (!p1)
			return nullptr;
		// rest determinants as tet_det evaluates them at the zero displacement
		const Eigen::VectorXd zero = Eigen::VectorXd::Zero(3 * n_bases);
		v->rest_det_.resize(v->tets_.rows());
		for (int e = 0; e < v->tets_.rows(); ++e)
			v->rest_det_[e] = v->tet_det(e, zero);
		return v;
	}

	double FastTetValidity::min_jacobian(const Eigen::VectorXd &uf) const
	{
		return par_min<1>(n_tets(), [&](const int e) { return std::array<double, 1>{{tet_det(e, uf) / rest_det_[e]}}; })[0];
	}

	Eigen::VectorXd FastTetValidity::dets(const Eigen::VectorXd &uf) const
	{
		Eigen::VectorXd d(n_tets());
		utils::maybe_parallel_for(n_tets(), [&](int start, int end, int /*thread_id*/) {
			for (int e = start; e < end; ++e)
				d[e] = tet_det(e, uf);
		});
		return d;
	}

	void FastTetValidity::min_jacobians(const Eigen::VectorXd &det_x, const Eigen::VectorXd &yf, const bool want_min, const bool want_ratio,
										double &min_j, double &min_ratio) const
	{
		// both minima in one sweep (an entry not wanted stays +inf)
		const std::array<double, 2> m = par_min<2>(n_tets(), [&](const int e) {
			const double dy = tet_det(e, yf);
			return std::array<double, 2>{{want_min ? dy / rest_det_[e] : INF, want_ratio ? dy / det_x[e] : INF}};
		});
		min_j = want_min ? m[0] : std::numeric_limits<double>::quiet_NaN();
		min_ratio = want_ratio ? m[1] : std::numeric_limits<double>::quiet_NaN();
	}

	double FastTetValidity::tet_det(int e, const Eigen::VectorXd &uf) const
	{
		const auto &Tt = tets_;
		const auto &X = nodes_;
		const Eigen::Vector3d p0 = X.row(Tt(e, 0)).transpose() + uf.segment<3>(3 * Tt(e, 0));
		Eigen::Matrix3d D;
		for (int c = 1; c < 4; ++c)
			D.col(c - 1) = X.row(Tt(e, c)).transpose() + uf.segment<3>(3 * Tt(e, c)) - p0;
		return D.determinant() * tet_sign_[e];
	}

	void FastTetValidity::anchor(const Eigen::VectorXd &x, const Eigen::VectorXd &xfull, Anchor &a) const
	{
		if (a.x.size() == x.size() && a.x == x)
			return;
		a.x = x;
		a.full = xfull;
		const auto &Tt = tets_;
		const auto &X = nodes_;
		a.tau.resize(Tt.rows());
		// one 3x3 SVD per tet, in parallel
		utils::maybe_parallel_for(int(Tt.rows()), [&](int start, int end, int /*thread_id*/) {
			for (int e = start; e < end; ++e)
			{
				const Eigen::Vector3d p0 = X.row(Tt(e, 0)).transpose() + a.full.segment<3>(3 * Tt(e, 0));
				Eigen::Matrix3d D;
				for (int c = 1; c < 4; ++c)
					D.col(c - 1) = X.row(Tt(e, c)).transpose() + a.full.segment<3>(3 * Tt(e, c)) - p0;
				const double smin = Eigen::JacobiSVD<Eigen::Matrix3d>(D).singularValues()(2);
				a.tau[e] = 0.9 * smin / (2 * std::sqrt(3.0));
			}
		});
	}

	double FastTetValidity::valid_fraction(const Anchor &anc, const Eigen::VectorXd &a, const Eigen::VectorXd &b,
										   const std::function<Eigen::VectorXd(const Eigen::VectorXd &)> &reduced_to_full) const
	{
		const Eigen::VectorXd ua = reduced_to_full(a) - anc.full, ub = reduced_to_full(b) - anc.full;
		const int nb = int(nodes_.rows());
		Eigen::VectorXd dv(nb);
		utils::maybe_parallel_for(nb, [&](int start, int end, int /*thread_id*/) {
			for (int v = start; v < end; ++v)
				dv[v] = std::max(ua.segment<3>(3 * v).norm(), ub.segment<3>(3 * v).norm());
		});
		const auto &Tt = tets_;
		// the tets not certified by the anchor, in order
		const std::vector<int> U = par_select(int(Tt.rows()), [&](const int e) {
			return std::max(std::max(dv[Tt(e, 0)], dv[Tt(e, 1)]), std::max(dv[Tt(e, 2)], dv[Tt(e, 3)])) >= anc.tau[e];
		});
		double alpha = 1;
		for (int i = 0; i < 40; ++i, alpha *= 0.5)
		{
			const Eigen::VectorXd yf = reduced_to_full(a + alpha * (b - a));
			std::atomic<bool> inverted{false};
			utils::maybe_parallel_for(int(U.size()), [&](int start, int end, int /*thread_id*/) {
				for (int j = start; j < end && !inverted.load(std::memory_order_relaxed); ++j)
					if (!(tet_det(U[j], yf) > 0))
						inverted.store(true, std::memory_order_relaxed);
			});
			if (!inverted.load())
				return alpha;
		}
		return 0;
	}
} // namespace polyfem::solver
