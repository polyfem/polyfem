#include "FastTetValidity.hpp"

#include <polyfem/utils/Logger.hpp>

#include <Eigen/Dense>
#include <Eigen/SVD>

#include <algorithm>
#include <cmath>

namespace polyfem::solver
{
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
		return v;
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
		for (int e = 0; e < Tt.rows(); ++e)
		{
			const Eigen::Vector3d p0 = X.row(Tt(e, 0)).transpose() + a.full.segment<3>(3 * Tt(e, 0));
			Eigen::Matrix3d D;
			for (int c = 1; c < 4; ++c)
				D.col(c - 1) = X.row(Tt(e, c)).transpose() + a.full.segment<3>(3 * Tt(e, c)) - p0;
			const double smin = Eigen::JacobiSVD<Eigen::Matrix3d>(D).singularValues()(2);
			a.tau[e] = 0.9 * smin / (2 * std::sqrt(3.0));
		}
	}

	double FastTetValidity::valid_fraction(const Anchor &anc, const Eigen::VectorXd &a, const Eigen::VectorXd &b,
										   const std::function<Eigen::VectorXd(const Eigen::VectorXd &)> &reduced_to_full) const
	{
		const Eigen::VectorXd ua = reduced_to_full(a) - anc.full, ub = reduced_to_full(b) - anc.full;
		const int nb = int(nodes_.rows());
		Eigen::VectorXd dv(nb);
		for (int v = 0; v < nb; ++v)
			dv[v] = std::max(ua.segment<3>(3 * v).norm(), ub.segment<3>(3 * v).norm());
		const auto &Tt = tets_;
		std::vector<int> U;
		for (int e = 0; e < Tt.rows(); ++e)
			if (std::max(std::max(dv[Tt(e, 0)], dv[Tt(e, 1)]), std::max(dv[Tt(e, 2)], dv[Tt(e, 3)])) >= anc.tau[e])
				U.push_back(e);
		double alpha = 1;
		for (int i = 0; i < 40; ++i, alpha *= 0.5)
		{
			const Eigen::VectorXd yf = reduced_to_full(a + alpha * (b - a));
			long ninv = 0;
			for (const int e : U)
				if (!(tet_det(e, yf) > 0))
					++ninv;
			if (ninv == 0)
				return alpha;
		}
		return 0;
	}
} // namespace polyfem::solver
