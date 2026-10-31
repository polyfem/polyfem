#pragma once

#include <polyfem/Common.hpp>
#include <polyfem/basis/ElementBases.hpp>

#include <Eigen/Core>

#include <functional>
#include <memory>
#include <vector>

namespace polyfem::solver
{
	/// @brief Fast element validity for P1 tetrahedra with a NeoHookean material (performance path of the NonsmoothNewton contact term).
	///
	/// For these elements "elastic energy finite" is equivalent to "every tet keeps a positive determinant". The check is
	/// anchored at the outer iterate x: with D(y) = D(x) + E and ||E||_2 <= 2 sqrt(3) delta_e (delta_e = largest displacement
	/// of the tet's nodes relative to x), sigma_min(D(y)) >= sigma_min(D(x)) - ||E||_2 > 0 whenever delta_e is below
	/// tau_e = 0.9 sigma_min(D(x)) / (2 sqrt(3)). Displacements on a segment are bounded by the larger end point, so a
	/// certified tet stays positively oriented on the whole segment; uncertified tets are checked exactly at the trial point.
	class FastTetValidity
	{
	public:
		/// @brief Build the tet table, or return null when not applicable (not all P1 tets with a linear geometry map, or a
		/// material other than NeoHookean)
		static std::shared_ptr<FastTetValidity> build(const std::vector<basis::ElementBases> &bases,
													  const std::vector<basis::ElementBases> &geom_bases,
													  const int n_bases, const json &materials);

		/// @brief Certificate radii anchored at one outer iterate
		struct Anchor
		{
			Eigen::VectorXd x;    ///< reduced anchor point (empty: no anchor)
			Eigen::VectorXd full; ///< full anchor point
			Eigen::VectorXd tau;  ///< per-tet certificate radius
		};

		/// @brief Anchor at the reduced point x (full point xfull); no-op if already anchored at x
		void anchor(const Eigen::VectorXd &x, const Eigen::VectorXd &xfull, Anchor &a) const;

		/// @brief Largest 2^-i (i < 40) such that the end point a + 2^-i (b - a) has no inverted tet; 0 if none
		/// @param reduced_to_full lift of a reduced point to the full dofs
		double valid_fraction(const Anchor &anc, const Eigen::VectorXd &a, const Eigen::VectorXd &b,
							  const std::function<Eigen::VectorXd(const Eigen::VectorXd &)> &reduced_to_full) const;

		int n_tets() const { return int(tets_.rows()); }

	private:
		double tet_det(int e, const Eigen::VectorXd &uf) const;

		Eigen::MatrixXi tets_;     ///< 4 FE node ids per tet
		Eigen::MatrixXd nodes_;    ///< rest node positions
		Eigen::VectorXd tet_sign_; ///< sign of each tet's rest determinant
	};
} // namespace polyfem::solver
