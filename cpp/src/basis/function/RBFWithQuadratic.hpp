#pragma once

#include "Quadrature.hpp"
#include <Eigen/Dense>

namespace poly_fem
{
	class RBFWithQuadratic
	{
	public:

		///
		/// @brief      { Initialize RBF functions over a polytope element. }
		///
		/// @param[in]  centers               { #C x dim positions of the kernels used to define
		///                                   functions over the polytope. The centers are placed at
		///                                   a small offset distance from the boundary of the
		///                                   element, due to the singularity at the centers }
		/// @param[in]  samples               { #S x dim positions of the collocation points, used
		///                                   to approximate the RBF functions over the boundary of
		///                                   the element }
		/// @param[in]  local_basis_integral  { #B x dim containing the expected value of the
		///                                   integral of each basis over the polytope }
		/// @param[in]  quadr                 { Quadrature points and weights inside the polytope }
		/// @param[in]  rhs                   { #S x #B of boundary conditions. Each column defines
		///                                   how the i-th basis of the mesh should evaluate on the
		///                                   collocation points sampled on the boundary of the
		///                                   polytope }
		///
		RBFWithQuadratic(const Eigen::MatrixXd &centers, const Eigen::MatrixXd &samples,
			const Eigen::MatrixXd &local_basis_integral, const Quadrature &quadr,
			Eigen::MatrixXd &rhs);

		///
		/// @brief      { Evaluates one RBF function over a list of coordinates }
		///
		/// @param[in]  local_index  { i-th RBF function to evaluate }
		/// @param[in]  uv           { #uv x dim matrix of coordinates to evaluate (in object
		///                          domain) }
		/// @param[out] val          { #uv x 1 matrix of computed values }
		///
		void basis(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) const;

		///
		/// @brief      { Evaluates the gradient of one RBF function over a list of coordinates }
		///
		/// @param[in]  local_index  { i-th RBF function to evaluate }
		/// @param[in]  uv           { #uv x dim matrix of coordinates to evaluate (in object
		///                          domain) }
		/// @param[out] val          { #uv x dim matrix of computed gradients }
		///
		void grad(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) const;

	private:
		// Compute the weights
		void compute(const Eigen::MatrixXd &samples, const Eigen::MatrixXd &local_basis_integral,
			const Quadrature &quadr, Eigen::MatrixXd &rhs);

		void create_matrix(const Eigen::MatrixXd &samples, Eigen::MatrixXd &A) const;

		// #C x dim matrix of kernel center positions
		Eigen::MatrixXd centers_;

		// (#C + dim + 1) x #B matrix of weights extending the #B bases that are non-vanishing on the polytope
		Eigen::MatrixXd weights_;

		bool is_volume_;

	};
}
