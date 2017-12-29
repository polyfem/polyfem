#ifndef HARMONIC_HPP
#define HARMONIC_HPP

#include <Eigen/Dense>

namespace poly_fem
{
	class Harmonic
	{
	public:

		///
		/// @brief      { Initialize a harmonic function over an element. }
		///
		/// @param[in]  centers  { #C x dim matrix containing the positions of
		///                      the centers of the RBFs defining the harmonic
		///                      functions over the element. The centers are
		///                      placed at a small offset distance from the
		///                      boundary of the element, due to the singularity
		///                      at the centers }
		/// @param[in]  samples  { #S x dim matrix of evaluation points, used to
		///                      approximate the harmonic functions over the
		///                      boundary of the element }
		/// @param[in]  rhs      { #S x #dofs of boundary conditions. Each
		///                      column define how the i-th harmonic function of
		///                      the element should evaluate over the evaluation
		///                      samples }
		///
		Harmonic(const Eigen::MatrixXd &centers, const Eigen::MatrixXd &samples, const Eigen::MatrixXd &rhs);

		///
		/// @brief      { Evaluates one harmonic function over a list of
		///             coordinates }
		///
		/// @param[in]  local_index  { i-th harmonic function to evaluate }
		/// @param[in]  uv           { #uv x dim matrix of coordinates to
		///                          evaluate (in object domain) }
		/// @param[out] val          { #uv x 1 matrix of computed values }
		///
		void basis(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) const;

		///
		/// @brief      { Evaluates the gradient of one harmonic function over a
		///             list of coordinates }
		///
		/// @param[in]  local_index  { i-th harmonic function to evaluate }
		/// @param[in]  uv           { #uv x dim matrix of coordinates to
		///                          evaluate (in object domain) }
		/// @param[out] val          { #uv x dim matrix of computed gradients }
		///
		void grad(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) const;

	private:
		// Compute the weights
		void compute(const Eigen::MatrixXd &samples, const Eigen::MatrixXd &rhs);

		// #C x dim matrix of centers positions
		Eigen::MatrixXd centers_;

		// (#C + dim + 1) x #dofs matrix of weights defining #dofs different
		// harmonic functions over the element
		Eigen::MatrixXd weights_;

	};
}

#endif
