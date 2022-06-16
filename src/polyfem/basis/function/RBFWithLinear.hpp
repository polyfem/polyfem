#pragma once

#include <polyfem/quadrature/Quadrature.hpp>
#include <Eigen/Dense>

namespace polyfem
{
	namespace basis
	{
		class RBFWithLinear
		{
		public:
			///
			/// @brief Initialize RBF functions over a polytope element.
			///
			/// @param[in]  centers                #C x dim positions of the kernels used to define functions over the polytope. The centers are placed at a small offset distance from the boundary of the element, due to the singularity at the centers
			/// @param[in]  collocation_points     #S x dim positions of the collocation points, used to approximate the RBF functions over the boundary of the element
			/// @param[in]  local_basis_integral   #B x dim containing the expected value of the integral of each basis over the polytope
			/// @param[in]  quadr                  Quadrature points and weights inside the polytope
			/// @param[in]  rhs                    #S x #B of boundary conditions. Each column defines how the i-th basis of the mesh should evaluate on the collocation points sampled on the boundary of the polytope
			/// @param[in]  with_constraints       Impose integral constraints to guarantee linear reproduction for the Poisson equation
			///
			RBFWithLinear(const Eigen::MatrixXd &centers, const Eigen::MatrixXd &collocation_points,
						  const Eigen::MatrixXd &local_basis_integral, const quadrature::Quadrature &quadr,
						  Eigen::MatrixXd &rhs, bool with_constraints = true);

			///
			/// @brief      Evaluates one RBF function over a list of coordinates
			///
			/// @param[in]  local_index   i-th RBF function to evaluate
			/// @param[in]  uv            #uv x dim matrix of coordinates to evaluate (in object domain)
			/// @param[out] val           #uv x 1 matrix of computed values
			///
			void basis(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) const;

			///
			/// @brief      Evaluates the gradient of one RBF function over a list of coordinates
			///
			/// @param[in]  local_index   i-th RBF function to evaluate
			/// @param[in]  uv            #uv x dim matrix of coordinates to evaluate (in object domain)
			/// @param[out] val           #uv x dim matrix of computed gradients
			///
			void grad(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) const;

			///
			/// @brief      Batch evaluates the RBF + polynomials on a set of sample points
			///
			/// @param[in]  uv     #uv x dim matrix of points to evaluate
			/// @param[out] val    #uv x n_loc_bases of bases values over the sample points
			///
			void bases_values(const Eigen::MatrixXd &samples, Eigen::MatrixXd &val) const;

			///
			/// @brief      Batch evaluates the gradient of the RBF + polynomials on a set of sample points
			///
			/// @param[in]  axis   The axis (0, 1, 2) with respect to which to compute the gradient
			/// @param[in]  uv     #uv x dim matrix of points to evaluate
			/// @param[out] val    #uv x n_loc_bases of bases gradient wrt axis over the sample points
			///
			void bases_grads(const int axis, const Eigen::MatrixXd &samples, Eigen::MatrixXd &val) const;

		private:
			bool is_volume() const { return centers_.cols() == 3; }

			// Computes the matrix that evaluates the kernels + polynomial terms on the given sample points
			void compute_kernels_matrix(const Eigen::MatrixXd &samples, Eigen::MatrixXd &A) const;

			// Computes the relationship w = L v + t between the unknowns (v) and the weights w
			void compute_constraints_matrix(const int num_bases, const quadrature::Quadrature &quadr,
											const Eigen::MatrixXd &local_basis_integral, Eigen::MatrixXd &L, Eigen::MatrixXd &t) const;

			// Computes the weights by solving a (possibly constrained) linear least square
			void compute_weights(const Eigen::MatrixXd &collocation_points,
								 const Eigen::MatrixXd &local_basis_integral, const quadrature::Quadrature &quadr,
								 Eigen::MatrixXd &rhs, bool with_constraints);

		private:
			// #C x dim matrix of kernel center positions
			Eigen::MatrixXd centers_;

			// (#C + dim + 1) x #B matrix of weights extending the #B bases that are non-vanishing on the polytope
			Eigen::MatrixXd weights_;
		};
	} // namespace basis
} // namespace polyfem
