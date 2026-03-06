#include "PyramidQuadrature.hpp"

#include <vector>
#include <cassert>
#include <cmath>
#include <iostream>

namespace polyfem
{
	namespace quadrature
	{
		namespace
		{
			void get_weight_and_points(const int order, Eigen::MatrixXd &points, Eigen::VectorXd &weights)
			{
				switch (order)
				{
#include <polyfem/autogen/auto_pyramid.ipp>

				default:
					assert(false);
				};
			}
		} // namespace

		PyramidQuadrature::PyramidQuadrature()
		{
		}

		void PyramidQuadrature::get_quadrature(const int order, Quadrature &quad)
		{
			if (order <= 5)
			{
				// Tabulated Felippa rules — exact for pyramid rational function space,
				// available for orders 1–5 (up to 27 quadrature points).
				get_weight_and_points(order, quad.points, quad.weights);
				assert(quad.weights.minCoeff() > 0 && "Felippa quadrature weight non-positive");
				assert(fabs(quad.weights.sum() - 1.0 / 3.0) < 1e-12);
				assert(quad.points.minCoeff() >= 0 && quad.points.maxCoeff() <= 1);
				assert(quad.points.rows() == quad.weights.size());
				return;
			}

			// For order > 5: Duffy transform + tensor-product Gauss-Legendre on [0,1]^3.
			//
			// Map:  x = xi*(1-zeta),  y = eta*(1-zeta),  z = zeta,  J = (1-zeta)^2
			//
			// Under this map, (x/(1-z))^a (y/(1-z))^b (1-z)^c becomes
			// xi^a eta^b (1-zeta)^(c+2) — a pure polynomial.  A monomial x^a y^b z^c
			// (degree d = a+b+c) becomes xi^a eta^b zeta^c (1-zeta)^(a+b+2), which has
			// zeta-degree d+2.  An n-pt GL rule is exact for degree 2n-1, so we need
			//   n >= ceil((d+3)/2)  =>  n = (order + 4) / 2  (integer division).

			const int n = (order + 4) / 2; // 1D Gauss-Legendre points per direction

			// 1D Gauss-Legendre nodes & weights on [-1,1], then mapped to [0,1].
			// Use Eigen's built-in: nodes t in [-1,1], weights w, map t_01=(t+1)/2, w_01=w/2.
			Eigen::VectorXd t(n), w(n);
			// Compute via eigenvalue method (Golub-Welsch).
			{
				// Tridiagonal symmetric Jacobi matrix for Gauss-Legendre.
				Eigen::MatrixXd J = Eigen::MatrixXd::Zero(n, n);
				for (int i = 1; i < n; ++i)
				{
					const double b = i / std::sqrt(4.0 * i * i - 1.0);
					J(i - 1, i) = b;
					J(i, i - 1) = b;
				}
				Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(J);
				t = es.eigenvalues();          // nodes on [-1,1]
				w = 2.0 * es.eigenvectors().row(0).cwiseAbs2().transpose(); // weights
			}
			// Map to [0,1]
			const Eigen::VectorXd xi1d = (t.array() + 1.0) * 0.5;
			const Eigen::VectorXd w1d  = w * 0.5;

			const int nq = n * n * n;
			quad.points.resize(nq, 3);
			quad.weights.resize(nq);

			int idx = 0;
			for (int i = 0; i < n; ++i)   // xi direction
			for (int j = 0; j < n; ++j)   // eta direction
			for (int k = 0; k < n; ++k)   // zeta direction
			{
				const double xi   = xi1d(i);
				const double eta  = xi1d(j);
				const double zeta = xi1d(k);
				const double one_minus_zeta = 1.0 - zeta;
				// Duffy Jacobian = (1-zeta)^2 must be strictly positive;
				// GL nodes never reach the endpoint zeta=1 (the apex singularity).
				assert(one_minus_zeta > 0 && "Duffy Jacobian degenerate: quadrature point at pyramid apex");
				quad.points(idx, 0) = xi  * one_minus_zeta;  // x
				quad.points(idx, 1) = eta * one_minus_zeta;  // y
				quad.points(idx, 2) = zeta;                   // z
				quad.weights(idx)   = w1d(i) * w1d(j) * w1d(k) * one_minus_zeta * one_minus_zeta;
				++idx;
			}

			assert(quad.weights.minCoeff() > 0 && "Duffy quadrature weight non-positive");
			assert(fabs(quad.weights.sum() - 1.0 / 3.0) < 1e-10);
			assert(quad.points.minCoeff() >= -1e-14);
			assert(quad.points.rows() == quad.weights.size());
		}
	} // namespace quadrature
} // namespace polyfem
