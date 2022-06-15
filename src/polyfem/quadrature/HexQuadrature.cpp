#include "HexQuadrature.hpp"
#include "LineQuadrature.hpp"

#include <vector>
#include <cassert>
#include <cmath>
#include <iostream>

namespace polyfem
{
	namespace quadrature
	{
		HexQuadrature::HexQuadrature()
		{
		}

		void HexQuadrature::get_quadrature(const int order, Quadrature &quad)
		{
			Quadrature tmp;
			LineQuadrature one_d_quad;
			one_d_quad.get_quadrature(order, tmp);

			const long n_quad_pts = tmp.weights.size();

			quad.points = Eigen::MatrixXd(n_quad_pts * n_quad_pts * n_quad_pts, 3);
			quad.weights = Eigen::MatrixXd(n_quad_pts * n_quad_pts * n_quad_pts, 1);

			for (long i = 0; i < n_quad_pts; ++i)
			{
				for (long j = 0; j < n_quad_pts; ++j)
				{
					for (long k = 0; k < n_quad_pts; ++k)
					{
						const long index = (i * n_quad_pts + j) * n_quad_pts + k;
						quad.points.row(index) = Eigen::Vector3d(tmp.points(k), tmp.points(j), tmp.points(i));
						quad.weights(index) = tmp.weights(i) * tmp.weights(j) * tmp.weights(k);
					}
				}
			}

			assert(fabs(quad.weights.sum() - 1) < 1e-14);
			assert(quad.points.minCoeff() >= 0 && quad.points.maxCoeff() <= 1);

			assert((quad.points.rows() == quad.weights.size()));
		}
	} // namespace quadrature
} // namespace polyfem