#include "TetQuadrature.hpp"
#include "LineQuadrature.hpp"

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
			void get_weight_and_points(const int order, const bool use_corner_quadrature, Eigen::MatrixXd &points, Eigen::VectorXd &weights)
			{
				if (use_corner_quadrature)
				{
					switch (order)
					{
#include <polyfem/autogen/auto_tetrahedron_corner.ipp>

					default:
						assert(false);
					};
				}
				else
				{
					switch (order)
					{
#include <polyfem/autogen/auto_tetrahedron.ipp>

					default:
						assert(false);
					};
				}
			}
		} // namespace

		TetQuadrature::TetQuadrature(bool use_corner_quadrature) : use_corner_quadrature_(use_corner_quadrature)
		{
		}

		void TetQuadrature::get_quadrature(const int order, Quadrature &quad)
		{
			Quadrature tmp;

			get_weight_and_points(order, use_corner_quadrature_, quad.points, quad.weights);

			assert(fabs(quad.weights.sum() - 1) < 1e-12);
			assert(quad.points.minCoeff() >= 0 && quad.points.maxCoeff() <= 1);

			assert(quad.points.rows() == quad.weights.size());

			quad.weights /= 6;
		}
	} // namespace quadrature
} // namespace polyfem
