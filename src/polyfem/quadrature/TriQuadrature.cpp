#include "TriQuadrature.hpp"
#include "LineQuadrature.hpp"

#include <vector>
#include <cassert>
#include <cmath>

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
#include <polyfem/autogen/auto_triangle.ipp>

				default:
					assert(false);
				};
			}
		} // namespace

		TriQuadrature::TriQuadrature()
		{
		}

		void TriQuadrature::get_quadrature(const int order, Quadrature &quad)
		{
			Quadrature tmp;

			get_weight_and_points(order, quad.points, quad.weights);

			assert(fabs(quad.weights.sum() - 1) < 1e-14);
			assert(quad.points.minCoeff() >= 0 && quad.points.maxCoeff() <= 1);

			assert(quad.points.rows() == quad.weights.size());

			quad.weights /= 2;
		}
	} // namespace quadrature
} // namespace polyfem
