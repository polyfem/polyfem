#include "PrismQuadrature.hpp"
#include "TriQuadrature.hpp"
#include "LineQuadrature.hpp"

#include <vector>
#include <cassert>
#include <cmath>
#include <iostream>

namespace polyfem
{
	namespace quadrature
	{
		PrismQuadrature::PrismQuadrature()
		{
		}

		void PrismQuadrature::get_quadrature(const int order, const int order_h, Quadrature &quad)
		{
			Quadrature tmpt, tmpl;

			TriQuadrature tc;
			tc.get_quadrature(order, tmpt);

			LineQuadrature lc;
			lc.get_quadrature(order, tmpl);

			quad.points.resize(tmpt.weights.size() * tmpl.weights.size(), 3);
			quad.weights.resize(tmpt.weights.size() * tmpl.weights.size());

			int index = 0;

			for (int i = 0; i < tmpt.weights.size(); ++i)
			{
				for (int j = 0; j < tmpl.weights.size(); ++j)
				{
					quad.points.row(index) << tmpt.points(i, 0), tmpt.points(i, 1), tmpl.points(j, 0);
					quad.weights(index) = tmpt.weights(i) * tmpl.weights(j);
					++index;
				}
			}

			assert(fabs(quad.weights.sum() - 0.5) < 1e-12);
			assert(quad.points.minCoeff() >= 0 && quad.points.maxCoeff() <= 1);

			assert(quad.points.rows() == quad.weights.size());
			assert(index == quad.weights.size());
		}
	} // namespace quadrature
} // namespace polyfem
