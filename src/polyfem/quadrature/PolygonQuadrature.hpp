#pragma once

#include "Quadrature.hpp"

namespace polyfem
{
	namespace quadrature
	{
		class PolygonQuadrature
		{
		public:
			PolygonQuadrature();

			///
			/// @brief      Gets the quadrature points & weights for a polygon.
			///
			/// @param[in]  poly   n x 2 matrix, coordinates of the polyline defining the boundary of the polygon
			/// @param[in]  order  order of the quadrature (ignored for now)
			/// @param[out] quadr  computed quadrature data
			///
			void get_quadrature(const Eigen::MatrixXd &poly, const int order, Quadrature &quadr);
		};
	} // namespace quadrature
} // namespace polyfem
