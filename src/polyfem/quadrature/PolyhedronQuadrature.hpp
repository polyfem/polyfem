#pragma once

#include "Quadrature.hpp"
#include <geogram/mesh/mesh.h>

namespace polyfem
{
	namespace quadrature
	{
		class PolyhedronQuadrature
		{
		public:
			///
			/// @brief      Gets the quadrature points & weights for a polyhedron.
			///
			/// @param[in]  V        #V x 3 input surface vertices (triangulated surface of the polyhedron)
			/// @param[in]  F        #F x 3 input surface faces
			/// @param[in]  kernel   A point in the kernel of the polyhedron
			/// @param[in]  order    Order of the quadrature
			/// @param[out] quadr    Computed quadrature data
			///
			static void get_quadrature(
				const Eigen::MatrixXd &V,
				const Eigen::MatrixXi &F,
				const Eigen::RowVector3d &kernel,
				const int order,
				Quadrature &quadr);
		};
	} // namespace quadrature
} // namespace polyfem
