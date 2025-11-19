#pragma once

#include "Quadrature.hpp"

namespace polyfem
{

	///
	/// @brief      Tetrahedralize a star-shaped mesh, with a given point in its kernel
	///
	/// @param[in]  V        #V x 3 input mesh vertices
	/// @param[in]  F        #F x 3 input mesh triangles
	/// @param[in]  kernel   A point in the kernel
	/// @param[out] OV       #OV x 3 output mesh vertices
	/// @param[out] OF       #OF x 3 output mesh surface triangles
	/// @param[out] OT       #OT x 4 output mesh tetrahedra
	///
	void tertrahedralize_star_shaped_surface(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F,
											 const Eigen::RowVector3d &kernel, Eigen::MatrixXd &OV, Eigen::MatrixXi &OF, Eigen::MatrixXi &OT);

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
