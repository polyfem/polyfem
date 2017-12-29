#ifndef FE_BASIS_3D_HPP
#define FE_BASIS_3D_HPP

#include "Mesh3D.hpp"

#include "ElementBases.hpp"
#include "LocalBoundary.hpp"

#include <Eigen/Dense>
#include <vector>

namespace poly_fem
{
	class FEBasis3d
	{
	public:

		///
		/// @brief      Builds FE basis functions over the entire mesh (H1, H2).
		///
		/// @param[in]  mesh              The input volume mesh
		/// @param[in]  quadrature_order  The quadrature order
		/// @param[out] bases             List of basis functions per element
		/// @param[out] local_boundary    List of descriptor per element,
		///                               indicating which facet of the
		///                               canonical hex lie on the boundary of
		///                               the mesh
		/// @param[out] boundary_nodes    List of dofs which are on the boundary
		///
		/// @return     The bases.
		///
		static int build_bases(
			const Mesh3D &mesh,
			const int quadrature_order,
			std::vector< ElementBases > &bases,
			std::vector< LocalBoundary > &local_boundary,
			std::vector< int > &boundary_nodes);
	};
}

#endif //FE_BASIS_3D_HPP
