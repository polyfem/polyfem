#ifndef FE_BASIS_3D_HPP
#define FE_BASIS_3D_HPP

#include "ElementBases.hpp"
#include "Mesh3D.hpp"
#include "LocalBoundary.hpp"

#include <Eigen/Dense>
#include <vector>

namespace poly_fem
{
	class FEBasis3d
	{
	public:

		///
		/// @brief      Builds FE basis functions over the entire mesh (Q1, Q2).
		///
		/// @param[in]  mesh              The input volume mesh
		/// @param[in]  quadrature_order  The quadrature order
		/// @param[in]  discr_order       The order of the elements (1 or 2)
		/// @param[out] bases             List of basis functions per element
		/// @param[out] local_boundary    List of descriptor per element,
		///                               indicating which facet of the
		///                               canonical hex lie on the boundary of
		///                               the mesh
		/// @param[out] boundary_nodes    List of dofs which are on the boundary
		///
		/// @return     The number of basis functions created.
		///
		static int build_bases(
			const Mesh3D &mesh,
			const int quadrature_order,
			const int discr_order,
			std::vector< ElementBases > &bases,
			std::vector< LocalBoundary > &local_boundary,
			std::vector< int > &boundary_nodes);

		static void quadr_hex_basis(const int local_index, const Eigen::MatrixXd &xne, Eigen::MatrixXd &val);
		static void quadr_hex_basis_grad(const int local_index, const Eigen::MatrixXd &xne, Eigen::MatrixXd &val);
	};
}

#endif //FE_BASIS_3D_HPP
