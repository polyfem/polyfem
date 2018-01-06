#ifndef FE_BASIS_2D_HPP
#define FE_BASIS_2D_HPP

#include "ElementBases.hpp"
#include "Mesh2D.hpp"
#include "LocalBoundary.hpp"
#include "InterfaceData.hpp"

#include <Eigen/Dense>
#include <vector>

namespace poly_fem
{
	class FEBasis2d
	{
	public:

		///
		/// @brief      Builds FE basis functions over the entire mesh (P1, P2
		///             over triangles, Q1, Q2 over quads). Polygonal facets
		///             with > 4 vertices should be triangulated...
		///
		/// @param[in]  mesh              The input surface mesh
		/// @param[in]  quadrature_order  The quadrature order
		/// @param[in]  discr_order       The order of the elements (1 or 2)
		/// @param[out] bases             List of basis functions per element
		/// @param[out] local_boundary    List of descriptor per element,
		///                               indicating which edge of the canonical
		///                               elements lie on the boundary of the
		///                               mesh
		/// @param[out] boundary_nodes    List of dofs which are on the boundary
		/// @param[out] poly_edge_to_data    Map from edge to InterfaceData for harmonic basis
		///
		/// @return     The number of basis functions created.
		///
		static int build_bases(
			const Mesh2D &mesh,
			const int quadrature_order,
			const int discr_order,
			std::vector<ElementBases> &bases,
			std::vector<LocalBoundary> &local_boundary,
			std::vector<int> &boundary_nodes,
			std::map<int, InterfaceData> &poly_edge_to_data);


		static void quad_basis_basis(const int discr_order, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val);
		static void quad_basis_grad(const int discr_order, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val);
	};
}

#endif //FE_BASIS_2D_HPP
