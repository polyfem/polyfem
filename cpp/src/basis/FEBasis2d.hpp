#ifndef FE_BASIS_2D_HPP
#define FE_BASIS_2D_HPP

#include "ElementBases.hpp"
#include "Mesh2D.hpp"
#include "LocalBoundary.hpp"
#include "InterfaceData.hpp"
#include "Navigation.hpp"

#include <Eigen/Dense>
#include <vector>

namespace poly_fem
{
	class FEBasis2d
	{
	public:

		///
		/// @brief      Builds FE basis functions over the entire mesh (P1, P2 over triangles, Q1,
		///             Q2 over quads). Polygonal facets with > 4 vertices are dealt later on by the
		///             PolygonalBasis2d class.
		///
		/// @param[in]  mesh               The input surface mesh
		/// @param[in]  quadrature_order   The quadrature order
		/// @param[in]  discr_order        The order of the elements (1 or 2)
		/// @param[out] bases              List of basis functions per element
		/// @param[out] local_boundary     List of descriptor per element, indicating which edge of
		///                                the canonical elements lie on the boundary of the mesh
		/// @param[out] boundary_nodes     List of nodes which are on the boundary of the mesh
		/// @param[out] poly_edge_to_data  Data for edges at the interface with a polygon (used to
		///                                build the harmonics inside polygons)
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

		static std::array<int, 2> linear_quad_edge_local_nodes(const Mesh2D &mesh, Navigation::Index index);
		static std::array<int, 3> quadr_quad_edge_local_nodes(const Mesh2D &mesh, Navigation::Index index);

		static Eigen::MatrixXd linear_quad_edge_local_nodes_coordinates(const Mesh2D &mesh, Navigation::Index index);
		static Eigen::MatrixXd quadr_quad_edge_local_nodes_coordinates(const Mesh2D &mesh, Navigation::Index index);

		///
		/// @brief      { Evaluates one local quadratic basis function over a
		///             set of parametric samples in the element }
		///
		/// @param[in]  local_index  { Local index of the basis to evaluate }
		/// @param[in]  uv           { #n x dim matrix with coordinates of the
		///                          parametric samples to evaluate }
		/// @param[out] val          { #n x 1 matrix of computed values}
		///
		static void quadr_quad_basis_value(const int local_index,
			const Eigen::MatrixXd &uv, Eigen::MatrixXd &val);

		///
		/// @brief      { Evaluates the gradient of one local quadratic basis
		///             function over a parametric samples in the element }
		///
		/// @param[in]  discr_order  { Discretization order }
		/// @param[in]  local_index  { Local index of the basis to evaluate }
		/// @param[in]  uv           { #n x dim matrix with coordinates of the
		///                          parametric samples to evaluate }
		/// @param[out] val          { #n x 1 matrix of computed gradients }
		///
		static void quadr_quad_basis_grad(const int local_index,
			const Eigen::MatrixXd &xne, Eigen::MatrixXd &val);

	};
}

#endif //FE_BASIS_2D_HPP
