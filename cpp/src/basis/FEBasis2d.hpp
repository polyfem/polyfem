#pragma once

#include <polyfem/ElementBases.hpp>
#include <polyfem/Mesh2D.hpp>
#include <polyfem/LocalBoundary.hpp>
#include <polyfem/InterfaceData.hpp>
#include <polyfem/Navigation.hpp>

#include <polyfem/MeshNodes.hpp>

#include <Eigen/Dense>
#include <vector>

namespace polyfem
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
		/// @param[out] poly_edge_to_data  Data for edges at the interface with a polygon (used to
		///                                build the harmonics inside polygons)
		///
		/// @return     The number of basis functions created.
		///
		static int build_bases(
			const Mesh2D &mesh,
			const int quadrature_order,
			const int discr_order,
			const bool has_polys,
			std::vector<ElementBases> &bases,
			std::vector<LocalBoundary> &local_boundary,
			std::map<int, InterfaceData> &poly_edge_to_data);

		static int build_bases(
			const Mesh2D &mesh,
			const int quadrature_order,
			const Eigen::VectorXi &discr_order,
			const bool has_polys,
			std::vector<ElementBases> &bases,
			std::vector<LocalBoundary> &local_boundary,
			std::map<int, InterfaceData> &poly_edge_to_data);



		static Eigen::VectorXi tri_edge_local_nodes(const int p, const Mesh2D &mesh, Navigation::Index index);
		static Eigen::VectorXi quad_edge_local_nodes(const int q, const Mesh2D &mesh, Navigation::Index index);
	};
}

