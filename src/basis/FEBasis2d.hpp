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
		/// @param[in]  mesh               The input planar mesh
		/// @param[in]  quadrature_order   The quadrature order
		/// @param[in]  discr_order        The order of the elements (1-4)
		/// @param[in]  serendipity        Uses serendipity bases or not (only for quads)
		/// @param[in]  has_polys          Does the mesh has polygons, if not the interface mapping is not necessary
		/// @param[in]  is_geom_bases      Flag to decide if build gemetric mapping or normal bases, used to decide if the ndoes are important
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
			const bool serendipity,
			const bool has_polys,
			const bool is_geom_bases,
			std::vector<ElementBases> &bases,
			std::vector<LocalBoundary> &local_boundary,
			std::map<int, InterfaceData> &poly_edge_to_data);

		///
		/// @brief      Builds FE basis functions over the entire mesh (P1, P2 over triangles, Q1,
		///             Q2 over quads). Polygonal facets with > 4 vertices are dealt later on by the
		///             PolygonalBasis2d class.
		///
		/// @param[in]  mesh               The input planar mesh
		/// @param[in]  quadrature_order   The quadrature order
		/// @param[in]  discr_order        The order for each element
		/// @param[in]  serendipity        Uses serendipity bases or not (only for quads)
		/// @param[in]  has_polys          Does the mesh has polygons, if not the interface mapping is not necessary
		/// @param[in]  is_geom_bases      Flag to decide if build gemetric mapping or normal bases, used to decide if the ndoes are important
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
			const Eigen::VectorXi &discr_order,
			const bool serendipity,
			const bool has_polys,
			const bool is_geom_bases,
			std::vector<ElementBases> &bases,
			std::vector<LocalBoundary> &local_boundary,
			std::map<int, InterfaceData> &poly_edge_to_data);

		//return the local edge nodes for a tri or a quad of order p, index points to the edge
		static Eigen::VectorXi tri_edge_local_nodes(const int p, const Mesh2D &mesh, Navigation::Index index);
		static Eigen::VectorXi quad_edge_local_nodes(const int q, const Mesh2D &mesh, Navigation::Index index);
	};
} // namespace polyfem
