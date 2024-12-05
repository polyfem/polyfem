#pragma once

#include <polyfem/basis/ElementBases.hpp>
#include <polyfem/mesh/mesh3D/CMesh3D.hpp>
#include <polyfem/mesh/mesh3D/NCMesh3D.hpp>
#include <polyfem/mesh/LocalBoundary.hpp>
#include <polyfem/basis/InterfaceData.hpp>
#include <polyfem/mesh/mesh3D/Navigation3D.hpp>
#include <polyfem/mesh/MeshNodes.hpp>

#include <Eigen/Dense>
#include <vector>

namespace polyfem
{
	namespace basis
	{
		class LagrangeBasis3d
		{
		public:
			///
			/// @brief      Builds FE basis functions over the entire mesh (P1, P2 over tets, Q1,
			///             Q2 over hes). Polygonal facets with > 4 vertices are dealt later on by the
			///             PolygonalBasis3d class.
			///
			/// @param[in]  mesh               The input volumetric mesh
			/// @param[in]  assembler          The pde to solve
			/// @param[in]  quadrature_order   The quadrature order
			/// @param[in]  mass_quadrature_order   The quadrature order for mass
			/// @param[in]  discr_order        The order of the elements (1-4)
			/// @param[in]  serendipity        Uses serendipity bases or not (only for hex)
			/// @param[in]  has_polys          Does the mesh has polygons, if not the interface mapping is not necessary
			/// @param[in]  is_geom_bases      Flag to decide if build geometric mapping or normal bases, used to decide if the nodes are important
			/// @param[out] bases              List of basis functions per element
			/// @param[out] local_boundary     List of descriptor per element, indicating which edge of
			///                                the canonical elements lie on the boundary of the mesh
			/// @param[out] poly_edge_to_data  Data for edges at the interface with a polygon (used to
			///                                build the harmonics inside polygons)
			///
			/// @return     The number of basis functions created.
			///
			static int build_bases(
				const mesh::Mesh3D &mesh,
				const std::string &assembler,
				const int quadrature_order,
				const int mass_quadrature_order,
				const int discr_order,
				const bool serendipity,
				const bool has_polys,
				const bool is_geom_bases,
				const bool use_corner_quadrature,
				std::vector<ElementBases> &bases,
				std::vector<mesh::LocalBoundary> &local_boundary,
				std::map<int, InterfaceData> &poly_face_to_data,
				std::shared_ptr<mesh::MeshNodes> &mesh_nodes);

			///
			/// @brief      Builds FE basis functions over the entire mesh (P1, P2 over tets, Q1,
			///             Q2 over hex). Polygonal facets with > 4 vertices are dealt later on by the
			///             PolygonalBasis3d class.
			///
			/// @param[in]  mesh               The input volumetric mesh
			/// @param[in]  assembler          The pde to solve
			/// @param[in]  quadrature_order   The quadrature order
			/// @param[in]  mass_quadrature_order   The quadrature order for mass
			/// @param[in]  discr_order        The order for each element
			/// @param[in]  serendipity        Uses serendipity bases or not (only for hex)
			/// @param[in]  has_polys          Does the mesh has polygons, if not the interface mapping is not necessary
			/// @param[in]  is_geom_bases      Flag to decide if build geometric mapping or normal bases, used to decide if the nodes are important
			/// @param[out] bases              List of basis functions per element
			/// @param[out] local_boundary     List of descriptor per element, indicating which edge of
			///                                the canonical elements lie on the boundary of the mesh
			/// @param[out] poly_edge_to_data  Data for edges at the interface with a polygon (used to
			///                                build the harmonics inside polygons)
			///
			/// @return     The number of basis functions created.
			///
			static int build_bases(
				const mesh::Mesh3D &mesh,
				const std::string &assembler,
				const int quadrature_order,
				const int mass_quadrature_order,
				const Eigen::VectorXi &discr_order,
				const bool serendipity,
				const bool has_polys,
				const bool is_geom_bases,
				const bool use_corner_quadrature,
				std::vector<ElementBases> &bases,
				std::vector<mesh::LocalBoundary> &local_boundary,
				std::map<int, InterfaceData> &poly_face_to_data,
				std::shared_ptr<mesh::MeshNodes> &mesh_nodes);

			// return the local faces nodes for a tet or a hex of order p, index points to a face
			static Eigen::VectorXi tet_face_local_nodes(const int p, const mesh::Mesh3D &mesh, mesh::Navigation3D::Index index);
			static Eigen::VectorXi hex_face_local_nodes(const bool serendipity, const int q, const mesh::Mesh3D &mesh, mesh::Navigation3D::Index index);

		private:
			static Eigen::MatrixXd linear_hex_face_local_nodes_coordinates(const mesh::Mesh3D &mesh, mesh::Navigation3D::Index index);

			static Eigen::RowVector3d quadr_hex_local_node_coordinates(int local_index);
		};
	} // namespace basis
} // namespace polyfem
