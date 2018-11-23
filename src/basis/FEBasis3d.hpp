#pragma once

#include <polyfem/ElementBases.hpp>
#include <polyfem/Mesh3D.hpp>
#include <polyfem/LocalBoundary.hpp>
#include <polyfem/InterfaceData.hpp>
#include <polyfem/Navigation3D.hpp>
#include <polyfem/MeshNodes.hpp>

#include <Eigen/Dense>
#include <vector>

namespace polyfem
{
	class FEBasis3d
	{
	public:

		///
		/// @brief      Builds FE basis functions over the entire mesh (Q1, Q2).
		///
		/// @param[in]  mesh               The input volume mesh
		/// @param[in]  quadrature_order   The quadrature order
		/// @param[in]  discr_order        The order of the elements (1 or 2)
		/// @param[out] bases              List of basis functions per element
		/// @param[out] local_boundary     List of descriptor per element, indicating which facet of
		///                                the canonical hex lie on the boundary of the mesh
		/// @param      poly_face_to_data  Data for faces at the interface with a polyhedra
		///
		/// @return     The number of basis functions created.
		///
		static int build_bases(
			const Mesh3D &mesh,
			const int quadrature_order,
			const int discr_order,
			const bool has_polys,
			std::vector< ElementBases > &bases,
			std::vector< LocalBoundary > &local_boundary,
			std::map<int, InterfaceData> &poly_face_to_data);

		static int build_bases(
			const Mesh3D &mesh,
			const int quadrature_order,
			const Eigen::VectorXi &discr_order,
			const bool has_polys,
			std::vector< ElementBases > &bases,
			std::vector< LocalBoundary > &local_boundary,
			std::map<int, InterfaceData> &poly_face_to_data);

		static Eigen::VectorXi tet_face_local_nodes(const int p, const Mesh3D &mesh, Navigation3D::Index index);
		static Eigen::VectorXi hex_face_local_nodes(const int q, const Mesh3D &mesh, Navigation3D::Index index);
	private:

		static Eigen::MatrixXd linear_hex_face_local_nodes_coordinates(const Mesh3D &mesh, Navigation3D::Index index);




		static Eigen::RowVector3d quadr_hex_local_node_coordinates(int local_index);
	};
}

