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

		//	Given a 3d navigation index (v0, e0, f0, c0), this function returns
		//	the local node indices on the face f0. If the Q2 nodes are labeled
		//	as follows:
		//
		// v3─────e2─────v2
		//  │      ┆      │
		//  │      ┆      │
		//  │      ┆      │
		// e3┄┄┄┄┄f0┄┄┄┄┄e1
		//  │      ┆      │
		//  │      ┆      │
		//  │      ┆      │
		// v0─────e0─────v1
		//
		// Then this functions returns the local node indices in the following order:
		// (v0, e0, v1, e1, v2, e2, v3, e3, f0)


		//	Given a 3d navigation index (v0, e0, f0, c0), this function returns
		//	the local node indices on the face f0. If the Q2 nodes are labeled
		//	as follows:
		//
		// v2
		//  │╲
		//  │ ╲
		//  │  ╲
		// e2┄┄e1
		//  │   ┆╲
		//  │   ┆ ╲
		// v0──e0──v1
		//
		//
		// Then this functions returns the local node indices in the following order:
		// (v0, e0, v1, e1, v2, e2)








		static std::array<int, 3> linear_tet_face_local_nodes(const Mesh3D &mesh, Navigation3D::Index index);
		static std::array<int, 6> quadr_tet_face_local_nodes(const Mesh3D &mesh, Navigation3D::Index index);

		static std::array<int, 4> linear_hex_face_local_nodes(const Mesh3D &mesh, Navigation3D::Index index);
		static std::array<int, 9> quadr_hex_face_local_nodes(const Mesh3D &mesh, Navigation3D::Index index);



		static Eigen::MatrixXd linear_tet_face_local_nodes_coordinates(const Mesh3D &mesh, Navigation3D::Index index);
		static Eigen::MatrixXd quadr_tet_face_local_nodes_coordinates(const Mesh3D &mesh, Navigation3D::Index index);

		static Eigen::MatrixXd linear_hex_face_local_nodes_coordinates(const Mesh3D &mesh, Navigation3D::Index index);
		static Eigen::MatrixXd quadr_hex_face_local_nodes_coordinates(const Mesh3D &mesh, Navigation3D::Index index);




		static Eigen::MatrixXd tet_local_node_coordinates_from_face(int lf);
		static Eigen::MatrixXd hex_local_node_coordinates_from_face(int lf);



		static void quadr_hex_basis_value(const int local_index, const Eigen::MatrixXd &xne, Eigen::MatrixXd &val);
		static void quadr_hex_basis_grad(const int local_index, const Eigen::MatrixXd &xne, Eigen::MatrixXd &val);

		static void tet_local_to_global(const int p, const Mesh3D &mesh, int c, const Eigen::VectorXi &discr_order, std::vector<int> &res, polyfem::MeshNodes &nodes);
		static Eigen::VectorXi tet_face_local_nodes(const int p, const Mesh3D &mesh, Navigation3D::Index index);

		static std::array<int, 10> quadr_tet_local_to_global(const Mesh3D &mesh, int c);
		static std::array<int, 27> quadr_hex_local_to_global(const Mesh3D &mesh, int c);

		static Eigen::RowVector3d quadr_tet_local_node_coordinates(int local_index);
		static Eigen::RowVector3d quadr_hex_local_node_coordinates(int local_index);
	};
}

