#pragma once

#include <polyfem/mesh/mesh2D/CMesh2D.hpp>
#include <polyfem/mesh/mesh2D/NCMesh2D.hpp>
#include <polyfem/assembler/Assembler.hpp>
#include <polyfem/basis/ElementBases.hpp>
#include <polyfem/assembler/ElementAssemblyValues.hpp>
#include <polyfem/basis/InterfaceData.hpp>

#include <Eigen/Dense>
#include <vector>
#include <map>

namespace polyfem
{
	namespace basis
	{
		class PolygonalBasis2d
		{
		public:
			// Compute the integral constraints for each basis of the mesh. This step is PDE-dependent.
			//
			// @param[in]  mesh             Input surface mesh
			// @param[in]  n_bases          Number of bases/nodes in the mesh
			// @param[in]  values           Shape functions for the PDE field evaluated over each
			//                              elements
			// @param[in]  gvalues          Shape functions for the geometric mapping evaluated over
			//                              each elements
			// @param[out] basis_integrals  Integral constraints over polygon that each basis of the
			//                              mesh must verify
			//
			static void compute_integral_constraints(
				const assembler::LinearAssembler &assembler,
				const mesh::Mesh2D &mesh,
				const int n_bases,
				const std::vector<ElementBases> &bases,
				const std::vector<ElementBases> &gbases,
				Eigen::MatrixXd &basis_integrals);

			///
			/// @brief         Build bases over the remaining polygons of a mesh.
			///
			/// @param[in]     n_samples_per_edge     Number of collocation samples per polygon edge (including endpoints)
			/// @param[in]     mesh                   Input surface mesh
			/// @param[in]     n_bases                Total number of bases functions in the mesh
			/// @param[in]     quadrature_order       Quadrature order for the polygons
			/// @param[in]     mass_quadrature_order       Quadrature order for the polygons for mass
			/// @param[in]     integral_constraints   Order of the integral constraints (0 = no constraints, 1 = linear, 2 = quadratic)
			/// @param[in,out] bases                  List of the different basis (shape functions) used to discretize the PDE
			/// @param[in]     gbases                 List of the different basis used to discretize the geometry of the mesh
			/// @param[in]     poly_edge_to_data      Additional data computed for edges at the interface with a polygon
			/// @param[out]    mapped_boundary        Map element id -> #S x dim polyline formed by the collocation points on the boundary of the polygon. The collocation points are mapped through the geometric mapping of the element across the edge, so this polyline may differ from the original polygon.
			/// @param[in]  element_types   Per-element tag indicating the type of each element (see Mesh.hpp)
			/// @param[in]  values          Per-element shape functions for the PDE, evaluated over the element, used for the system matrix assembly (used for linear reproduction)
			/// @param[in]  gvalues         Per-element shape functions for the geometric mapping, evaluated over the element (get boundary of the polygon)
			///
			static int build_bases(
				const assembler::LinearAssembler &assembler,
				const int n_samples_per_edge,
				const mesh::Mesh2D &mesh,
				const int n_bases,
				const int quadrature_order,
				const int mass_quadrature_order,
				const int integral_constraints,
				std::vector<ElementBases> &bases,
				const std::vector<ElementBases> &gbases,
				const std::map<int, InterfaceData> &poly_edge_to_data,
				std::map<int, Eigen::MatrixXd> &mapped_boundary);
		};
	} // namespace basis
} // namespace polyfem
