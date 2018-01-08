#ifndef POLYGONAL_BASIS_HPP
#define POLYGONAL_BASIS_HPP

#include "Mesh2D.hpp"
#include "ElementBases.hpp"
#include "ElementAssemblyValues.hpp"
#include "InterfaceData.hpp"

#include <Eigen/Dense>
#include <vector>
#include <map>


namespace poly_fem
{

	class PolygonalBasis2d
	{
	public:

		///
		/// @brief         Build bases over the remaining polygons of a mesh.
		///
		/// @param[in]     samples_res        { Number of samples per edge }
		/// @param[in]     mesh               { Input surface mesh }
		/// @param[in]     n_bases            { The number of basis functions
		///                                   that have already been created }
		/// @param[in]     els_tag            { Per-element tag indicating the
		///                                   type of each element (see
		///                                   Mesh.hpp) }
		/// @param[in]     quadrature_order   { Quadrature order for the
		///                                   polygons }
		/// @param[in]     values             { Per-element shape functions for
		///                                   the PDE, evaluated over the
		///                                   element, used for the system
		///                                   matrix assembly (used for linear
		///                                   reproduction) }
		/// @param[in]     gvalues            { Per-element shape functions for
		///                                   the geometric mapping, evaluated
		///                                   over the element (get boundary of
		///                                   the polygon) }
		/// @param[in,out] bases              { List of the different basis
		///                                   (shape functions) used to
		///                                   discretize the PDE }
		/// @param[in]     gbases             { List of the different basis used
		///                                   to discretize the geometry of the
		///                                   mesh }
		/// @param[out]    poly_edge_to_data  { Additional data computed for
		///                                   edges at the interface with a
		///                                   polygon }
		/// @param[out]    polys              { Map element id -> #S x dim set
		///                                   of evaluation samples on the
		///                                   boundary of the polygon }
		///
		static void build_bases(
			const int samples_res,
			const Mesh2D &mesh,
			const int n_bases,
			const std::vector<ElementType> &els_tag,
			const int quadrature_order,
			const std::vector< ElementAssemblyValues > &values,
			const std::vector< ElementAssemblyValues > &gvalues,
			std::vector< ElementBases > &bases,
			const std::vector< ElementBases > &gbases,
			std::map<int, InterfaceData> &poly_edge_to_data,
			std::map<int, Eigen::MatrixXd> &polys);
	};
}
#endif //POLYGONAL_BASIS_HPP

