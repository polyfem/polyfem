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
		/// @brief      Build bases over the remaining polygons of a mesh.
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

