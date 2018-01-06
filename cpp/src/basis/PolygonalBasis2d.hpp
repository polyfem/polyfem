#ifndef POLYGONAL_BASIS_HPP
#define POLYGONAL_BASIS_HPP

#include "Mesh2D.hpp"
#include "ElementBases.hpp"
#include "ElementAssemblyValues.hpp"

#include <Eigen/Dense>
#include <vector>
#include <map>


namespace poly_fem
{


	struct BoundaryData
	{
		int face_id = -1;
		int flag;
		std::vector<int> node_id;

		std::vector<int> local_indices;
		std::vector<double> vals;
	};

	class PolygonalBasis2d
	{
	public:
		static const int LEFT_FLAG = 1;
		static const int TOP_FLAG = 2;
		static const int RIGHT_FLAG = 4;
		static const int BOTTOM_FLAG = 8;

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
			std::map<int, BoundaryData> &poly_edge_to_data,
			std::map<int, Eigen::MatrixXd> &polys);
	};
}
#endif //POLYGONAL_BASIS_HPP

