#pragma once

#include <polyfem/Mesh2D.hpp>
#include <polyfem/ElementBases.hpp>
#include <polyfem/ElementAssemblyValues.hpp>
#include <polyfem/InterfaceData.hpp>

#include <Eigen/Dense>
#include <vector>
#include <map>


namespace polyfem
{

	class MVPolygonalBasis2d
	{
	public:
		static int build_bases(
			const std::string &assembler_name,
			const Mesh2D &mesh,
			const int n_bases,
			const int quadrature_order,
			std::vector< ElementBases > &bases,
			const std::vector< ElementBases > &gbases,
			const  std::map<int, InterfaceData> &poly_edge_to_data,
			std::map<int, Eigen::MatrixXd> &mapped_boundary);
	};
}

