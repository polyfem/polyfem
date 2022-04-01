#ifndef SPLINE_BASIS_2D_HPP
#define SPLINE_BASIS_2D_HPP

#include <polyfem/CMesh2D.hpp>
#include <polyfem/NCMesh2D.hpp>
#include <polyfem/ElementBases.hpp>
#include <polyfem/LocalBoundary.hpp>

#include <polyfem/InterfaceData.hpp>

#include <Eigen/Dense>
#include <vector>
#include <map>

namespace polyfem
{
	class SplineBasis2d
	{
	public:

		static int build_bases(
			const Mesh2D &mesh,
			const int quadrature_order,
			std::vector< ElementBases > &bases,
			std::vector< LocalBoundary > &local_boundary,
			std::map<int, InterfaceData> &poly_edge_to_data);

		static void fit_nodes(const Mesh2D &mesh, const int n_bases, std::vector< ElementBases > &gbases);
	};
}

#endif //SPLINE_BASIS_2D_HPP
