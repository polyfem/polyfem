#ifndef FE_BASIS_2D_HPP
#define FE_BASIS_2D_HPP

#include "ElementBases.hpp"
#include "Mesh2D.hpp"
#include "LocalBoundary.hpp"

#include <Eigen/Dense>
#include <vector>

namespace poly_fem
{
	class FEBasis2d
	{
	public:
		static int build_bases(const Mesh2D &mesh, const int quadrature_order, const int discr_order, std::vector< ElementBases > &bases, std::vector< LocalBoundary > &local_boundary, std::vector< int > &bounday_nodes);
	};
}

#endif //FE_BASIS_2D_HPP
