#ifndef FE_BASIS_3D_HPP
#define FE_BASIS_3D_HPP

#include "Mesh3D.hpp"

#include "ElementBases.hpp"
#include "LocalBoundary.hpp"

#include <Eigen/Dense>
#include <vector>

namespace poly_fem
{
	class FEBasis3d
	{
	public:
		static int build_bases(const Mesh3D &mesh, const int quadrature_order, std::vector< ElementBases > &bases, std::vector< LocalBoundary > &local_boundary, std::vector< int > &bounday_nodes);
	};
}

#endif //FE_BASIS_3D_HPP
