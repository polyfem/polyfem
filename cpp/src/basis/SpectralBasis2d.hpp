#pragma once

#include "Mesh2D.hpp"
#include "ElementBases.hpp"
#include "LocalBoundary.hpp"

#include "InterfaceData.hpp"

#include <Eigen/Dense>
#include <vector>
#include <map>

namespace poly_fem
{
	class SpectralBasis2d
	{
	public:

		static int build_bases(
			const Mesh2D &mesh,
			const int quadrature_order,
			const int order,
			std::vector< ElementBases > &bases,
			std::vector< ElementBases > &gbases,
			std::vector< LocalBoundary > &local_boundary);
	};
}

