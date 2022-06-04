#pragma once

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
	namespace basis
	{
		class SpectralBasis2d
		{
		public:
			static int build_bases(
				const mesh::Mesh2D &mesh,
				const int quadrature_order,
				const int order,
				std::vector<ElementBases> &bases,
				std::vector<ElementBases> &gbases,
				std::vector<mesh::LocalBoundary> &local_boundary);
		};
	} // namespace basis
} // namespace polyfem
