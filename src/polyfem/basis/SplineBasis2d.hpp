#pragma once

#include <polyfem/mesh/mesh2D/CMesh2D.hpp>
#include <polyfem/mesh/mesh2D/NCMesh2D.hpp>
#include <polyfem/basis/ElementBases.hpp>
#include <polyfem/mesh/LocalBoundary.hpp>

#include <polyfem/basis/InterfaceData.hpp>

#include <Eigen/Dense>
#include <vector>
#include <map>

namespace polyfem
{
	namespace basis
	{
		class SplineBasis2d
		{
		public:
			static int build_bases(
				const mesh::Mesh2D &mesh,
				const std::string &assembler,
				const int quadrature_order,
				const int mass_quadrature_order,
				std::vector<ElementBases> &bases,
				std::vector<mesh::LocalBoundary> &local_boundary,
				std::map<int, InterfaceData> &poly_edge_to_data);

			static void fit_nodes(const mesh::Mesh2D &mesh, const int n_bases, std::vector<ElementBases> &gbases);
		};
	} // namespace basis
} // namespace polyfem
