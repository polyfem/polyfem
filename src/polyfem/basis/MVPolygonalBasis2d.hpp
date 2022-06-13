#pragma once

#include <polyfem/mesh/mesh2D/CMesh2D.hpp>
#include <polyfem/mesh/mesh2D/NCMesh2D.hpp>
#include <polyfem/basis/ElementBases.hpp>
#include <polyfem/assembler/ElementAssemblyValues.hpp>
#include <polyfem/basis/InterfaceData.hpp>
#include <polyfem/mesh/LocalBoundary.hpp>

#include <Eigen/Dense>
#include <vector>
#include <map>

namespace polyfem
{
	namespace basis
	{
		class MVPolygonalBasis2d
		{
		public:
			static int build_bases(
				const std::string &assembler_name,
				const mesh::Mesh2D &mesh,
				const int n_bases,
				const int quadrature_order,
				std::vector<ElementBases> &bases,
				const std::vector<ElementBases> &gbases,
				const std::map<int, InterfaceData> &poly_edge_to_data,
				std::vector<mesh::LocalBoundary> &local_boundary,
				std::map<int, Eigen::MatrixXd> &mapped_boundary);

			static void meanvalue(const Eigen::MatrixXd &polygon, const Eigen::RowVector2d &point, Eigen::MatrixXd &b, const double tol);
			static void meanvalue_derivative(const Eigen::MatrixXd &polygon, const Eigen::RowVector2d &point, Eigen::MatrixXd &derivatives, const double tol);
		};
	} // namespace basis
} // namespace polyfem
