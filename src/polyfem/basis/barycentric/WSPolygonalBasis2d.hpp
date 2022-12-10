#pragma once

#include <polyfem/mesh/mesh2D/Mesh2D.hpp>
#include <polyfem/basis/ElementBases.hpp>
#include <polyfem/mesh/LocalBoundary.hpp>

#include <Eigen/Dense>
#include <vector>

namespace polyfem
{
	namespace basis
	{
		class WSPolygonalBasis2d
		{
		public:
			static int build_bases(
				const std::string &assembler_name,
				const mesh::Mesh2D &mesh,
				const int n_bases,
				const int quadrature_order,
				const int mass_quadrature_order,
				std::vector<ElementBases> &bases,
				std::vector<mesh::LocalBoundary> &local_boundary,
				std::map<int, Eigen::MatrixXd> &mapped_boundary);

			static void wachspress(const Eigen::MatrixXd &polygon, const Eigen::RowVector2d &point, Eigen::MatrixXd &b, const double tol);
			static void wachspress_derivative(const Eigen::MatrixXd &polygon, const Eigen::RowVector2d &point, Eigen::MatrixXd &derivatives, const double tol);
		};
	} // namespace basis
} // namespace polyfem
