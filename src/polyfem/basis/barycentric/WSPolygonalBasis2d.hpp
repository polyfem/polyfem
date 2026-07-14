#pragma once

#include <polyfem/mesh/mesh2D/Mesh2D.hpp>
#include <polyfem/assembler/AssemblyData.hpp>
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
				const int dim,
				const mesh::Mesh2D &mesh,
				const int n_bases,
				const int quadrature_order,
				const int mass_quadrature_order,
				std::vector<ElementBases> &bases,
				std::vector<mesh::LocalBoundary> &local_boundary,
				std::map<int, Eigen::MatrixXd> &mapped_boundary);

			static int build_bases(
				const std::string &assembler_name,
				int dim,
				const mesh::Mesh2D &mesh,
				int n_bases,
				int quadrature_order,
				int mass_quadrature_order,
				std::vector<ElementBases> &bases,
				assembler::AssemblyData &assembly_data,
				std::vector<mesh::LocalBoundary> &local_boundary,
				std::map<int, Eigen::MatrixXd> &mapped_boundary);

			static void wachspress(const Eigen::MatrixXd &polygon, const Eigen::RowVector2d &point, Eigen::MatrixXd &b, const double tol);
			static void wachspress_derivative(const Eigen::MatrixXd &polygon, const Eigen::RowVector2d &point, Eigen::MatrixXd &derivatives, const double tol);
		};
	} // namespace basis
} // namespace polyfem
