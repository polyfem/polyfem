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
		class BarycentricBasis2d
		{
		public:
			static int build_bases(
				const std::string &assembler_name,
				const mesh::Mesh2D &mesh,
				const int n_bases,
				const int quadrature_order,
				const int mass_quadrature_order,
				const std::function<void(const Eigen::MatrixXd &, const Eigen::RowVector2d &, Eigen::MatrixXd &, const double)> bc,
				const std::function<void(const Eigen::MatrixXd &, const Eigen::RowVector2d &, Eigen::MatrixXd &, const double)> bc_prime,
				std::vector<ElementBases> &bases,
				std::vector<mesh::LocalBoundary> &local_boundary,
				std::map<int, Eigen::MatrixXd> &mapped_boundary);
		};
	} // namespace basis
} // namespace polyfem
