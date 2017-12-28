#ifndef SPLINE_BASIS_2D_HPP
#define SPLINE_BASIS_2D_HPP

#include "Mesh2D.hpp"
#include "ElementBases.hpp"
#include "LocalBoundary.hpp"

#include <Eigen/Dense>
#include <vector>
#include <map>

namespace poly_fem
{
	class SplineBasis2d
	{
	public:
		static int build_bases(const Mesh2D &mesh, const int quadrature_order, std::vector< ElementBases > &bases, std::vector< LocalBoundary > &local_boundary, std::vector< int > &bounday_nodes, std::map<int, Eigen::MatrixXd> &polys);
		// static void basis(const Eigen::MatrixXd &uv, Eigen::MatrixXd &val);
		// static void grad(const Eigen::MatrixXd &uv, Eigen::MatrixXd &val);
	};
}

#endif //SPLINE_BASIS_2D_HPP
