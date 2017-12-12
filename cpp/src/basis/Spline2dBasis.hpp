#ifndef SPLINE_2D_BASIS_HPP
#define SPLINE_2D_BASIS_HPP

#include "Mesh.hpp"
#include "Basis.hpp"
#include "LocalBoundary.hpp"

#include <Eigen/Dense>
#include <vector>

namespace poly_fem
{
	class Spline2dBasis
	{
	public:
		static int build_bases(const Mesh &mesh, std::vector< std::vector<Basis> > &bases, std::vector< LocalBoundary > &local_boundary, std::vector< int > &bounday_nodes);
		// static void basis(const Eigen::MatrixXd &uv, Eigen::MatrixXd &val);
		// static void grad(const Eigen::MatrixXd &uv, Eigen::MatrixXd &val);
	};
}

#endif //SPLINE_2D_BASIS_HPP
