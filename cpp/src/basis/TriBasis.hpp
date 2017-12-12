#ifndef TRI_BASIS_HPP
#define TRI_BASIS_HPP

#include "Basis.hpp"

#include <Eigen/Dense>
#include <vector>

namespace poly_fem
{
	class TriBasis
	{
	public:
		static void basis(const int disc_order, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val);
		static void grad(const int disc_order, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val);
	};
}

#endif //TRI_BASIS_HPP
