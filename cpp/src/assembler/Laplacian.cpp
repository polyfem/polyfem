#include "Laplacian.hpp"

#include <iostream>

namespace poly_fem
{
	void Laplacian::assemble(const Eigen::MatrixXd &gradi, const Eigen::MatrixXd &gradj, const Eigen::MatrixXd &da, Eigen::MatrixXd &res) const
	{
		res = (gradi.array() * gradj.array()).rowwise().sum().array() * da.array();
	}

}