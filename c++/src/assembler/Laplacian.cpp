#include "Laplacian.hpp"

namespace poly_fem
{
	void Laplacian::assemble(const Eigen::MatrixXd &gradi, const Eigen::MatrixXd &gradj, Eigen::MatrixXd &res) const
	{
		res = (gradi.array() * gradj.array()).rowwise().sum();
	}

}