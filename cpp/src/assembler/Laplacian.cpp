#include "Laplacian.hpp"

#include <iostream>

namespace poly_fem
{
	Eigen::Matrix<double, 1, 1> Laplacian::assemble(const Eigen::MatrixXd &gradi, const Eigen::MatrixXd &gradj, const Eigen::VectorXd &da) const
	{
		// return ((gradi.array() * gradj.array()).rowwise().sum().array() * da.array()).colwise().sum();
		double res = 0;
		for (int k = 0; k < gradi.rows(); ++k) {
			res += gradi.row(k).dot(gradj.row(k)) * da(k);
		}
		return Eigen::Matrix<double, 1, 1>::Constant(res);
	}

}
