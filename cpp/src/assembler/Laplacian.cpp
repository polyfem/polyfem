#include "Laplacian.hpp"

#include <iostream>

namespace poly_fem
{
	Eigen::Matrix<double, 1, 1> Laplacian::assemble(const ElementAssemblyValues &vals, const AssemblyValues &values_i, const AssemblyValues &values_j, const Eigen::VectorXd &da) const
	{
		const Eigen::MatrixXd &gradi = values_i.grad_t_m;
		const Eigen::MatrixXd &gradj = values_j.grad_t_m;
		// return ((gradi.array() * gradj.array()).rowwise().sum().array() * da.array()).colwise().sum();
		double res = 0;
		for (int k = 0; k < gradi.rows(); ++k) {
			res += gradi.row(k).dot(gradj.row(k)) * da(k);
		}
		return Eigen::Matrix<double, 1, 1>::Constant(res);
	}

}
