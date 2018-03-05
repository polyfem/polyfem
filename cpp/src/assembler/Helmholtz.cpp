#include "Helmholtz.hpp"

#include <iostream>

namespace poly_fem
{
	Eigen::Matrix<double, 1, 1> Helmholtz::assemble(const ElementAssemblyValues &vals, const int i, const int j, const Eigen::VectorXd &da) const
	{
		const Eigen::MatrixXd &gradi = vals.basis_values[i].grad_t_m;
		const Eigen::MatrixXd &gradj = vals.basis_values[j].grad_t_m;
		// return ((gradi.array() * gradj.array()).rowwise().sum().array() * da.array()).colwise().sum();
		double res = 0;
		for (int k = 0; k < gradi.rows(); ++k) {
			res += gradi.row(k).dot(gradj.row(k)) * da(k);
		}

		res += (vals.basis_values[i].val.array() * vals.basis_values[i].val.array() * da.array()).sum() * k_;

		return Eigen::Matrix<double, 1, 1>::Constant(res);
	}

	void Helmholtz::set_parameters(const json &params)
	{
		if (params.find("k") != params.end()) {
			k_ = params["k"];
		}
	}

}
