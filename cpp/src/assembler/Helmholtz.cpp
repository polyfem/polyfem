#include "Helmholtz.hpp"

#include <iostream>

namespace poly_fem
{
	Eigen::Matrix<double, 1, 1> Helmholtz::assemble(const ElementAssemblyValues &vals, const int i, const int j, const Eigen::VectorXd &da) const
	{
		const Eigen::MatrixXd &gradi = vals.basis_values[i].grad_t_m;
		const Eigen::MatrixXd &gradj = vals.basis_values[j].grad_t_m;

		double res = 0;
		for (int k = 0; k < gradi.rows(); ++k) {
			res += gradi.row(k).dot(gradj.row(k)) * da(k);
		}

		res += (vals.basis_values[i].val.array() * vals.basis_values[j].val.array() * da.array()).sum() * k_;

		return Eigen::Matrix<double, 1, 1>::Constant(res);
	}

	Eigen::Matrix<double, 1, 1> Helmholtz::compute_rhs(const AutodiffHessianPt &pt) const
	{
		Eigen::Matrix<double, 1, 1> result;
		assert(pt.size() == 1);
		result(0) = pt(0).getHessian().trace() + k_ * pt(0).getValue();
		return result;
	}

	void Helmholtz::set_parameters(const json &params)
	{
		if (params.find("k") != params.end()) {
			k_ = params["k"];
		}
	}

}
