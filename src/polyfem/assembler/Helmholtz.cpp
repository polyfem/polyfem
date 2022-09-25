#include "Helmholtz.hpp"
#include <polyfem/utils/Bessel.hpp>

#include <iostream>

namespace polyfem::assembler
{
	Eigen::Matrix<double, 1, 1> Helmholtz::assemble(const LinearAssemblerData &data) const
	{
		const Eigen::MatrixXd &gradi = data.vals.basis_values[data.i].grad_t_m;
		const Eigen::MatrixXd &gradj = data.vals.basis_values[data.j].grad_t_m;

		double res = 0;
		for (int k = 0; k < gradi.rows(); ++k)
		{
			res += gradi.row(k).dot(gradj.row(k)) * data.da(k);
		}

		res -= (data.vals.basis_values[data.i].val.array() * data.vals.basis_values[data.j].val.array() * data.da.array()).sum() * k_ * k_;

		return Eigen::Matrix<double, 1, 1>::Constant(res);
	}

	Eigen::Matrix<double, 1, 1> Helmholtz::compute_rhs(const AutodiffHessianPt &pt) const
	{
		Eigen::Matrix<double, 1, 1> result;
		assert(pt.size() == 1);
		result(0) = pt(0).getHessian().trace() + k_ * k_ * pt(0).getValue();
		return result;
	}

	void Helmholtz::add_multimaterial(const int index, const json &params)
	{
		if (params.contains("k"))
		{
			k_ = params["k"];
		}
	}

	Eigen::Matrix<AutodiffScalarGrad, Eigen::Dynamic, 1, 0, 3, 1> Helmholtz::kernel(const int dim, const AutodiffScalarGrad &r) const
	{
		Eigen::Matrix<AutodiffScalarGrad, Eigen::Dynamic, 1, 0, 3, 1> res(1);

		if (dim == 2)
			res(0) = -0.25 * utils::bessy0(k_ * r);
		else if (dim == 3)
			res(0) = 0.25 * cos(k_ * r) / (M_PI * r);
		else
			assert(false);

		return res;
	}
} // namespace polyfem::assembler
