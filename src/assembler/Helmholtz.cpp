#include <polyfem/Helmholtz.hpp>
#include <polyfem/Bessel.hpp>

#include <iostream>

namespace polyfem
{
	Eigen::Matrix<double, 1, 1> Helmholtz::assemble(const ElementAssemblyValues &vals, const int i, const int j, const QuadratureVector &da) const
	{
		const Eigen::MatrixXd &gradi = vals.basis_values[i].grad_t_m;
		const Eigen::MatrixXd &gradj = vals.basis_values[j].grad_t_m;

		double res = 0;
		for (int k = 0; k < gradi.rows(); ++k) {
			res += gradi.row(k).dot(gradj.row(k)) * da(k);
		}

		res -= (vals.basis_values[i].val.array() * vals.basis_values[j].val.array() * da.array()).sum() * k_* k_;

		return Eigen::Matrix<double, 1, 1>::Constant(res);
	}

	Eigen::Matrix<double, 1, 1> Helmholtz::compute_rhs(const AutodiffHessianPt &pt) const
	{
		Eigen::Matrix<double, 1, 1> result;
		assert(pt.size() == 1);
		result(0) = pt(0).getHessian().trace() + k_ * k_ * pt(0).getValue();
		return result;
	}

	void Helmholtz::set_parameters(const json &params)
	{
		if (params.contains("k")) {
			k_ = params["k"];
		}
	}

	Eigen::Matrix<AutodiffScalarGrad, Eigen::Dynamic, 1, 0, 3, 1> Helmholtz::kernel(const int dim, const AutodiffScalarGrad &r) const
	{
		Eigen::Matrix<AutodiffScalarGrad, Eigen::Dynamic, 1, 0, 3, 1> res(1);

		if(dim == 2)
			res(0) = -0.25*bessy0(k_*r);
		else if(dim == 3)
			res(0) = 0.25*cos(k_*r)/(M_PI*r);
		else
			assert(false);

		return res;
	}

}
