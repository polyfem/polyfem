#include "Laplacian.hpp"

#include <iostream>

namespace polyfem
{
	namespace assembler
	{
		Eigen::Matrix<double, 1, 1> Laplacian::assemble(const ElementAssemblyValues &vals, const int i, const int j, const QuadratureVector &da) const
		{
			const Eigen::MatrixXd &gradi = vals.basis_values[i].grad_t_m;
			const Eigen::MatrixXd &gradj = vals.basis_values[j].grad_t_m;
			// return ((gradi.array() * gradj.array()).rowwise().sum().array() * da.array()).colwise().sum();
			double res = 0;
			for (int k = 0; k < gradi.rows(); ++k)
			{
				res += gradi.row(k).dot(gradj.row(k)) * da(k);
			}
			return Eigen::Matrix<double, 1, 1>::Constant(res);
		}

		Eigen::Matrix<double, 1, 1> Laplacian::compute_rhs(const AutodiffHessianPt &pt) const
		{
			Eigen::Matrix<double, 1, 1> result;
			assert(pt.size() == 1);
			result(0) = pt(0).getHessian().trace();
			return result;
		}

		Eigen::Matrix<AutodiffScalarGrad, Eigen::Dynamic, 1, 0, 3, 1> Laplacian::kernel(const int dim, const AutodiffScalarGrad &r) const
		{
			Eigen::Matrix<AutodiffScalarGrad, Eigen::Dynamic, 1, 0, 3, 1> res(1);

			if (dim == 2)
				res(0) = -1. / (2 * M_PI) * log(r);
			else if (dim == 3)
				res(0) = 1. / (4 * M_PI * r);
			else
				assert(false);

			return res;
		}
	} // namespace assembler
} // namespace polyfem
