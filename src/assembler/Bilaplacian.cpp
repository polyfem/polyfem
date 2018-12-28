#include <polyfem/Bilaplacian.hpp>

#include <polyfem/Basis.hpp>
#include <polyfem/ElementAssemblyValues.hpp>
#include <polyfem/ElasticityUtils.hpp>

namespace polyfem
{



	Eigen::Matrix<double, 1, 1>
	BilaplacianMain::compute_rhs(const AutodiffHessianPt &pt) const
	{
		assert(pt.size() == size());
		Eigen::Matrix<double, 1, 1> res(size());
		assert(false);
		return res;
	}


	Eigen::Matrix<double, 1, 1>
	BilaplacianMixed::assemble(const ElementAssemblyValues &psi_vals, const ElementAssemblyValues &phi_vals, const int i, const int j, const QuadratureVector &da) const
	{
		const Eigen::MatrixXd &gradi = psi_vals.basis_values[i].grad_t_m;
		const Eigen::MatrixXd &gradj = phi_vals.basis_values[j].grad_t_m;

		// return ((psii.array() * phij.array()).rowwise().sum().array() * da.array()).colwise().sum();
		double res = 0;
		for (int k = 0; k < gradi.rows(); ++k) {
			res += gradi.row(k).dot(gradj.row(k)) * da(k);
		}
		return Eigen::Matrix<double, 1, 1>::Constant(res);
	}

	Eigen::Matrix<double, 1, 1>
	BilaplacianAux::assemble(const ElementAssemblyValues &vals, const int i, const int j, const QuadratureVector &da) const
	{
		const double tmp = (vals.basis_values[i].val.array() * vals.basis_values[j].val.array() * da.array()).sum();
		return Eigen::Matrix<double, 1, 1>::Constant(tmp);
	}

}
