#include "Bilaplacian.hpp"

#include <polyfem/basis/Basis.hpp>
#include <polyfem/assembler/ElementAssemblyValues.hpp>
#include <polyfem/utils/ElasticityUtils.hpp>

namespace polyfem::assembler
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
	BilaplacianMixed::assemble(const MixedAssemblerData &data) const
	{
		const Eigen::MatrixXd &gradi = data.psi_vals.basis_values[data.i].grad_t_m;
		const Eigen::MatrixXd &gradj = data.phi_vals.basis_values[data.j].grad_t_m;

		// return ((psii.array() * phij.array()).rowwise().sum().array() * da.array()).colwise().sum();
		double res = 0;
		for (int k = 0; k < gradi.rows(); ++k)
		{
			res += gradi.row(k).dot(gradj.row(k)) * data.da(k);
		}
		return Eigen::Matrix<double, 1, 1>::Constant(res);
	}

	Eigen::Matrix<double, 1, 1>
	BilaplacianAux::assemble(const LinearAssemblerData &data) const
	{
		const double tmp = (data.vals.basis_values[data.i].val.array() * data.vals.basis_values[data.j].val.array() * data.da.array()).sum();
		return Eigen::Matrix<double, 1, 1>::Constant(tmp);
	}

} // namespace polyfem::assembler
