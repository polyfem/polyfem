#include "Bilaplacian.hpp"

namespace polyfem::assembler
{
	Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1>
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

	Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 9, 1>
	BilaplacianAux::assemble(const LinearAssemblerData &data) const
	{
		const double tmp = (data.vals.basis_values[data.i].val.array() * data.vals.basis_values[data.j].val.array() * data.da.array()).sum();
		return Eigen::Matrix<double, 1, 1>::Constant(tmp);
	}

} // namespace polyfem::assembler
