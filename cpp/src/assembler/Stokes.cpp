#include <polyfem/Stokes.hpp>

#include <polyfem/Basis.hpp>
#include <polyfem/ElementAssemblyValues.hpp>
#include <polyfem/ElasticityUtils.hpp>

namespace polyfem
{
	void StokesVelocity::set_parameters(const json &params)
	{
		set_size(params["size"]);

		if (params.count("viscosity")) {
			viscosity_ = params["viscosity"];
		}
	}

	void StokesVelocity::set_size(const int size)
	{
		size_ = size;
	}

	Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 9, 1>
	StokesVelocity::assemble(const ElementAssemblyValues &vals, const int i, const int j, const QuadratureVector &da) const
	{
		// (gradi : gradj)  = <gradi, gradj> * Id

		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 9, 1> res(size()*size());
		res.setZero();

		const Eigen::MatrixXd &gradi = vals.basis_values[i].grad_t_m;
		const Eigen::MatrixXd &gradj = vals.basis_values[j].grad_t_m;
		double dot = 0;
		for (int k = 0; k < gradi.rows(); ++k) {
			dot += gradi.row(k).dot(gradj.row(k)) * da(k);
		}

		dot *= viscosity_;

		for(int d = 0; d < size(); ++d)
			res(d*size() + d) = dot;

		return res;
	}

	Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1>
	StokesVelocity::compute_rhs(const AutodiffHessianPt &pt) const
	{
		assert(pt.size() == size());
		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> res(size());
		assert(false);
		return res;
	}




	void StokesMixed::set_parameters(const json &params)
	{
		set_size(params["size"]);
	}

	void StokesMixed::set_size(const int size)
	{
		size_ = size;
	}

	Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1>
	StokesMixed::assemble(const ElementAssemblyValues &psi_vals, const ElementAssemblyValues &phi_vals, const int i, const int j, const QuadratureVector &da) const
	{
		// -(psii : div phij)  = psii * gradphij

		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> res(rows()*cols());
		res.setZero();

		const Eigen::MatrixXd &psii 	= psi_vals.basis_values[i].val;
		const Eigen::MatrixXd &gradphij = phi_vals.basis_values[j].grad_t_m;
		assert(psii.size() == gradphij.rows());
		assert(gradphij.cols() == rows());

		for (int k = 0; k < gradphij.rows(); ++k) {
			res -= psii(k) * gradphij.row(k) * da(k);
		}

		return res;
	}

	Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1>
	StokesMixed::compute_rhs(const AutodiffHessianPt &pt) const
	{
		assert(pt.size() == rows());
		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> res(rows());
		assert(false);
		return res;
	}
}
