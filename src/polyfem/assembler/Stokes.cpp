#include "Stokes.hpp"

namespace polyfem::assembler
{
	void StokesVelocity::add_multimaterial(const int index, const json &params, const Units &units)
	{
		// TODO add units to viscosity
		assert(size() == 2 || size() == 3);

		if (params.count("viscosity"))
		{
			viscosity_ = params["viscosity"];
		}
	}

	Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 9, 1>
	StokesVelocity::assemble(const LinearAssemblerData &data) const
	{
		// (gradi : gradj)  = <gradi, gradj> * Id

		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 9, 1> res(size() * size());
		res.setZero();

		const Eigen::MatrixXd &gradi = data.vals.basis_values[data.i].grad_t_m;
		const Eigen::MatrixXd &gradj = data.vals.basis_values[data.j].grad_t_m;
		double dot = 0;
		for (int k = 0; k < gradi.rows(); ++k)
		{
			dot += gradi.row(k).dot(gradj.row(k)) * data.da(k);
		}

		dot *= viscosity_;

		for (int d = 0; d < size(); ++d)
			res(d * size() + d) = dot;

		return res;
	}

	VectorNd StokesVelocity::compute_rhs(const AutodiffHessianPt &pt) const
	{
		assert(pt.size() == size());
		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> res(size());

		for (int d = 0; d < size(); ++d)
		{
			res(d) = viscosity_ * pt(d).getHessian().trace();
		}

		return res;
	}

	std::map<std::string, Assembler::ParamFunc> StokesVelocity::parameters() const
	{
		std::map<std::string, ParamFunc> res;
		const double nu = this->viscosity();
		res["viscosity"] = [nu](const RowVectorNd &, const RowVectorNd &, double, int) { return nu; };

		return res;
	}

	Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1>
	StokesMixed::assemble(const MixedAssemblerData &data) const
	{
		// -(psii : div phij)  = psii * gradphij

		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> res(rows() * cols());
		res.setZero();

		const Eigen::MatrixXd &psii = data.psi_vals.basis_values[data.i].val;
		const Eigen::MatrixXd &gradphij = data.phi_vals.basis_values[data.j].grad_t_m;
		assert(psii.size() == gradphij.rows());
		assert(gradphij.cols() == rows());

		for (int k = 0; k < gradphij.rows(); ++k)
		{
			res -= psii(k) * gradphij.row(k) * data.da(k);
		}

		return res;
	}
} // namespace polyfem::assembler
