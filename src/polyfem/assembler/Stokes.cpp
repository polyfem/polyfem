#include "Stokes.hpp"

namespace polyfem::assembler
{
	StokesVelocity::StokesVelocity()
		: viscosity_("viscosity")
	{
	}
	void StokesVelocity::add_multimaterial(const int index, const json &params, const Units &units)
	{
		assert(domain_size() == 2 || domain_size() == 3);

		viscosity_.add_multimaterial(index, params, units.viscosity());
	}

	FlatMatrixNd StokesVelocity::assemble(const LinearAssemblerData &data) const
	{
		// (gradi : gradj)  = <gradi, gradj> * Id

		FlatMatrixNd res(codomain_size() * codomain_size());
		res.setZero();

		const Eigen::MatrixXd &gradi = data.vals.basis_values[data.i].grad_t_m;
		const Eigen::MatrixXd &gradj = data.vals.basis_values[data.j].grad_t_m;
		double dot = 0;
		for (int k = 0; k < gradi.rows(); ++k)
		{
			dot += gradi.row(k).dot(gradj.row(k)) * data.da(k) * viscosity_(data.vals.val.row(k), data.t, data.vals.element_id);
		}

		for (int d = 0; d < codomain_size(); ++d)
		{
			res(d * codomain_size() + d) = dot;
		}

		return res;
	}

	VectorNd StokesVelocity::compute_rhs(const AutodiffHessianPt &pt) const
	{
		assert(pt.size() == codomain_size());
		VectorNd res(codomain_size());

		VectorNd val(codomain_size());
		for (int d = 0; d < codomain_size(); ++d)
		{
			val(d) = pt(d).getValue();
		}

		const auto nu = viscosity_(val, 0, 0);
		for (int d = 0; d < codomain_size(); ++d)
		{
			res(d) = nu * pt(d).getHessian().trace();
		}

		return res;
	}

	std::map<std::string, Assembler::ParamFunc> StokesVelocity::parameters() const
	{
		std::map<std::string, ParamFunc> res;
		const auto &nu = viscosity_;
		res["viscosity"] = [&nu](const RowVectorNd &, const RowVectorNd &p, double t, int e) {
			return nu(p, t, e);
		};

		return res;
	}

	VectorNd StokesMixed::assemble(const MixedAssemblerData &data) const
	{
		// -(psii : div phij)  = psii * gradphij

		VectorNd res = VectorNd::Zero(rows() * cols());

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
