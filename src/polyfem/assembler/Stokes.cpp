#include "Stokes.hpp"

#include <polyfem/basis/Basis.hpp>
#include <polyfem/assembler/ElementAssemblyValues.hpp>
#include <polyfem/utils/ElasticityUtils.hpp>

namespace polyfem::assembler
{
	void StokesVelocity::add_multimaterial(const int index, const json &params)
	{
		assert(size_ == 2 || size_ == 3);

		if (params.count("viscosity"))
		{
			viscosity_ = params["viscosity"];
		}
	}

	void StokesVelocity::set_size(const int size)
	{
		size_ = size;
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

	Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1>
	StokesVelocity::compute_rhs(const AutodiffHessianPt &pt) const
	{
		assert(pt.size() == size());
		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> res(size());

		for (int d = 0; d < size(); ++d)
		{
			res(d) = viscosity_ * pt(d).getHessian().trace();
		}

		return res;
	}

	void StokesVelocity::compute_norm_velocity(const basis::ElementBases &bs, const basis::ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &velocity, Eigen::MatrixXd &norms) const
	{
		norms.resize(local_pts.rows(), 1);
		assert(velocity.cols() == 1);

		Eigen::MatrixXd vel(1, size());

		ElementAssemblyValues vals;
		vals.compute(-1, size() == 3, local_pts, bs, gbs);

		for (long p = 0; p < local_pts.rows(); ++p)
		{
			vel.setZero();

			for (std::size_t j = 0; j < bs.bases.size(); ++j)
			{
				const basis::Basis &b = bs.bases[j];
				const auto &loc_val = vals.basis_values[j];

				assert(bs.bases.size() == vals.basis_values.size());
				assert(loc_val.val.rows() == local_pts.rows());
				assert(loc_val.val.cols() == 1);

				for (int d = 0; d < size(); ++d)
				{
					for (std::size_t ii = 0; ii < b.global().size(); ++ii)
					{
						vel(d) += b.global()[ii].val * loc_val.val(p) * velocity(b.global()[ii].index * size() + d);
					}
				}
			}

			norms(p) = vel.norm();
		}
	}

	void StokesVelocity::compute_stress_tensor(const basis::ElementBases &bs, const basis::ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &velocity, Eigen::MatrixXd &tensor) const
	{
		tensor.resize(local_pts.rows(), size());
		assert(velocity.cols() == 1);

		Eigen::MatrixXd vel(1, size());

		ElementAssemblyValues vals;
		vals.compute(-1, size() == 3, local_pts, bs, gbs);

		for (long p = 0; p < local_pts.rows(); ++p)
		{
			vel.setZero();

			for (std::size_t j = 0; j < bs.bases.size(); ++j)
			{
				const basis::Basis &b = bs.bases[j];
				const auto &loc_val = vals.basis_values[j];

				assert(bs.bases.size() == vals.basis_values.size());
				assert(loc_val.grad.rows() == local_pts.rows());
				assert(loc_val.grad.cols() == size());

				for (int d = 0; d < size(); ++d)
				{
					for (std::size_t ii = 0; ii < b.global().size(); ++ii)
					{
						vel(d) += b.global()[ii].val * loc_val.val(p) * velocity(b.global()[ii].index * size() + d);
					}
				}
			}

			tensor.row(p) = vel;
		}
	}

	void StokesMixed::set_size(const int size)
	{
		size_ = size;
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

	Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1>
	StokesMixed::compute_rhs(const AutodiffHessianPt &pt) const
	{
		assert(pt.size() == rows());
		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> res(rows());
		assert(false);
		return res;
	}
} // namespace polyfem::assembler
