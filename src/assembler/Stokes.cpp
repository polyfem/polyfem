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

		for(int d = 0; d < size(); ++d)
		{
			res(d) = viscosity_ * pt(d).getHessian().trace();
		}

		return res;
	}

	void StokesVelocity::compute_norm_velocity(const ElementBases &bs, const ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &velocity, Eigen::MatrixXd &norms) const
	{
		norms.resize(local_pts.rows(), 1);
		assert(velocity.cols() == 1);

		Eigen::MatrixXd vel(1, size());

		ElementAssemblyValues vals;
		vals.compute(-1, size() == 3, local_pts, bs, gbs);

		for(long p = 0; p < local_pts.rows(); ++p)
		{
			vel.setZero();

			for(std::size_t j = 0; j < bs.bases.size(); ++j)
			{
				const Basis &b = bs.bases[j];
				const auto &loc_val = vals.basis_values[j];

				assert(bs.bases.size() == vals.basis_values.size());
				assert(loc_val.val.rows() == local_pts.rows());
				assert(loc_val.val.cols() == 1);

				for(int d = 0; d < size(); ++d)
				{
					for(std::size_t ii = 0; ii < b.global().size(); ++ii)
					{
						vel(d) += b.global()[ii].val * loc_val.val(p) * velocity(b.global()[ii].index*size() + d);
					}
				}
			}

			norms(p) = vel.norm();
		}
	}

	void StokesVelocity::compute_stress_tensor(const ElementBases &bs, const ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &velocity, Eigen::MatrixXd &tensor) const
	{
		tensor.resize(local_pts.rows(), size());
		assert(velocity.cols() == 1);

		Eigen::MatrixXd vel(1, size());

		ElementAssemblyValues vals;
		vals.compute(-1, size() == 3, local_pts, bs, gbs);

		for(long p = 0; p < local_pts.rows(); ++p)
		{
			vel.setZero();

			for(std::size_t j = 0; j < bs.bases.size(); ++j)
			{
				const Basis &b = bs.bases[j];
				const auto &loc_val = vals.basis_values[j];

				assert(bs.bases.size() == vals.basis_values.size());
				assert(loc_val.grad.rows() == local_pts.rows());
				assert(loc_val.grad.cols() == size());

				for(int d = 0; d < size(); ++d)
				{
					for(std::size_t ii = 0; ii < b.global().size(); ++ii)
					{
						vel(d) += b.global()[ii].val * loc_val.val(p) * velocity(b.global()[ii].index*size() + d);
					}
				}
			}

			tensor.row(p) = vel;
		}
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
