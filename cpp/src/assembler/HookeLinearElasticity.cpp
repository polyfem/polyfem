#include "HookeLinearElasticity.hpp"

#include "Basis.hpp"
#include "ElementAssemblyValues.hpp"

namespace poly_fem
{

	namespace
	{
		double von_mises_stress_for_stress_tensor(const Eigen::MatrixXd &stress)
		{
			double von_mises_stress =  0.5 * ( stress(0, 0) - stress(1, 1) ) * ( stress(0, 0) - stress(1, 1) ) + 3.0  *  stress(0, 1) * stress(0, 1);

			if(stress.rows() == 3)
			{
				von_mises_stress += 0.5 * (stress(2, 2) - stress(1, 1)) * (stress(2, 2) - stress(1, 1)) + 3.0  * stress(2, 1) * stress(2, 1);
				von_mises_stress += 0.5 * (stress(2, 2) - stress(0, 0)) * (stress(2, 2) - stress(0, 0)) + 3.0  * stress(2, 0) * stress(2, 0);
			}

			von_mises_stress = sqrt( fabs(von_mises_stress) );

			return von_mises_stress;
		}

		Eigen::Matrix2d strain2d(const Eigen::MatrixXd &grad, const Eigen::MatrixXd &jac_it, int k, int coo)
		{
			Eigen::Matrix2d jac;
			jac.setZero();
			jac.row(coo) = grad.row(k);

			jac = jac*jac_it;

			return 0.5*(jac + jac.transpose());
		}

		Eigen::Matrix3d strain3d(const Eigen::MatrixXd &grad, const Eigen::MatrixXd &jac_it, int k, int coo)
		{
			Eigen::Matrix3d jac;
			jac.setZero();
			jac.row(coo) = grad.row(k);

			jac = jac*jac_it;

			return 0.5*(jac + jac.transpose());
		}
	}



	HookeLinearElasticity::HookeLinearElasticity()
	{
		set_size(size_);
	}

	void HookeLinearElasticity::set_size(const int size)
	{
		if(size == 2)
			stifness_tensor_.resize(6, 1);
		else
			stifness_tensor_.resize(21, 1);

		size_ = size;
	}

	double HookeLinearElasticity::stress2d(const std::array<double, 3> &strain, const int j) const
	{
		double res = 0;

		for(int k = 0; k < 3; ++k)
			res += stifness_tensor(j, k)*strain[k];

		return res;
	}

	double HookeLinearElasticity::stress3d(const std::array<double, 6> &strain, const int j) const
	{
		double res = 0;

		for(int k = 0; k < 6; ++k)
			res += stifness_tensor(j, k)*strain[k];

		return res;
	}

	void HookeLinearElasticity::set_stiffness_tensor(int i, int j, const double val)
	{
		if(j < i)
		{
			int tmp=i;
			i = j;
			j = tmp;
		}
		assert(j>=i);
		const int n = size_ == 2 ? 3 : 6;
		assert(i < n);
		assert(j < n);
		assert(i >= 0);
		assert(j >= 0);
		const int index = n * i + j - i * (i + 1) / 2;
		assert(index < stifness_tensor_.size());

		stifness_tensor_(index) = val;
	}

	double HookeLinearElasticity::stifness_tensor(int i, int j) const
	{
		if(j < i)
		{
			int tmp=i;
			i = j;
			j = tmp;
		}

		assert(j>=i);
		const int n = size_ == 2 ? 3 : 6;
		assert(i < n);
		assert(j < n);
		assert(i >= 0);
		assert(j >= 0);
		const int index = n * i + j - i * (i + 1) / 2;
		assert(index < stifness_tensor_.size());

		return stifness_tensor_(index);
	}

	void HookeLinearElasticity::set_lambda_mu(const double lambda, const double mu)
	{
		if(size_ == 2)
		{
			set_stiffness_tensor(0, 0, 2*mu+lambda);
			set_stiffness_tensor(0, 1, lambda);
			set_stiffness_tensor(0, 2, 0);

			set_stiffness_tensor(1, 1, 2*mu+lambda);
			set_stiffness_tensor(1, 2, 0);

			set_stiffness_tensor(2, 2, mu);
		}
		else
		{
			set_stiffness_tensor(0, 0, 2*mu+lambda);
			set_stiffness_tensor(0, 1, lambda);
			set_stiffness_tensor(0, 2, lambda);
			set_stiffness_tensor(0, 3, 0);
			set_stiffness_tensor(0, 4, 0);
			set_stiffness_tensor(0, 5, 0);

			set_stiffness_tensor(1, 1, 2*mu+lambda);
			set_stiffness_tensor(1, 2, lambda);
			set_stiffness_tensor(1, 3, 0);
			set_stiffness_tensor(1, 4, 0);
			set_stiffness_tensor(1, 5, 0);

			set_stiffness_tensor(2, 2, 2*mu+lambda);
			set_stiffness_tensor(2, 3, 0);
			set_stiffness_tensor(2, 4, 0);
			set_stiffness_tensor(2, 5, 0);

			set_stiffness_tensor(3, 3, mu);
			set_stiffness_tensor(3, 4, 0);
			set_stiffness_tensor(3, 5, 0);

			set_stiffness_tensor(4, 4, mu);
			set_stiffness_tensor(4, 5, 0);

			set_stiffness_tensor(5, 5, mu);

		}
	}

	Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 9, 1>
	HookeLinearElasticity::assemble(const ElementAssemblyValues &vals, const int i, const int j, const Eigen::VectorXd &da) const
	{
		const Eigen::MatrixXd &gradi = vals.basis_values[i].grad;
		const Eigen::MatrixXd &gradj = vals.basis_values[j].grad;


		// (C : gradi) : gradj
		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 9, 1> res(size()*size());
		res.setZero();
		assert(gradi.cols() == size());
		assert(gradj.cols() == size());
		assert(gradi.rows() ==  vals.jac_it.size());

		for(long k = 0; k < gradi.rows(); ++k)
		{
			Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 9, 1> res_k(size()*size());

			if(size_ == 2)
			{
				const Eigen::Matrix2d eps_x_i = strain2d(gradi, vals.jac_it[k], k, 0);
				const Eigen::Matrix2d eps_y_i = strain2d(gradi, vals.jac_it[k], k, 1);

				const Eigen::Matrix2d eps_x_j = strain2d(gradj, vals.jac_it[k], k, 0);
				const Eigen::Matrix2d eps_y_j = strain2d(gradj, vals.jac_it[k], k, 1);

				std::array<double, 3> e_x, e_y;
				e_x[0] = eps_x_i(0,0);
				e_x[1] = eps_x_i(1,1);
				e_x[2] = 2*eps_x_i(0,1);

				e_y[0] = eps_y_i(0,0);
				e_y[1] = eps_y_i(1,1);
				e_y[2] = 2*eps_y_i(0,1);

				Eigen::Matrix2d sigma_x; sigma_x <<
				stress2d(e_x, 0), stress2d(e_x, 2),
				stress2d(e_x, 2), stress2d(e_x, 1);

				Eigen::Matrix2d sigma_y; sigma_y <<
				stress2d(e_y, 0), stress2d(e_y, 2),
				stress2d(e_y, 2), stress2d(e_y, 1);


				res_k(0) = (sigma_x*eps_x_j).trace();
				res_k(1) = (sigma_x*eps_y_j).trace();
				res_k(2) = (sigma_y*eps_x_j).trace();
				res_k(3) = (sigma_y*eps_y_j).trace();
			}
			else
			{
				const Eigen::Matrix3d eps_x_i = strain3d(gradi, vals.jac_it[k], k, 0);
				const Eigen::Matrix3d eps_y_i = strain3d(gradi, vals.jac_it[k], k, 1);
				const Eigen::Matrix3d eps_z_i = strain3d(gradi, vals.jac_it[k], k, 2);

				const Eigen::Matrix3d eps_x_j = strain3d(gradj, vals.jac_it[k], k, 0);
				const Eigen::Matrix3d eps_y_j = strain3d(gradj, vals.jac_it[k], k, 1);
				const Eigen::Matrix3d eps_z_j = strain3d(gradj, vals.jac_it[k], k, 2);


				std::array<double, 6> e_x, e_y, e_z;
				e_x[0] = eps_x_i(0,0);
				e_x[1] = eps_x_i(1,1);
				e_x[2] = eps_x_i(2,2);
				e_x[3] = 2*eps_x_i(1,2);
				e_x[4] = 2*eps_x_i(0,2);
				e_x[5] = 2*eps_x_i(0,1);

				e_y[0] = eps_y_i(0,0);
				e_y[1] = eps_y_i(1,1);
				e_y[2] = eps_y_i(2,2);
				e_y[3] = 2*eps_y_i(1,2);
				e_y[4] = 2*eps_y_i(0,2);
				e_y[5] = 2*eps_y_i(0,1);

				e_z[0] = eps_z_i(0,0);
				e_z[1] = eps_z_i(1,1);
				e_z[2] = eps_z_i(2,2);
				e_z[3] = 2*eps_z_i(1,2);
				e_z[4] = 2*eps_z_i(0,2);
				e_z[5] = 2*eps_z_i(0,1);


				Eigen::Matrix3d sigma_x; sigma_x <<
				stress3d(e_x, 0), stress3d(e_x, 5), stress3d(e_x, 4),
				stress3d(e_x, 5), stress3d(e_x, 1), stress3d(e_x, 3),
				stress3d(e_x, 4), stress3d(e_x, 3), stress3d(e_x, 2);

				Eigen::Matrix3d sigma_y; sigma_y <<
				stress3d(e_y, 0), stress3d(e_y, 5), stress3d(e_y, 4),
				stress3d(e_y, 5), stress3d(e_y, 1), stress3d(e_y, 3),
				stress3d(e_y, 4), stress3d(e_y, 3), stress3d(e_y, 2);

				Eigen::Matrix3d sigma_z; sigma_z <<
				stress3d(e_z, 0), stress3d(e_z, 5), stress3d(e_z, 4),
				stress3d(e_z, 5), stress3d(e_z, 1), stress3d(e_z, 3),
				stress3d(e_z, 4), stress3d(e_z, 3), stress3d(e_z, 2);


				res_k(0) = (sigma_x*eps_x_j).trace();
				res_k(1) = (sigma_x*eps_y_j).trace();
				res_k(2) = (sigma_x*eps_z_j).trace();

				res_k(3) = (sigma_y*eps_x_j).trace();
				res_k(4) = (sigma_y*eps_y_j).trace();
				res_k(5) = (sigma_y*eps_z_j).trace();

				res_k(6) = (sigma_z*eps_x_j).trace();
				res_k(7) = (sigma_z*eps_y_j).trace();
				res_k(8) = (sigma_z*eps_z_j).trace();
			}

			res += res_k * da(k);
		}

		// std::cout<<"res\n"<<res<<"\n"<<std::endl;

		return res;
	}

	void HookeLinearElasticity::compute_von_mises_stresses(const ElementBases &bs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, Eigen::MatrixXd &stresses) const
	{
		Eigen::MatrixXd displacement_grad(size(), size());

		assert(displacement.cols() == 1);

		stresses.resize(local_pts.rows(), 1);

		ElementAssemblyValues vals;
		vals.compute(-1, size() == 3, local_pts, bs, bs);


		for(long p = 0; p < local_pts.rows(); ++p)
		{
			displacement_grad.setZero();

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
						displacement_grad.row(d) += b.global()[ii].val * loc_val.grad.row(p) * displacement(b.global()[ii].index*size() + d);
					}
				}
			}

			displacement_grad = displacement_grad * vals.jac_it[p].transpose().inverse();

			Eigen::MatrixXd strain = (displacement_grad + displacement_grad.transpose())/2;
			Eigen::MatrixXd stress(size(), size());

			if(size() == 2)
			{
				std::array<double, 3> eps;
				eps[0] = strain(0,0);
				eps[1] = strain(1,1);
				eps[2] = 2*strain(0,1);


				stress <<
				stress2d(eps, 0), stress2d(eps, 2),
				stress2d(eps, 2), stress2d(eps, 1);
			}
			else
			{
				std::array<double, 6> eps;
				eps[0] = strain(0,0);
				eps[1] = strain(1,1);
				eps[2] = strain(2,2);
				eps[3] = 2*strain(1,2);
				eps[4] = 2*strain(0,2);
				eps[5] = 2*strain(0,1);

				stress <<
				stress3d(eps, 0), stress3d(eps, 5), stress3d(eps, 4),
				stress3d(eps, 5), stress3d(eps, 1), stress3d(eps, 3),
				stress3d(eps, 4), stress3d(eps, 3), stress3d(eps, 2);
			}

			stresses(p) = von_mises_stress_for_stress_tensor(stress);
		}
	}

}
