#include "SaintVenantElasticity.hpp"

#include "Basis.hpp"
#include "ElementAssemblyValues.hpp"

#include <unsupported/Eigen/AutoDiff>

namespace poly_fem
{

	namespace
	{
		typedef Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> Derivative_type;
		typedef Eigen::AutoDiffScalar<Derivative_type> Scalar_type;
		typedef Eigen::Matrix<Scalar_type, Eigen::Dynamic, Eigen::Dynamic, 0, Eigen::Dynamic, 3> Matrix_t;

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

		template<typename T, int dim>
		Eigen::Matrix<T, dim, dim> strain(const Eigen::MatrixXd &grad, const Eigen::MatrixXd &jac_it, Eigen::Matrix<T, Eigen::Dynamic, 1, 0, 3, 1> disp, int k, int coo)
		{
			Eigen::Matrix<double, dim, dim> tmp;
			tmp.setZero();
			tmp.row(coo) = grad.row(k);
			tmp = tmp*jac_it;

			Eigen::Matrix<T, dim, dim> jac;
			for(int i = 0; i < dim; ++i)
			{
				for(int j = 0; j < dim; ++j)
					jac(i, j) = tmp(i, j) * disp(i);
			}

			return (jac*jac.transpose() + jac + jac.transpose())*0.5;
		}
	}



	SaintVenantElasticity::SaintVenantElasticity()
	{
		set_size(size_);
	}

	void SaintVenantElasticity::set_size(const int size)
	{
		if(size == 2)
			stifness_tensor_.resize(6, 1);
		else
			stifness_tensor_.resize(21, 1);

		size_ = size;
	}

	template <typename T, unsigned long N>
	T SaintVenantElasticity::stress(const std::array<T, N> &strain, const int j) const
	{
		T res = 0;

		for(unsigned long k = 0; k < N; ++k)
			res += stifness_tensor(j, k)*strain[k];

		return res;
	}

	void SaintVenantElasticity::set_stiffness_tensor(int i, int j, const double val)
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

	double SaintVenantElasticity::stifness_tensor(int i, int j) const
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

	void SaintVenantElasticity::set_lambda_mu(const double lambda, const double mu)
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

	Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1>
	SaintVenantElasticity::assemble(const ElementAssemblyValues &vals, const AssemblyValues &values_j, const Eigen::MatrixXd &displacement, const Eigen::VectorXd &da) const
	{
		Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, Eigen::Dynamic, 3> local_disp(vals.basis_values.size(), size());
		local_disp.setZero();
		for(size_t i = 0; i < vals.basis_values.size(); ++i){
			const auto &bs = vals.basis_values[i];
			for(size_t ii = 0; ii < bs.global.size(); ++ii){
				for(int d = 0; d < size(); ++d){
					local_disp(i,d) += bs.global[ii].val * displacement(bs.global[ii].index*size() + d);
				}
			}
		}

		return assemble_aux(vals, values_j, da, local_disp);
	}

	Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 9, 1>
	SaintVenantElasticity::assemble_grad(const ElementAssemblyValues &vals, const AssemblyValues &values_j, const Eigen::MatrixXd &displacement, const Eigen::VectorXd &da) const
	{
		Matrix_t local_disp(vals.basis_values.size(), size());
		local_disp.setZero();
		for(size_t i = 0; i < vals.basis_values.size(); ++i){
			const auto &bs = vals.basis_values[i];
			for(size_t ii = 0; ii < bs.global.size(); ++ii){
				for(int d = 0; d < size(); ++d){
					local_disp(i,d) += bs.global[ii].val * displacement(bs.global[ii].index*size() + d);
				}
			}
		}

		//set unit vectors for the derivative directions (partial derivatives of the input vector)
		for(int d = 0; d < size(); ++d)
		{
			local_disp(d).derivatives().resize(size());
			local_disp(d).derivatives().setZero();
			local_disp(d).derivatives()(d)=1;
		}

		const auto val = assemble_aux(vals, values_j, da, local_disp);

		// std::cout << y.derivatives() << std::endl;
		// std::cout << y.value() << std::endl;

		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 9, 1> res(size()*size());

		for(int i = 0; i < size(); ++i)
		{
			const auto jac = val(i).derivatives();

			for(int j = 0; j < size(); ++j)
			{
				res(j*size() + i) = jac(j);
			}
		}

		return res;
	}

	template <typename T>
	Eigen::Matrix<T, Eigen::Dynamic, 1, 0, 3, 1>
	SaintVenantElasticity::assemble_aux(const ElementAssemblyValues &vals, const AssemblyValues &values_j,
		const Eigen::VectorXd &da, const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, 0, Eigen::Dynamic, 3> &local_disp) const
	{
		const Eigen::MatrixXd &gradj = values_j.grad;

		// sum (C : gradi) : gradj
		Eigen::Matrix<T, Eigen::Dynamic, 1, 0, 3, 1> res(size());
		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> ones(size()); ones.setOnes();

		res.setZero();

		assert(gradj.cols() == size());
		assert(size_t(gradj.rows()) ==  vals.jac_it.size());

		for(size_t i = 0; i < vals.basis_values.size(); ++i)
		{
			const Eigen::MatrixXd &gradi = vals.basis_values[i].grad;
			assert(gradi.cols() == size());
			assert(size_t(gradi.rows()) ==  vals.jac_it.size());

			for(long k = 0; k < gradi.rows(); ++k)
			{
				Eigen::Matrix<T, Eigen::Dynamic, 1, 0, 3, 1> res_k(size());

				if(size_ == 2)
				{
					const auto eps_x_i = strain<T, 2>(gradi, vals.jac_it[k], local_disp.row(i), k, 0);
					const auto eps_y_i = strain<T, 2>(gradi, vals.jac_it[k], local_disp.row(i), k, 1);

					const auto eps_x_j = strain<double, 2>(gradj, vals.jac_it[k], ones, k, 0);
					const auto eps_y_j = strain<double, 2>(gradj, vals.jac_it[k], ones, k, 1);

					std::array<T, 3> e_x, e_y;
					e_x[0] = eps_x_i(0,0);
					e_x[1] = eps_x_i(1,1);
					e_x[2] = 2*eps_x_i(0,1);

					e_y[0] = eps_y_i(0,0);
					e_y[1] = eps_y_i(1,1);
					e_y[2] = 2*eps_y_i(0,1);

					Eigen::Matrix<T, 2, 2> sigma_x; sigma_x <<
					stress(e_x, 0), stress(e_x, 2),
					stress(e_x, 2), stress(e_x, 1);

					Eigen::Matrix<T, 2, 2> sigma_y; sigma_y <<
					stress(e_y, 0), stress(e_y, 2),
					stress(e_y, 2), stress(e_y, 1);


					res_k(0) = (sigma_x*eps_x_j).trace();
					res_k(1) = (sigma_x*eps_y_j).trace();
					res_k(2) = (sigma_y*eps_x_j).trace();
					res_k(3) = (sigma_y*eps_y_j).trace();
				}
				else
				{
					const auto eps_x_i = strain<T, 3>(gradi, vals.jac_it[k],local_disp.row(i), k, 0);
					const auto eps_y_i = strain<T, 3>(gradi, vals.jac_it[k],local_disp.row(i), k, 1);
					const auto eps_z_i = strain<T, 3>(gradi, vals.jac_it[k],local_disp.row(i), k, 2);

					const auto eps_x_j = strain<double, 3>(gradj, vals.jac_it[k], ones, k, 0);
					const auto eps_y_j = strain<double, 3>(gradj, vals.jac_it[k], ones, k, 1);
					const auto eps_z_j = strain<double, 3>(gradj, vals.jac_it[k], ones, k, 2);


					std::array<T, 6> e_x, e_y, e_z;
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


					Eigen::Matrix<T, 3, 3> sigma_x; sigma_x <<
					stress(e_x, 0), stress(e_x, 5), stress(e_x, 4),
					stress(e_x, 5), stress(e_x, 1), stress(e_x, 3),
					stress(e_x, 4), stress(e_x, 3), stress(e_x, 2);

					Eigen::Matrix<T, 3, 3> sigma_y; sigma_y <<
					stress(e_y, 0), stress(e_y, 5), stress(e_y, 4),
					stress(e_y, 5), stress(e_y, 1), stress(e_y, 3),
					stress(e_y, 4), stress(e_y, 3), stress(e_y, 2);

					Eigen::Matrix<T, 3, 3> sigma_z; sigma_z <<
					stress(e_z, 0), stress(e_z, 5), stress(e_z, 4),
					stress(e_z, 5), stress(e_z, 1), stress(e_z, 3),
					stress(e_z, 4), stress(e_z, 3), stress(e_z, 2);


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
		}

		// std::cout<<"res\n"<<res<<"\n"<<std::endl;

		return res;
	}

	void SaintVenantElasticity::compute_von_mises_stresses(const ElementBases &bs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, Eigen::MatrixXd &stresses) const
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

			Eigen::MatrixXd strain = (displacement_grad.transpose()*displacement_grad + displacement_grad + displacement_grad.transpose())/2;
			Eigen::MatrixXd stress_tensor(size(), size());

			if(size() == 2)
			{
				std::array<double, 3> eps;
				eps[0] = strain(0,0);
				eps[1] = strain(1,1);
				eps[2] = 2*strain(0,1);


				stress_tensor <<
				stress(eps, 0), stress(eps, 2),
				stress(eps, 2), stress(eps, 1);
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

				stress_tensor <<
				stress(eps, 0), stress(eps, 5), stress(eps, 4),
				stress(eps, 5), stress(eps, 1), stress(eps, 3),
				stress(eps, 4), stress(eps, 3), stress(eps, 2);
			}

			stresses(p) = von_mises_stress_for_stress_tensor(stress_tensor);
		}
	}




	//explicit instantiation
	template double SaintVenantElasticity::stress(const std::array<double, 3> &strain, const int j) const;
	template double SaintVenantElasticity::stress(const std::array<double, 6> &strain, const int j) const;

	template Scalar_type SaintVenantElasticity::stress(const std::array<Scalar_type, 3> &strain, const int j) const;
	template Scalar_type SaintVenantElasticity::stress(const std::array<Scalar_type, 6> &strain, const int j) const;

	template Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> SaintVenantElasticity::assemble_aux(const ElementAssemblyValues &vals, const AssemblyValues &values_j, const Eigen::VectorXd &da, const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, Eigen::Dynamic, 3> &local_disp) const;
	template Eigen::Matrix<Scalar_type, Eigen::Dynamic, 1, 0, 3, 1> SaintVenantElasticity::assemble_aux(const ElementAssemblyValues &vals, const AssemblyValues &values_j, const Eigen::VectorXd &da, const Eigen::Matrix<Scalar_type, Eigen::Dynamic, Eigen::Dynamic, 0, Eigen::Dynamic, 3> &local_disp) const;
}
