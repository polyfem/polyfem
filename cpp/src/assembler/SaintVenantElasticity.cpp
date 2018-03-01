#include "SaintVenantElasticity.hpp"

#include "Basis.hpp"
#include "ElementAssemblyValues.hpp"


namespace poly_fem
{

	namespace
	{
		template<class Matrix>
		Matrix strain_from_disp_grad(const Matrix &disp_grad)
		{
			// Matrix mat =  (disp_grad + disp_grad.transpose());
			Matrix mat = (disp_grad.transpose()*disp_grad + disp_grad + disp_grad.transpose());

			for(int i = 0; i < mat.size(); ++i)
				mat(i) *= 0.5;

			return mat;
		}

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

		template<int dim>
		Eigen::Matrix<double, dim, dim> strain(const Eigen::MatrixXd &grad, const Eigen::MatrixXd &jac_it, int k, int coo)
		{
			Eigen::Matrix<double, dim, dim> jac;
			jac.setZero();
			jac.row(coo) = grad.row(k);
			jac = jac*jac_it;

			return strain_from_disp_grad(jac);
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
		T res = stifness_tensor(j, 0)*strain[0];

		for(unsigned long k = 1; k < N; ++k)
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

	Eigen::VectorXd
	SaintVenantElasticity::assemble(const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const Eigen::VectorXd &da) const
	{
		auto auto_diff_energy = compute_energy_aux<AutoDiffScalar>(vals, displacement, da);
		return auto_diff_energy.getGradient();
	}

	Eigen::MatrixXd
	SaintVenantElasticity::assemble_grad(const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const Eigen::VectorXd &da) const
	{
		auto auto_diff_energy = compute_energy_aux<AutoDiffScalar>(vals, displacement, da);
		return auto_diff_energy.getHessian();
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

			displacement_grad = displacement_grad * vals.jac_it[p];

			Eigen::MatrixXd strain = strain_from_disp_grad(displacement_grad);
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

	double SaintVenantElasticity::compute_energy(const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const Eigen::VectorXd &da) const
	{
		return compute_energy_aux<double>(vals, displacement, da);
		// return auto_diff_energy.getValue();
	}

	template<typename T>
	T SaintVenantElasticity::compute_energy_aux(const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const Eigen::VectorXd &da) const
	{
		typedef Eigen::Matrix<T, Eigen::Dynamic, 1> 						AutoDiffVect;
		typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> 	AutoDiffGradMat;

		assert(displacement.cols() == 1);

		const int n_pts = da.size();

		Eigen::Matrix<double, Eigen::Dynamic, 1> local_dispv(vals.basis_values.size() * size(), 1);
		local_dispv.setZero();
		for(size_t i = 0; i < vals.basis_values.size(); ++i){
			const auto &bs = vals.basis_values[i];
			for(size_t ii = 0; ii < bs.global.size(); ++ii){
				for(int d = 0; d < size(); ++d){
					local_dispv(i*size() + d) += bs.global[ii].val * displacement(bs.global[ii].index*size() + d);
				}
			}
		}

		DiffScalarBase::setVariableCount(local_dispv.rows());
		AutoDiffVect local_disp(local_dispv.rows(), 1);
		T energy = T(0.0);

		AutoDiffAllocator<T> allocate_auto_diff_scalar;

		for(long i = 0; i < local_dispv.rows(); ++i){
			local_disp(i) = allocate_auto_diff_scalar(i, local_dispv(i));
		}

		AutoDiffGradMat displacement_grad(size(), size());

		for(long p = 0; p < n_pts; ++p)
		{
			bool is_disp_grad_set = false;

			for(size_t i = 0; i < vals.basis_values.size(); ++i)
			{
				const auto &bs = vals.basis_values[i];
				const Eigen::MatrixXd grad = bs.grad*vals.jac_it[p];
				assert(grad.cols() == size());
				assert(size_t(grad.rows()) ==  vals.jac_it.size());

				for(int d = 0; d < size(); ++d)
				{
					for(int c = 0; c < size(); ++c)
					{
						if(is_disp_grad_set)
							displacement_grad(d, c) += grad(p, c) * local_disp(i*size() + d);
						else
							displacement_grad(d, c) = grad(p, c) * local_disp(i*size() + d);
					}
				}

				is_disp_grad_set = true;
			}

			// displacement_grad = displacement_grad * vals.jac_it[p];

			AutoDiffGradMat strain = strain_from_disp_grad(displacement_grad);
			AutoDiffGradMat stress_tensor(size(), size());

			if(size() == 2)
			{
				std::array<T, 3> eps;
				eps[0] = strain(0,0);
				eps[1] = strain(1,1);
				eps[2] = 2*strain(0,1);


				stress_tensor <<
				stress(eps, 0), stress(eps, 2),
				stress(eps, 2), stress(eps, 1);
			}
			else
			{
				std::array<T, 6> eps;
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

			energy += (stress_tensor * strain).trace() * da(p);
		}

		return energy * 0.5;
	}


	//explicit instantiation
	template double SaintVenantElasticity::stress(const std::array<double, 3> &strain, const int j) const;
	template double SaintVenantElasticity::stress(const std::array<double, 6> &strain, const int j) const;

	template AutoDiffScalar SaintVenantElasticity::stress(const std::array<AutoDiffScalar, 3> &strain, const int j) const;
	template AutoDiffScalar SaintVenantElasticity::stress(const std::array<AutoDiffScalar, 6> &strain, const int j) const;

	template double SaintVenantElasticity::compute_energy_aux(const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const Eigen::VectorXd &da) const;
	template AutoDiffScalar SaintVenantElasticity::compute_energy_aux(const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const Eigen::VectorXd &da) const;
}
