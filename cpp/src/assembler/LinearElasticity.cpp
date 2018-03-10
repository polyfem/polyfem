#include "LinearElasticity.hpp"

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
	}

	void LinearElasticity::set_parameters(const json &params)
	{
		size() = params["size"];

		lambda() = params["lambda"];
		mu() = params["mu"];
	}

	Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 9, 1>
	LinearElasticity::assemble(const ElementAssemblyValues &vals, const int i, const int j, const Eigen::VectorXd &da) const
	{
		// mu ((gradi' gradj) Id + gradi gradj') + lambda gradi *gradj';
		const Eigen::MatrixXd &gradi = vals.basis_values[i].grad_t_m;
		const Eigen::MatrixXd &gradj = vals.basis_values[j].grad_t_m;

		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 9, 1> res(size()*size());
		res.setZero();

		const auto &dot = (gradi.array() * gradj.array()).rowwise().sum();

		for(long k = 0; k < gradi.rows(); ++k)
		{
			Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 9, 1> res_k(size()*size());
			res_k.setZero();
			const Eigen::MatrixXd outer = gradi.row(k).transpose() * gradj.row(k);
			for(int ii = 0; ii < size(); ++ii)
			{
				for(int jj = 0; jj < size(); ++jj)
				{
					res_k(ii * size() + jj) = outer(ii * size() + jj)* mu_ + outer(jj * size() + ii)* lambda_;
					if(ii == jj) res_k(ii * size() + jj) += mu_ * dot(k);
				}
			}
			res += res_k * da(k);
		}

		return res;
	}

	Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1>
	LinearElasticity::compute_rhs(const AutodiffHessianPt &pt) const
	{
	assert(pt.size() == size());
		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> res(size());


		if(size() == 2)
		{
			res(0) = (lambda_+2*mu_)*pt(0).getHessian()(0,0)  +(lambda_+mu_)  *(pt(1).getHessian()(1,0))+mu_*(pt(0).getHessian()(1,1));
			res(1) = (lambda_+mu_)  *(pt(0).getHessian()(1,0))+(lambda_+2*mu_)*(pt(1).getHessian()(1,1))+mu_*(pt(1).getHessian()(0,0));
		}
		else if(size() == 3)
		{
			res(0) = (lambda_+2*mu_)*pt(0).getHessian()(0,0)+(lambda_+mu_)*pt(1).getHessian()(1,0)+(lambda_+mu_)*pt(2).getHessian()(2,0)+mu_*(pt(0).getHessian()(1,1)+pt(0).getHessian()(2,2));
			res(1) = (lambda_+mu_)*pt(0).getHessian()(1,0)+(lambda_+2*mu_)*pt(1).getHessian()(1,1)+(lambda_+mu_)*pt(2).getHessian()(2,1)+mu_*(pt(1).getHessian()(0,0)+pt(1).getHessian()(2,2));
			res(2) = (lambda_+mu_)*pt(0).getHessian()(2,0)+(lambda_+mu_)*pt(1).getHessian()(2,1)+(lambda_+2*mu_)*pt(2).getHessian()(2,2)+mu_*(pt(2).getHessian()(0,0)+pt(2).getHessian()(1,1));
		}
		else
			assert(false);

		return res;
	}

	void LinearElasticity::compute_von_mises_stresses(const ElementBases &bs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, Eigen::MatrixXd &stresses) const
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

			Eigen::MatrixXd strain = (displacement_grad + displacement_grad.transpose())/2;
			const Eigen::MatrixXd stress =
			2 * mu_ * strain + lambda_ * strain.trace() * Eigen::MatrixXd::Identity(size(), size());

			stresses(p) = von_mises_stress_for_stress_tensor(stress);
		}
	}

}
