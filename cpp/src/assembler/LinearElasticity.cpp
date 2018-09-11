#include <polyfem/LinearElasticity.hpp>

#include <polyfem/Basis.hpp>
#include <polyfem/ElementAssemblyValues.hpp>
#include <polyfem/ElasticityUtils.hpp>

namespace polyfem
{
	void LinearElasticity::set_parameters(const json &params)
	{
		size() = params["size"];

		if(params.count("young")) {
			lambda() = convert_to_lambda(size_ == 3, params["young"], params["nu"]);
			mu() = convert_to_mu(params["young"], params["nu"]);
		} else if(params.count("E")) {
			lambda() = convert_to_lambda(size_ == 3, params["E"], params["nu"]);
			mu() = convert_to_mu(params["E"], params["nu"]);
		}
		else
		{
			lambda() = params["lambda"];
			mu() = params["mu"];
		}
	}

	Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 9, 1>
	LinearElasticity::assemble(const ElementAssemblyValues &vals, const int i, const int j, const QuadratureVector &da) const
	{
		// mu ((gradi' gradj) Id + gradi gradj') + lambda gradi *gradj';
		const Eigen::MatrixXd &gradi = vals.basis_values[i].grad_t_m;
		const Eigen::MatrixXd &gradj = vals.basis_values[j].grad_t_m;

		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 9, 1> res(size()*size());
		res.setZero();

		for(long k = 0; k < gradi.rows(); ++k)
		{
			Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 9, 1> res_k(size()*size());
//            res_k.setZero();
			const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> outer = gradi.row(k).transpose() * gradj.row(k);
            const double dot = gradi.row(k).dot(gradj.row(k));
			for(int ii = 0; ii < size(); ++ii)
			{
				for(int jj = 0; jj < size(); ++jj)
				{
					res_k(jj * size() + ii) = outer(ii * size() + jj)* mu_ + outer(jj * size() + ii)* lambda_;
					if(ii == jj) res_k(jj * size() + ii) += mu_ * dot;
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

	void LinearElasticity::compute_stress_tensor(const ElementBases &bs, const ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, Eigen::MatrixXd &stresses) const
	{
		assign_stress_tensor(bs, gbs, local_pts, displacement, size()*size(), stresses, [&](const Eigen::MatrixXd &stress)
		{
			Eigen::MatrixXd tmp = stress;
			return Eigen::Map<Eigen::MatrixXd>(tmp.data(), 1, size()*size());
		});
	}

	void LinearElasticity::compute_von_mises_stresses(const ElementBases &bs, const ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, Eigen::MatrixXd &stresses) const
	{
		assign_stress_tensor(bs, gbs, local_pts, displacement, 1, stresses, [&](const Eigen::MatrixXd &stress)
		{
			Eigen::Matrix<double, 1,1> res; res.setConstant(von_mises_stress_for_stress_tensor(stress));
			return res;
		});
	}

	void LinearElasticity::assign_stress_tensor(const ElementBases &bs, const ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, const int all_size, Eigen::MatrixXd &all, const std::function<Eigen::MatrixXd(const Eigen::MatrixXd &)> &fun) const
	{
		all.resize(local_pts.rows(), all_size);
		assert(displacement.cols() == 1);

		Eigen::MatrixXd displacement_grad(size(), size());
		// Eigen::MatrixXd xxx(1, size());
		// Eigen::MatrixXd xxx1(1, size());

		ElementAssemblyValues vals; //, valsdx;
		// Eigen::MatrixXd dx = local_pts;
		// int asdasd=2;
		// dx.col(asdasd).array() += 1e-5;
		// valsdx.compute(-1, size() == 3, dx, bs, gbs);
		vals.compute(-1, size() == 3, local_pts, bs, gbs);


		for(long p = 0; p < local_pts.rows(); ++p)
		{
			displacement_grad.setZero();
			// xxx.setZero();
			// xxx1.setZero();

			for(std::size_t j = 0; j < bs.bases.size(); ++j)
			{
				const Basis &b = bs.bases[j];
				const auto &loc_val = vals.basis_values[j];
				// const auto &loc_valdx = valsdx.basis_values[j];

				assert(bs.bases.size() == vals.basis_values.size());
				assert(loc_val.grad.rows() == local_pts.rows());
				assert(loc_val.grad.cols() == size());

				for(int d = 0; d < size(); ++d)
				{
					for(std::size_t ii = 0; ii < b.global().size(); ++ii)
					{
						displacement_grad.row(d) += b.global()[ii].val * loc_val.grad.row(p) * displacement(b.global()[ii].index*size() + d);
						// xxx(d) 					 += b.global()[ii].val * loc_val.val(p)      * displacement(b.global()[ii].index*size() + d);
						// xxx1(d) += b.global()[ii].val * loc_valdx.val(p) * displacement(b.global()[ii].index*size() + d);
						// assert(!std::isnan(xxx(d)));
					}
				}
			}
			// const Eigen::MatrixXd fd = (xxx1 - xxx)/1e-5;
			// std::cout<<"err "<<(displacement_grad.col(asdasd).transpose()-fd)<<std::endl<<std::endl;
			// u(x) = \sum_i\sum_d phi_i(x) * u_{i,d}

			// grad phi_i(x) = grad (\hat phi_i(G-1(x)))) = grad \hat phi_i(G-1(x)) * (grad G-1)^T = grad \hat phi_i(\hat x) * (grad G-1)^T
			// grad u(x) = \sum_i\sum_d grad phi_i(x) * u_{i,d} = (\sum_i\sum_d grad \hat phi_i(\hat x) * u_{i,d}) * (grad G-1)^T
			displacement_grad = (displacement_grad * vals.jac_it[p]).eval();

			// std::cout<<fd<<std::endl<<std::endl;
			// std::cout<<(valsdx.val.row(p)- vals.val.row(p))<<std::endl<<std::endl;
			// std::cout<<displacement_grad<<std::endl<<std::endl<<std::endl;

			// const double errp = (displacement_grad.col(0).transpose()-fd*20).array().abs().maxCoeff();
			// const double errm = (displacement_grad.col(0).transpose()+fd*20).array().abs().maxCoeff();


			const Eigen::MatrixXd strain = (displacement_grad + displacement_grad.transpose())/2;
			const Eigen::MatrixXd stress = 2 * mu_ * strain + lambda_ * strain.trace() * Eigen::MatrixXd::Identity(size(), size());

			all.row(p) = fun(stress);

			// if(all_size > size())
			// {
			// 	for(int a = 0; a < all_size; ++a)
			// 		all(p, a) = displacement_grad(a);

			// 	all(p,  0) = 0;
			// 	for(int a = 0; a < all_size; ++a)
			// 		all(p, 0) += displacement_grad(a);
			// 	// all(p, 0) = errp;
			// 	// all(p, 1) = errm;
			// // 	std::cout<<vals.jac_it[p]<<std::endl<<std::endl;
			// // 	for(int a = 0; a < size(); ++a)
			// // 		all(p, a) = xxx(a);

			// // 	// std::cout<<all<<std::endl<<std::endl;
			// // 	// std::cout<<xxx<<std::endl<<std::endl;
			// // 	// exit(0);
			// }
		}

					// exit(0);
	}
}
