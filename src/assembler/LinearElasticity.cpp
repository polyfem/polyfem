#include <polyfem/LinearElasticity.hpp>

#include <polyfem/Basis.hpp>
#include <polyfem/ElementAssemblyValues.hpp>

#include <polyfem/auto_elasticity_rhs.hpp>

namespace polyfem
{
	void LinearElasticity::init_multimaterial(Eigen::MatrixXd &Es, Eigen::MatrixXd &nus)
	{
		params_.init_multimaterial(Es, nus);
	}


	void LinearElasticity::set_parameters(const json &params)
	{
		size() = params["size"];

		params_.init(params);

		// if(params.count("young")) {
		// 	lambda() = convert_to_lambda(size_ == 3, params["young"], params["nu"]);
		// 	mu() = convert_to_mu(params["young"], params["nu"]);
		// } else if(params.count("E")) {
		// 	lambda() = convert_to_lambda(size_ == 3, params["E"], params["nu"]);
		// 	mu() = convert_to_mu(params["E"], params["nu"]);
		// }
		// else
		// {
		// 	lambda() = params["lambda"];
		// 	mu() = params["mu"];
		// }

		// std::cout<<mu_<<std::endl;
		// std::cout<<lambda_<<std::endl;
	}

	Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 9, 1>
	LinearElasticity::assemble(const ElementAssemblyValues &vals, const int i, const int j, const QuadratureVector &da) const
	{
		// mu ((gradi' gradj) Id + ((gradi gradj')') + lambda gradi *gradj';
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
					double lambda, mu;
					params_.lambda_mu(vals.val(k, 0), vals.val(k, 1), size_ == 2 ? 0. : vals.val(k, 2), vals.element_id, lambda, mu);

					res_k(jj * size() + ii) = outer(ii * size() + jj)* mu + outer(jj * size() + ii) * lambda;
					if(ii == jj) res_k(jj * size() + ii) += mu * dot;
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

		double lambda, mu;
		params_.lambda_mu(pt(0).getValue(), pt(1).getValue(), size_ == 2 ? 0. : pt(2).getValue(), 0, lambda, mu);

		if(size() == 2)
			autogen::linear_elasticity_2d_function(pt, lambda, mu, res);
		else if(size() == 3)
			autogen::linear_elasticity_3d_function(pt, lambda, mu, res);
		else
			assert(false);

		return res;
	}

	void LinearElasticity::compute_stress_tensor(const int el_id, const ElementBases &bs, const ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, Eigen::MatrixXd &stresses) const
	{
		assign_stress_tensor(el_id, bs, gbs, local_pts, displacement, size()*size(), stresses, [&](const Eigen::MatrixXd &stress)
		{
			Eigen::MatrixXd tmp = stress;
			auto a = Eigen::Map<Eigen::MatrixXd>(tmp.data(), 1, size()*size());
			return Eigen::MatrixXd(a);
		});
	}

	void LinearElasticity::compute_von_mises_stresses(const int el_id, const ElementBases &bs, const ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, Eigen::MatrixXd &stresses) const
	{
		assign_stress_tensor(el_id, bs, gbs, local_pts, displacement, 1, stresses, [&](const Eigen::MatrixXd &stress)
		{
			Eigen::Matrix<double, 1,1> res; res.setConstant(von_mises_stress_for_stress_tensor(stress));
			return res;
		});
	}

	void LinearElasticity::assign_stress_tensor(const int el_id, const ElementBases &bs, const ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, const int all_size, Eigen::MatrixXd &all, const std::function<Eigen::MatrixXd(const Eigen::MatrixXd &)> &fun) const
	{
		all.resize(local_pts.rows(), all_size);
		assert(displacement.cols() == 1);

		Eigen::MatrixXd displacement_grad(size(), size());

		ElementAssemblyValues vals;
		vals.compute(el_id, size() == 3, local_pts, bs, gbs);

		for(long p = 0; p < local_pts.rows(); ++p)
		{
			// displacement_grad.setZero();

			// for(std::size_t j = 0; j < bs.bases.size(); ++j)
			// {
			// 	const Basis &b = bs.bases[j];
			// 	const auto &loc_val = vals.basis_values[j];

			// 	assert(bs.bases.size() == vals.basis_values.size());
			// 	assert(loc_val.grad.rows() == local_pts.rows());
			// 	assert(loc_val.grad.cols() == size());

			// 	for(int d = 0; d < size(); ++d)
			// 	{
			// 		for(std::size_t ii = 0; ii < b.global().size(); ++ii)
			// 		{
			// 			displacement_grad.row(d) += b.global()[ii].val * loc_val.grad.row(p) * displacement(b.global()[ii].index*size() + d);
			// 		}
			// 	}
			// }

			// displacement_grad = (displacement_grad * vals.jac_it[p]).eval();

			compute_diplacement_grad(size(), bs, vals, local_pts, p, displacement, displacement_grad);

			double lambda, mu;
			params_.lambda_mu(vals.val(p, 0), vals.val(p, 1), size_ == 2 ? 0. : vals.val(p, 2), vals.element_id, lambda, mu);

			const Eigen::MatrixXd strain = (displacement_grad + displacement_grad.transpose())/2;
			const Eigen::MatrixXd stress = 2 * mu * strain + lambda * strain.trace() * Eigen::MatrixXd::Identity(size(), size());

			all.row(p) = fun(stress);
		}
	}
}
