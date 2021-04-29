#include <polyfem/LinearElasticity.hpp>

#include <polyfem/Basis.hpp>
#include <polyfem/ElementAssemblyValues.hpp>

#include <polyfem/auto_elasticity_rhs.hpp>

namespace polyfem
{
	void LinearElasticity::init_multimaterial(const bool is_volume, const Eigen::MatrixXd &Es, const Eigen::MatrixXd &nus)
	{
		params_.init_multimaterial(is_volume, Es, nus);
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

		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 9, 1> res(size() * size());
		res.setZero();

		for (long k = 0; k < gradi.rows(); ++k)
		{
			Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 9, 1> res_k(size() * size());
			//            res_k.setZero();
			const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> outer = gradi.row(k).transpose() * gradj.row(k);
			const double dot = gradi.row(k).dot(gradj.row(k));
			for (int ii = 0; ii < size(); ++ii)
			{
				for (int jj = 0; jj < size(); ++jj)
				{
					double lambda, mu;
					params_.lambda_mu(vals.val(k, 0), vals.val(k, 1), size_ == 2 ? 0. : vals.val(k, 2), vals.element_id, lambda, mu);

					res_k(jj * size() + ii) = outer(ii * size() + jj) * mu + outer(jj * size() + ii) * lambda;
					if (ii == jj)
						res_k(jj * size() + ii) += mu * dot;
				}
			}
			res += res_k * da(k);
		}

		return res;
	}

	double LinearElasticity::compute_energy(const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) const
	{
		return compute_energy_aux<double>(vals, displacement, da);
	}

	Eigen::VectorXd LinearElasticity::assemble_grad(const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) const
	{
		const int n_bases = vals.basis_values.size();
		return polyfem::gradient_from_energy(
			size(), n_bases, vals, displacement, da,
			[&](const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) { return compute_energy_aux<DScalar1<double, Eigen::Matrix<double, 6, 1>>>(vals, displacement, da); },
			[&](const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) { return compute_energy_aux<DScalar1<double, Eigen::Matrix<double, 8, 1>>>(vals, displacement, da); },
			[&](const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) { return compute_energy_aux<DScalar1<double, Eigen::Matrix<double, 12, 1>>>(vals, displacement, da); },
			[&](const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) { return compute_energy_aux<DScalar1<double, Eigen::Matrix<double, 18, 1>>>(vals, displacement, da); },
			[&](const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) { return compute_energy_aux<DScalar1<double, Eigen::Matrix<double, 24, 1>>>(vals, displacement, da); },
			[&](const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) { return compute_energy_aux<DScalar1<double, Eigen::Matrix<double, 30, 1>>>(vals, displacement, da); },
			[&](const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) { return compute_energy_aux<DScalar1<double, Eigen::Matrix<double, 60, 1>>>(vals, displacement, da); },
			[&](const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) { return compute_energy_aux<DScalar1<double, Eigen::Matrix<double, 81, 1>>>(vals, displacement, da); },
			[&](const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) { return compute_energy_aux<DScalar1<double, Eigen::Matrix<double, Eigen::Dynamic, 1, 0, SMALL_N, 1>>>(vals, displacement, da); },
			[&](const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) { return compute_energy_aux<DScalar1<double, Eigen::Matrix<double, Eigen::Dynamic, 1, 0, BIG_N, 1>>>(vals, displacement, da); },
			[&](const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) { return compute_energy_aux<DScalar1<double, Eigen::VectorXd>>(vals, displacement, da); });
	}

	Eigen::MatrixXd LinearElasticity::assemble_hessian(const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) const
	{
		const int n_bases = vals.basis_values.size();
		return polyfem::hessian_from_energy(
			size(), n_bases, vals, displacement, da,
			[&](const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) { return compute_energy_aux<DScalar2<double, Eigen::Matrix<double, 6, 1>, Eigen::Matrix<double, 6, 6>>>(vals, displacement, da); },
			[&](const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) { return compute_energy_aux<DScalar2<double, Eigen::Matrix<double, 8, 1>, Eigen::Matrix<double, 8, 8>>>(vals, displacement, da); },
			[&](const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) { return compute_energy_aux<DScalar2<double, Eigen::Matrix<double, 12, 1>, Eigen::Matrix<double, 12, 12>>>(vals, displacement, da); },
			[&](const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) { return compute_energy_aux<DScalar2<double, Eigen::Matrix<double, 18, 1>, Eigen::Matrix<double, 18, 18>>>(vals, displacement, da); },
			[&](const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) { return compute_energy_aux<DScalar2<double, Eigen::Matrix<double, 24, 1>, Eigen::Matrix<double, 24, 24>>>(vals, displacement, da); },
			[&](const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) { return compute_energy_aux<DScalar2<double, Eigen::Matrix<double, 30, 1>, Eigen::Matrix<double, 30, 30>>>(vals, displacement, da); },
			[&](const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) { return compute_energy_aux<DScalar2<double, Eigen::Matrix<double, 60, 1>, Eigen::Matrix<double, 60, 60>>>(vals, displacement, da); },
			[&](const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) { return compute_energy_aux<DScalar2<double, Eigen::Matrix<double, 81, 1>, Eigen::Matrix<double, 81, 81>>>(vals, displacement, da); },
			[&](const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) { return compute_energy_aux<DScalar2<double, Eigen::Matrix<double, Eigen::Dynamic, 1, 0, SMALL_N, 1>, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, SMALL_N, SMALL_N>>>(vals, displacement, da); },
			[&](const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) { return compute_energy_aux<DScalar2<double, Eigen::VectorXd, Eigen::MatrixXd>>(vals, displacement, da); });
	}

	//Compute \int mu eps : eps + lambda/2 tr(eps)^2 = \int mu tr(eps^2) + lambda/2 tr(eps)^2
	template <typename T>
	T LinearElasticity::compute_energy_aux(const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) const
	{
		typedef Eigen::Matrix<T, Eigen::Dynamic, 1> AutoDiffVect;
		typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> AutoDiffGradMat;

		assert(displacement.cols() == 1);

		const int n_pts = da.size();

		Eigen::Matrix<double, Eigen::Dynamic, 1> local_dispv(vals.basis_values.size() * size(), 1);
		local_dispv.setZero();
		for (size_t i = 0; i < vals.basis_values.size(); ++i)
		{
			const auto &bs = vals.basis_values[i];
			for (size_t ii = 0; ii < bs.global.size(); ++ii)
			{
				for (int d = 0; d < size(); ++d)
				{
					local_dispv(i * size() + d) += bs.global[ii].val * displacement(bs.global[ii].index * size() + d);
				}
			}
		}

		DiffScalarBase::setVariableCount(local_dispv.rows());
		AutoDiffVect local_disp(local_dispv.rows(), 1);
		T energy = T(0.0);

		const AutoDiffAllocator<T> allocate_auto_diff_scalar;

		for (long i = 0; i < local_dispv.rows(); ++i)
		{
			local_disp(i) = allocate_auto_diff_scalar(i, local_dispv(i));
		}

		AutoDiffGradMat def_grad(size(), size());

		for (long p = 0; p < n_pts; ++p)
		{
			for (long k = 0; k < def_grad.size(); ++k)
				def_grad(k) = T(0);

			for (size_t i = 0; i < vals.basis_values.size(); ++i)
			{
				const auto &bs = vals.basis_values[i];
				const Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> grad = bs.grad.row(p);
				assert(grad.size() == size());

				for (int d = 0; d < size(); ++d)
				{
					for (int c = 0; c < size(); ++c)
					{
						def_grad(d, c) += grad(c) * local_disp(i * size() + d);
					}
				}
			}

			AutoDiffGradMat jac_it(size(), size());
			for (long k = 0; k < jac_it.size(); ++k)
				jac_it(k) = T(vals.jac_it[p](k));
			def_grad = def_grad * jac_it;

			const AutoDiffGradMat strain = (def_grad + def_grad.transpose()) / T(2);

			double lambda, mu;
			params_.lambda_mu(vals.val(p, 0), vals.val(p, 1), size_ == 2 ? 0. : vals.val(p, 2), vals.element_id, lambda, mu);

			const T val = mu * (strain.transpose() * strain).trace() + lambda / 2 * strain.trace() * strain.trace();

			energy += val * da(p);
		}
		return energy;
	}

	Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1>
	LinearElasticity::compute_rhs(const AutodiffHessianPt &pt) const
	{
		assert(pt.size() == size());
		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> res(size());

		double lambda, mu;
		params_.lambda_mu(pt(0).getValue(), pt(1).getValue(), size_ == 2 ? 0. : pt(2).getValue(), 0, lambda, mu);

		if (size() == 2)
			autogen::linear_elasticity_2d_function(pt, lambda, mu, res);
		else if (size() == 3)
			autogen::linear_elasticity_3d_function(pt, lambda, mu, res);
		else
			assert(false);

		return res;
	}

	void LinearElasticity::compute_stress_tensor(const int el_id, const ElementBases &bs, const ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, Eigen::MatrixXd &stresses) const
	{
		assign_stress_tensor(el_id, bs, gbs, local_pts, displacement, size() * size(), stresses, [&](const Eigen::MatrixXd &stress) {
			Eigen::MatrixXd tmp = stress;
			auto a = Eigen::Map<Eigen::MatrixXd>(tmp.data(), 1, size() * size());
			return Eigen::MatrixXd(a);
		});
	}

	void LinearElasticity::compute_von_mises_stresses(const int el_id, const ElementBases &bs, const ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, Eigen::MatrixXd &stresses) const
	{
		assign_stress_tensor(el_id, bs, gbs, local_pts, displacement, 1, stresses, [&](const Eigen::MatrixXd &stress) {
			Eigen::Matrix<double, 1, 1> res;
			res.setConstant(von_mises_stress_for_stress_tensor(stress));
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

		for (long p = 0; p < local_pts.rows(); ++p)
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

			const Eigen::MatrixXd strain = (displacement_grad + displacement_grad.transpose()) / 2;
			const Eigen::MatrixXd stress = 2 * mu * strain + lambda * strain.trace() * Eigen::MatrixXd::Identity(size(), size());

			all.row(p) = fun(stress);
		}
	}

	Eigen::Matrix<AutodiffScalarGrad, Eigen::Dynamic, 1, 0, 3, 1> LinearElasticity::kernel(const int dim, const AutodiffGradPt &r) const
	{
		Eigen::Matrix<AutodiffScalarGrad, Eigen::Dynamic, 1, 0, 3, 1> res(dim);
		assert(r.size() == dim);

		double mu, nu, lambda;
		//per body lame parameter dont work here!
		params_.lambda_mu(0, 0, 0, 0, lambda, mu);

		// convert to nu!
		nu = lambda / 2 / (lambda + mu);

		if (dim == 2)
		{
			res(0) = 1. / (8 * M_PI * mu * (1 - nu)) * ((3 - 4 * nu) * log(1. / r.norm()) + r(0) * r(0) / r.squaredNorm());
			res(1) = 1. / (8 * M_PI * mu * (1 - nu)) * r(0) * r(1) / r.squaredNorm();
		}
		else if (dim == 3)
		{
			res(0) = 1. / (16 * M_PI * mu * (1 - nu)) * ((3 - 4 * nu) / r.norm() + r(0) * r(0) / r.norm() / r.squaredNorm());
			res(1) = 1. / (16 * M_PI * mu * (1 - nu)) * r(0) * r(1) / r.norm() / r.squaredNorm();
			res(2) = 1. / (16 * M_PI * mu * (1 - nu)) * r(0) * r(2) / r.norm() / r.squaredNorm();
		}
		else
			assert(false);

		return res;
	}

} // namespace polyfem
