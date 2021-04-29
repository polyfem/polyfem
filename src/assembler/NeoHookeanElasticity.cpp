// #define EIGEN_STACK_ALLOCATION_LIMIT 0

#include <polyfem/NeoHookeanElasticity.hpp>

#include <polyfem/Basis.hpp>
#include <polyfem/auto_elasticity_rhs.hpp>

#include <polyfem/MatrixUtils.hpp>
#include <igl/Timer.h>

namespace polyfem
{

	NeoHookeanElasticity::NeoHookeanElasticity()
	{
		set_size(size_);
	}

	void NeoHookeanElasticity::init_multimaterial(const bool is_volume, const Eigen::MatrixXd &Es, const Eigen::MatrixXd &nus)
	{
		params_.init_multimaterial(is_volume, Es, nus);
	}

	void NeoHookeanElasticity::set_parameters(const json &params)
	{
		set_size(params["size"]);

		params_.init(params);
	}

	void NeoHookeanElasticity::set_size(const int size)
	{
		size_ = size;
	}

	Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1>
	NeoHookeanElasticity::compute_rhs(const AutodiffHessianPt &pt) const
	{
		assert(pt.size() == size());
		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> res;

		double lambda, mu;
		params_.lambda_mu(pt(0).getValue(), pt(1).getValue(), size_ == 2 ? 0. : pt(2).getValue(), 0, lambda, mu);

		if (size() == 2)
			autogen::neo_hookean_2d_function(pt, lambda, mu, res);
		else if (size() == 3)
			autogen::neo_hookean_3d_function(pt, lambda, mu, res);
		else
			assert(false);

		return res;
	}

	Eigen::VectorXd
	NeoHookeanElasticity::assemble_grad(const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) const
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

	Eigen::MatrixXd
	NeoHookeanElasticity::assemble_hessian(const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) const
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

	void NeoHookeanElasticity::compute_stress_tensor(const int el_id, const ElementBases &bs, const ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, Eigen::MatrixXd &stresses) const
	{
		assign_stress_tensor(el_id, bs, gbs, local_pts, displacement, size() * size(), stresses, [&](const Eigen::MatrixXd &stress) {
			Eigen::MatrixXd tmp = stress;
			auto a = Eigen::Map<Eigen::MatrixXd>(tmp.data(), 1, size() * size());
			return Eigen::MatrixXd(a);
		});
	}

	void NeoHookeanElasticity::compute_von_mises_stresses(const int el_id, const ElementBases &bs, const ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, Eigen::MatrixXd &stresses) const
	{
		assign_stress_tensor(el_id, bs, gbs, local_pts, displacement, 1, stresses, [&](const Eigen::MatrixXd &stress) {
			Eigen::Matrix<double, 1, 1> res;
			res.setConstant(von_mises_stress_for_stress_tensor(stress));
			return res;
		});
	}

	void NeoHookeanElasticity::assign_stress_tensor(const int el_id, const ElementBases &bs, const ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, const int all_size, Eigen::MatrixXd &all, const std::function<Eigen::MatrixXd(const Eigen::MatrixXd &)> &fun) const
	{
		Eigen::MatrixXd displacement_grad(size(), size());

		assert(displacement.cols() == 1);

		all.resize(local_pts.rows(), all_size);

		ElementAssemblyValues vals;
		vals.compute(el_id, size() == 3, local_pts, bs, gbs);

		for (long p = 0; p < local_pts.rows(); ++p)
		{
			compute_diplacement_grad(size(), bs, vals, local_pts, p, displacement, displacement_grad);

			const Eigen::MatrixXd def_grad = Eigen::MatrixXd::Identity(size(), size()) + displacement_grad;
			const Eigen::MatrixXd FmT = def_grad.inverse().transpose();
			// const double J = def_grad.determinant();

			double lambda, mu;
			params_.lambda_mu(vals.val(p, 0), vals.val(p, 1), size_ == 2 ? 0. : vals.val(p, 2), vals.element_id, lambda, mu);

			//stress = mu (F - F^{-T}) + lambda ln J F^{-T}
			//stress = mu * (def_grad - def_grad^{-T}) + lambda ln (det def_grad) def_grad^{-T}
			Eigen::MatrixXd stress_tensor = mu * (def_grad - FmT) + lambda * std::log(def_grad.determinant()) * FmT;

			//stess = (mu displacement_grad + lambda ln(J) I)/J
			// Eigen::MatrixXd stress_tensor = (mu_/J) * displacement_grad + (lambda_/J) * std::log(J)  * Eigen::MatrixXd::Identity(size(), size());

			all.row(p) = fun(stress_tensor);
		}
	}

	double NeoHookeanElasticity::compute_energy(const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) const
	{
		return compute_energy_aux<double>(vals, displacement, da);
	}

	//Compute \int (mu/2 (trace(F^T F) - 3 - 2*ln(J)) + lambda/2 ln^2(J))
	template <typename T>
	T NeoHookeanElasticity::compute_energy_aux(const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) const
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

			//Id + grad d
			for (int d = 0; d < size(); ++d)
				def_grad(d, d) += T(1);

			double lambda, mu;
			params_.lambda_mu(vals.val(p, 0), vals.val(p, 1), size_ == 2 ? 0. : vals.val(p, 2), vals.element_id, lambda, mu);

			const T log_det_j = log(polyfem::determinant(def_grad));
			const T val = mu / 2 * ((def_grad.transpose() * def_grad).trace() - size() - 2 * log_det_j) + lambda / 2 * log_det_j * log_det_j;

			energy += val * da(p);
		}
		return energy;
	}
} // namespace polyfem
