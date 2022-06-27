#include "NeoHookeanElasticity.hpp"

#include <polyfem/basis/Basis.hpp>
#include <polyfem/autogen/auto_elasticity_rhs.hpp>

#include <polyfem/utils/MatrixUtils.hpp>
#include <igl/Timer.h>

namespace polyfem
{
	using namespace basis;

	namespace assembler
	{
		NeoHookeanElasticity::NeoHookeanElasticity()
		{
		}

		void NeoHookeanElasticity::add_multimaterial(const int index, const json &params)
		{
			assert(size_ == 2 || size_ == 3);

			params_.add_multimaterial(index, params, size_ == 3);
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
			//TODO!
			params_.lambda_mu(0, 0, 0, pt(0).getValue(), pt(1).getValue(), size_ == 2 ? 0. : pt(2).getValue(), 0, lambda, mu);

			if (size() == 2)
				autogen::neo_hookean_2d_function(pt, lambda, mu, res);
			else if (size() == 3)
				autogen::neo_hookean_3d_function(pt, lambda, mu, res);
			else
				assert(false);

			return res;
		}

		// Eigen::VectorXd
		// NeoHookeanElasticity::assemble_grad(const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) const
		// {
		// 	const int n_bases = vals.basis_values.size();

		// 	return polyfem::gradient_from_energy(
		// 		size(), n_bases, vals, displacement, da,
		// 		[&](const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) { return compute_energy_aux<DScalar1<double, Eigen::Matrix<double, 6, 1>>>(vals, displacement, da); },
		// 		[&](const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) { return compute_energy_aux<DScalar1<double, Eigen::Matrix<double, 8, 1>>>(vals, displacement, da); },
		// 		[&](const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) { return compute_energy_aux<DScalar1<double, Eigen::Matrix<double, 12, 1>>>(vals, displacement, da); },
		// 		[&](const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) { return compute_energy_aux<DScalar1<double, Eigen::Matrix<double, 18, 1>>>(vals, displacement, da); },
		// 		[&](const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) { return compute_energy_aux<DScalar1<double, Eigen::Matrix<double, 24, 1>>>(vals, displacement, da); },
		// 		[&](const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) { return compute_energy_aux<DScalar1<double, Eigen::Matrix<double, 30, 1>>>(vals, displacement, da); },
		// 		[&](const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) { return compute_energy_aux<DScalar1<double, Eigen::Matrix<double, 60, 1>>>(vals, displacement, da); },
		// 		[&](const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) { return compute_energy_aux<DScalar1<double, Eigen::Matrix<double, 81, 1>>>(vals, displacement, da); },
		// 		[&](const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) { return compute_energy_aux<DScalar1<double, Eigen::Matrix<double, Eigen::Dynamic, 1, 0, SMALL_N, 1>>>(vals, displacement, da); },
		// 		[&](const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) { return compute_energy_aux<DScalar1<double, Eigen::Matrix<double, Eigen::Dynamic, 1, 0, BIG_N, 1>>>(vals, displacement, da); },
		// 		[&](const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) { return compute_energy_aux<DScalar1<double, Eigen::VectorXd>>(vals, displacement, da); });
		// }

		// Eigen::MatrixXd
		// NeoHookeanElasticity::assemble_hessian(const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) const
		// {
		// 	const int n_bases = vals.basis_values.size();
		// 	const auto HH = polyfem::hessian_from_energy(
		// 		size(), n_bases, vals, displacement, da,
		// 		[&](const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) { return compute_energy_aux<DScalar2<double, Eigen::Matrix<double, 6, 1>, Eigen::Matrix<double, 6, 6>>>(vals, displacement, da); },
		// 		[&](const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) { return compute_energy_aux<DScalar2<double, Eigen::Matrix<double, 8, 1>, Eigen::Matrix<double, 8, 8>>>(vals, displacement, da); },
		// 		[&](const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) { return compute_energy_aux<DScalar2<double, Eigen::Matrix<double, 12, 1>, Eigen::Matrix<double, 12, 12>>>(vals, displacement, da); },
		// 		[&](const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) { return compute_energy_aux<DScalar2<double, Eigen::Matrix<double, 18, 1>, Eigen::Matrix<double, 18, 18>>>(vals, displacement, da); },
		// 		[&](const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) { return compute_energy_aux<DScalar2<double, Eigen::Matrix<double, 24, 1>, Eigen::Matrix<double, 24, 24>>>(vals, displacement, da); },
		// 		[&](const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) { return compute_energy_aux<DScalar2<double, Eigen::Matrix<double, 30, 1>, Eigen::Matrix<double, 30, 30>>>(vals, displacement, da); },
		// 		[&](const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) { return compute_energy_aux<DScalar2<double, Eigen::Matrix<double, 60, 1>, Eigen::Matrix<double, 60, 60>>>(vals, displacement, da); },
		// 		[&](const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) { return compute_energy_aux<DScalar2<double, Eigen::Matrix<double, 81, 1>, Eigen::Matrix<double, 81, 81>>>(vals, displacement, da); },
		// 		[&](const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) { return compute_energy_aux<DScalar2<double, Eigen::Matrix<double, Eigen::Dynamic, 1, 0, SMALL_N, 1>, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, SMALL_N, SMALL_N>>>(vals, displacement, da); },
		// 		[&](const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) { return compute_energy_aux<DScalar2<double, Eigen::VectorXd, Eigen::MatrixXd>>(vals, displacement, da); });

		// 	Eigen::MatrixXd hessian;
		// 	hessian.resize(12, 12);
		// 	hessian.setZero();
		// 	compute_energy_aux_fast<4, 3>(vals, displacement, da, hessian);

		// 	double bad = false;
		// 	for (int i = 0; i < HH.size(); ++i)
		// 	{
		// 		if (fabs(HH(i) - hessian(i)) > 1e-10)
		// 		{
		// 			std::cout << HH(i) << " " << hessian(i) << " " << fabs(HH(i) - hessian(i)) << std::endl;
		// 			bad = true;
		// 		}
		// 	}

		// 	if (bad)
		// 		exit(0);

		// 	return HH;
		// }

		Eigen::VectorXd
		NeoHookeanElasticity::assemble_grad(const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) const
		{
			Eigen::Matrix<double, Eigen::Dynamic, 1> gradient;

			if (size() == 2)
			{
				if (vals.basis_values.size() == 3)
				{
					gradient.resize(6);
					compute_energy_aux_gradient_fast<3, 2>(vals, displacement, da, gradient);
				}
				else if (vals.basis_values.size() == 6)
				{
					gradient.resize(12);
					compute_energy_aux_gradient_fast<6, 2>(vals, displacement, da, gradient);
				}
				else if (vals.basis_values.size() == 10)
				{
					gradient.resize(20);
					compute_energy_aux_gradient_fast<10, 2>(vals, displacement, da, gradient);
				}
				else
				{
					gradient.resize(vals.basis_values.size() * 2);
					compute_energy_aux_gradient_fast<Eigen::Dynamic, 2>(vals, displacement, da, gradient);
				}
			}
			else //if (size() == 3)
			{
				assert(size() == 3);
				if (vals.basis_values.size() == 4)
				{
					gradient.resize(12);
					compute_energy_aux_gradient_fast<4, 3>(vals, displacement, da, gradient);
				}
				else if (vals.basis_values.size() == 10)
				{
					gradient.resize(30);
					compute_energy_aux_gradient_fast<10, 3>(vals, displacement, da, gradient);
				}
				else if (vals.basis_values.size() == 20)
				{
					gradient.resize(60);
					compute_energy_aux_gradient_fast<20, 3>(vals, displacement, da, gradient);
				}
				else
				{
					gradient.resize(vals.basis_values.size() * 3);
					compute_energy_aux_gradient_fast<Eigen::Dynamic, 3>(vals, displacement, da, gradient);
				}
			}

			return gradient;
		}

		Eigen::MatrixXd
		NeoHookeanElasticity::assemble_hessian(const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) const
		{
			Eigen::MatrixXd hessian;

			if (size() == 2)
			{
				if (vals.basis_values.size() == 3)
				{
					hessian.resize(6, 6);
					hessian.setZero();
					compute_energy_hessian_aux_fast<3, 2>(vals, displacement, da, hessian);
				}
				else if (vals.basis_values.size() == 6)
				{
					hessian.resize(12, 12);
					hessian.setZero();
					compute_energy_hessian_aux_fast<6, 2>(vals, displacement, da, hessian);
				}
				else if (vals.basis_values.size() == 10)
				{
					hessian.resize(20, 20);
					hessian.setZero();
					compute_energy_hessian_aux_fast<10, 2>(vals, displacement, da, hessian);
				}
				else
				{
					hessian.resize(vals.basis_values.size() * 2, vals.basis_values.size() * 2);
					hessian.setZero();
					compute_energy_hessian_aux_fast<Eigen::Dynamic, 2>(vals, displacement, da, hessian);
				}
			}
			else //if (size() == 3)
			{
				assert(size() == 3);
				if (vals.basis_values.size() == 4)
				{
					hessian.resize(12, 12);
					hessian.setZero();
					compute_energy_hessian_aux_fast<4, 3>(vals, displacement, da, hessian);
				}
				else if (vals.basis_values.size() == 10)
				{
					hessian.resize(30, 30);
					hessian.setZero();
					compute_energy_hessian_aux_fast<10, 3>(vals, displacement, da, hessian);
				}
				else if (vals.basis_values.size() == 20)
				{
					hessian.resize(60, 60);
					hessian.setZero();
					compute_energy_hessian_aux_fast<20, 3>(vals, displacement, da, hessian);
				}
				else
				{
					hessian.resize(vals.basis_values.size() * 3, vals.basis_values.size() * 3);
					hessian.setZero();
					compute_energy_hessian_aux_fast<Eigen::Dynamic, 3>(vals, displacement, da, hessian);
				}
			}

			return hessian;
		}

		void NeoHookeanElasticity::compute_stress_tensor(const int el_id, const ElementBases &bs, const ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, Eigen::MatrixXd &stresses) const
		{
			assign_stress_tensor(el_id, bs, gbs, local_pts, displacement, size() * size(), stresses, [&](const Eigen::MatrixXd &stress) {
				Eigen::MatrixXd tmp = stress;
				auto a = Eigen::Map<Eigen::MatrixXd>(tmp.data(), 1, size() * size());
				return Eigen::MatrixXd(a);
			});
		}

		void NeoHookeanElasticity::compute_von_mises_stresses(const int el_id, const basis::ElementBases &bs, const basis::ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, Eigen::MatrixXd &stresses) const
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
				params_.lambda_mu(local_pts.row(p), vals.val.row(p), vals.element_id, lambda, mu);

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

		// Compute ∫ ½μ (tr(FᵀF) - 3 - 2ln(J)) + ½λ ln²(J) du
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
				params_.lambda_mu(vals.quadrature.points.row(p), vals.val.row(p), vals.element_id, lambda, mu);

				const T log_det_j = log(polyfem::utils::determinant(def_grad));
				const T val = mu / 2 * ((def_grad.transpose() * def_grad).trace() - size() - 2 * log_det_j) + lambda / 2 * log_det_j * log_det_j;

				energy += val * da(p);
			}
			return energy;
		}

		template <int dim>
		Eigen::Matrix<double, dim, dim> hat(const Eigen::Matrix<double, dim, 1> &x)
		{

			Eigen::Matrix<double, dim, dim> prod;
			prod.setZero();

			prod(0, 1) = -x(2);
			prod(0, 2) = x(1);
			prod(1, 0) = x(2);
			prod(1, 2) = -x(0);
			prod(2, 0) = -x(1);
			prod(2, 1) = x(0);

			return prod;
		}

		template <int dim>
		Eigen::Matrix<double, dim, 1> cross(const Eigen::Matrix<double, dim, 1> &x, const Eigen::Matrix<double, dim, 1> &y)
		{

			Eigen::Matrix<double, dim, 1> z;
			z.setZero();

			z(0) = x(1) * y(2) - x(2) * y(1);
			z(1) = x(2) * y(0) - x(0) * y(2);
			z(2) = x(0) * y(1) - x(1) * y(0);

			return z;
		}

		template <int n_basis, int dim>
		void NeoHookeanElasticity::compute_energy_aux_gradient_fast(const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da, Eigen::Matrix<double, Eigen::Dynamic, 1> &G_flattened) const
		{

			assert(displacement.cols() == 1);

			const int n_pts = da.size();

			Eigen::Matrix<double, n_basis, dim> local_disp(vals.basis_values.size(), size());
			local_disp.setZero();
			for (size_t i = 0; i < vals.basis_values.size(); ++i)
			{
				const auto &bs = vals.basis_values[i];
				for (size_t ii = 0; ii < bs.global.size(); ++ii)
				{
					for (int d = 0; d < size(); ++d)
					{
						local_disp(i, d) += bs.global[ii].val * displacement(bs.global[ii].index * size() + d);
					}
				}
			}

			Eigen::Matrix<double, dim, dim> def_grad(size(), size());

			Eigen::Matrix<double, n_basis, dim> G(vals.basis_values.size(), size());
			G.setZero();

			for (long p = 0; p < n_pts; ++p)
			{
				Eigen::Matrix<double, n_basis, dim> grad(vals.basis_values.size(), size());

				for (size_t i = 0; i < vals.basis_values.size(); ++i)
				{
					grad.row(i) = vals.basis_values[i].grad.row(p);
				}

				Eigen::Matrix<double, dim, dim> jac_it = vals.jac_it[p];

				//Id + grad d
				def_grad = local_disp.transpose() * grad * jac_it + Eigen::Matrix<double, dim, dim>::Identity(size(), size());

				const double J = def_grad.determinant();
				const double log_det_j = log(J);

				Eigen::Matrix<double, dim, dim> delJ_delF(size(), size());
				delJ_delF.setZero();

				if (dim == 2)
				{

					delJ_delF(0, 0) = def_grad(1, 1);
					delJ_delF(0, 1) = -def_grad(1, 0);
					delJ_delF(1, 0) = -def_grad(0, 1);
					delJ_delF(1, 1) = def_grad(0, 0);
				}

				else if (dim == 3)
				{

					Eigen::Matrix<double, dim, 1> u(def_grad.rows());
					Eigen::Matrix<double, dim, 1> v(def_grad.rows());
					Eigen::Matrix<double, dim, 1> w(def_grad.rows());

					u = def_grad.col(0);
					v = def_grad.col(1);
					w = def_grad.col(2);

					delJ_delF.col(0) = cross<dim>(v, w);
					delJ_delF.col(1) = cross<dim>(w, u);
					delJ_delF.col(2) = cross<dim>(u, v);
				}

				double lambda, mu;
				params_.lambda_mu(vals.quadrature.points.row(p), vals.val.row(p), vals.element_id, lambda, mu);

				Eigen::Matrix<double, n_basis, dim> delF_delU = grad * jac_it;

				Eigen::Matrix<double, dim, dim> gradient_temp = mu * def_grad - mu * (1 / J) * delJ_delF + lambda * log_det_j * (1 / J) * delJ_delF;
				Eigen::Matrix<double, n_basis, dim> gradient = delF_delU * gradient_temp.transpose();

				double val = mu / 2 * ((def_grad.transpose() * def_grad).trace() - size() - 2 * log_det_j) + lambda / 2 * log_det_j * log_det_j;

				G.noalias() += gradient * da(p);
			}

			Eigen::Matrix<double, dim, n_basis> G_T = G.transpose();

			constexpr int N = (n_basis == Eigen::Dynamic) ? Eigen::Dynamic : n_basis * dim;
			Eigen::Matrix<double, N, 1> temp(Eigen::Map<Eigen::Matrix<double, N, 1>>(G_T.data(), G_T.size()));
			G_flattened = temp;
		}

		template <int n_basis, int dim>
		void NeoHookeanElasticity::compute_energy_hessian_aux_fast(const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da, Eigen::MatrixXd &H) const
		{
			assert(displacement.cols() == 1);

			constexpr int N = (n_basis == Eigen::Dynamic) ? Eigen::Dynamic : n_basis * dim;
			const int n_pts = da.size();

			Eigen::Matrix<double, n_basis, dim> local_disp(vals.basis_values.size(), size());
			local_disp.setZero();
			for (size_t i = 0; i < vals.basis_values.size(); ++i)
			{
				const auto &bs = vals.basis_values[i];
				for (size_t ii = 0; ii < bs.global.size(); ++ii)
				{
					for (int d = 0; d < size(); ++d)
					{
						local_disp(i, d) += bs.global[ii].val * displacement(bs.global[ii].index * size() + d);
					}
				}
			}

			Eigen::Matrix<double, dim, dim> def_grad(size(), size());

			for (long p = 0; p < n_pts; ++p)
			{
				Eigen::Matrix<double, n_basis, dim> grad(vals.basis_values.size(), size());

				for (size_t i = 0; i < vals.basis_values.size(); ++i)
				{
					grad.row(i) = vals.basis_values[i].grad.row(p);
				}

				Eigen::Matrix<double, dim, dim> jac_it = vals.jac_it[p];

				//Id + grad d
				def_grad = local_disp.transpose() * grad * jac_it + Eigen::Matrix<double, dim, dim>::Identity(size(), size());

				const double J = def_grad.determinant();
				double log_det_j = log(J);

				Eigen::Matrix<double, dim, dim> delJ_delF(size(), size());
				delJ_delF.setZero();
				Eigen::Matrix<double, dim * dim, dim * dim> del2J_delF2(size() * size(), size() * size());
				del2J_delF2.setZero();

				if (dim == 2)
				{
					delJ_delF(0, 0) = def_grad(1, 1);
					delJ_delF(0, 1) = -def_grad(1, 0);
					delJ_delF(1, 0) = -def_grad(0, 1);
					delJ_delF(1, 1) = def_grad(0, 0);

					del2J_delF2(0, 3) = 1;
					del2J_delF2(1, 2) = -1;
					del2J_delF2(2, 1) = -1;
					del2J_delF2(3, 0) = 1;
				}
				else if (size() == 3)
				{
					Eigen::Matrix<double, dim, 1> u(def_grad.rows());
					Eigen::Matrix<double, dim, 1> v(def_grad.rows());
					Eigen::Matrix<double, dim, 1> w(def_grad.rows());

					u = def_grad.col(0);
					v = def_grad.col(1);
					w = def_grad.col(2);

					delJ_delF.col(0) = cross<dim>(v, w);
					delJ_delF.col(1) = cross<dim>(w, u);
					delJ_delF.col(2) = cross<dim>(u, v);

					del2J_delF2.template block<dim, dim>(0, 6) = hat<dim>(v);
					del2J_delF2.template block<dim, dim>(6, 0) = -hat<dim>(v);
					del2J_delF2.template block<dim, dim>(0, 3) = -hat<dim>(w);
					del2J_delF2.template block<dim, dim>(3, 0) = hat<dim>(w);
					del2J_delF2.template block<dim, dim>(3, 6) = -hat<dim>(u);
					del2J_delF2.template block<dim, dim>(6, 3) = hat<dim>(u);
				}

				double lambda, mu;
				params_.lambda_mu(vals.quadrature.points.row(p), vals.val.row(p), vals.element_id, lambda, mu);

				Eigen::Matrix<double, dim * dim, dim *dim> id = Eigen::Matrix<double, dim * dim, dim * dim>::Identity(size() * size(), size() * size());

				Eigen::Matrix<double, dim * dim, 1> g_j = Eigen::Map<const Eigen::Matrix<double, dim * dim, 1>>(delJ_delF.data(), delJ_delF.size());

				Eigen::Matrix<double, dim * dim, dim *dim> hessian_temp = (mu * id) + (((mu + lambda * (1 - log_det_j)) / (J * J)) * (g_j * g_j.transpose())) + (((lambda * log_det_j - mu) / (J)) * del2J_delF2);

				Eigen::Matrix<double, dim * dim, N> delF_delU_tensor(jac_it.size(), grad.size());

				for (size_t i = 0; i < local_disp.rows(); ++i)
				{
					for (size_t j = 0; j < local_disp.cols(); ++j)
					{
						Eigen::Matrix<double, dim, dim> temp(size(), size());
						temp.setZero();
						temp.row(j) = grad.row(i);
						temp = temp * jac_it;
						Eigen::Matrix<double, dim * dim, 1> temp_flattened(Eigen::Map<Eigen::Matrix<double, dim * dim, 1>>(temp.data(), temp.size()));
						delF_delU_tensor.col(i * size() + j) = temp_flattened;
					}
				}

				Eigen::Matrix<double, N, N> hessian = delF_delU_tensor.transpose() * hessian_temp * delF_delU_tensor;

				double val = mu / 2 * ((def_grad.transpose() * def_grad).trace() - size() - 2 * log_det_j) + lambda / 2 * log_det_j * log_det_j;

				H += hessian * da(p);
			}
		}

		void NeoHookeanElasticity::compute_dstress_dgradu_multiply_mat(const int el_id, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &global_pts, const Eigen::MatrixXd &grad_u_i, const Eigen::MatrixXd &mat, Eigen::MatrixXd &stress, Eigen::MatrixXd &result) const
		{
			double lambda, mu;
			params_.lambda_mu(local_pts, global_pts, el_id, lambda, mu);

			Eigen::MatrixXd def_grad = Eigen::MatrixXd::Identity(grad_u_i.rows(), grad_u_i.cols()) + grad_u_i;
			Eigen::MatrixXd FmT = def_grad.inverse().transpose();

			stress = mu * (def_grad - FmT) + lambda * std::log(def_grad.determinant()) * FmT;
			result = mu * mat + FmT * mat.transpose() * FmT * (mu - lambda * std::log(def_grad.determinant())) + lambda * (FmT.array() * mat.array()).sum() * FmT;
		}

		void NeoHookeanElasticity::compute_dstress_dmu_dlambda(const int el_id, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &global_pts, const Eigen::MatrixXd &grad_u_i, Eigen::MatrixXd &dstress_dmu, Eigen::MatrixXd &dstress_dlambda) const
		{
			Eigen::MatrixXd def_grad = Eigen::MatrixXd::Identity(grad_u_i.rows(), grad_u_i.cols()) + grad_u_i;
			Eigen::MatrixXd FmT = def_grad.inverse().transpose();
			dstress_dmu = def_grad - FmT;
			dstress_dlambda = std::log(def_grad.determinant()) * FmT;
		}
	} // namespace assembler
} // namespace polyfem
