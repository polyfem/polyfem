#include "MooneyRivlinElasticity.hpp"

#include <polyfem/basis/Basis.hpp>
#include <polyfem/autogen/auto_elasticity_rhs.hpp>

#include <polyfem/utils/MatrixUtils.hpp>
#include <igl/Timer.h>

namespace polyfem
{
	using namespace basis;

	namespace assembler
	{
		MooneyRivlinElasticity::MooneyRivlinElasticity()
		{
		}

		void MooneyRivlinElasticity::set_size(const int size)
		{
			size_ = size;
		}

		void MooneyRivlinElasticity::add_multimaterial(const int index, const json &params)
	{
		assert(size_ == 2 || size_ == 3);

		if (params.count("c1"))
			c1_ = params["c1"];
		if (params.count("c2"))
			c2_ = params["c2"];
		if (params.count("k"))
			k_ = params["k"];
	}

		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1>
		MooneyRivlinElasticity::compute_rhs(const AutodiffHessianPt &pt) const
		{
			assert(pt.size() == size());
			Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> res;
			assert(false);

			return res;
		}

		void MooneyRivlinElasticity::compute_stress_tensor(const int el_id, const ElementBases &bs, const ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, Eigen::MatrixXd &stresses) const
		{
			assign_stress_tensor(el_id, bs, gbs, local_pts, displacement, size() * size(), stresses, [&](const Eigen::MatrixXd &stress) {
				Eigen::MatrixXd tmp = stress;
				auto a = Eigen::Map<Eigen::MatrixXd>(tmp.data(), 1, size() * size());
				return Eigen::MatrixXd(a);
			});
		}

		void MooneyRivlinElasticity::compute_von_mises_stresses(const int el_id, const basis::ElementBases &bs, const basis::ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, Eigen::MatrixXd &stresses) const
		{
			assign_stress_tensor(el_id, bs, gbs, local_pts, displacement, 1, stresses, [&](const Eigen::MatrixXd &stress) {
				Eigen::Matrix<double, 1, 1> res;
				res.setConstant(von_mises_stress_for_stress_tensor(stress));
				return res;
			});
		}

		void MooneyRivlinElasticity::assign_stress_tensor(const int el_id, const basis::ElementBases &bs, const basis::ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, const int all_size, Eigen::MatrixXd &all, const std::function<Eigen::MatrixXd(const Eigen::MatrixXd &)> &fun) const
		{
			Eigen::MatrixXd displacement_grad(size(), size());

			assert(displacement.cols() == 1);

			all.resize(local_pts.rows(), all_size);

			ElementAssemblyValues vals;
			vals.compute(el_id, size() == 3, local_pts, bs, gbs);

			for (long p = 0; p < local_pts.rows(); ++p)
			{
				compute_diplacement_grad(size(), bs, vals, local_pts, p, displacement, displacement_grad);

				// const Eigen::MatrixXd def_grad = Eigen::MatrixXd::Identity(size(), size()) + displacement_grad;
				const Eigen::MatrixXd def_grad = displacement_grad;
				const Eigen::MatrixXd FmT = def_grad.inverse().transpose();

				// stress = 2*c1*F + 4*c2*FF^{T}F + k*lnJ*F^{-T}
				Eigen::MatrixXd stress_tensor = 2*c1_*def_grad + 4*c2_*def_grad*def_grad.transpose()*def_grad + k_*std::log(def_grad.determinant())*FmT;

				all.row(p) = fun(stress_tensor);
			}
		}

		Eigen::VectorXd
		MooneyRivlinElasticity::assemble_grad(const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) const
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
		MooneyRivlinElasticity::assemble_hessian(const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) const
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

		double MooneyRivlinElasticity::compute_energy(const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) const
		{
			return compute_energy_aux<double>(vals, displacement, da);
		}

		template <typename T>
		T MooneyRivlinElasticity::compute_energy_aux(const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) const
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

				// // Id + grad d
				// for (int d = 0; d < size(); ++d)
				// 	def_grad(d, d) += T(1);

				const T log_det_j = log(polyfem::utils::determinant(def_grad));
				const T val = c1_ * ((def_grad.transpose() * def_grad).trace() - size()) + c2_ * ((def_grad.transpose() * def_grad * def_grad.transpose() * def_grad).trace() - size()) + k_/2 * (log_det_j * log_det_j);

				energy += val * da(p);
			}
			return energy;
		}
	} // namespace assembler
} // namespace polyfem