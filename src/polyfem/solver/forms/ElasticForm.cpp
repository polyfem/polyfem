#include "ElasticForm.hpp"

#include <polyfem/io/Evaluator.hpp>
#include <polyfem/basis/ElementBases.hpp>
#include <polyfem/assembler/MatParams.hpp>
#include <polyfem/utils/Timer.hpp>
#include <polyfem/utils/MaybeParallelFor.hpp>
#include <polyfem/assembler/ViscousDamping.hpp>

using namespace polyfem::assembler;
using namespace polyfem::utils;

namespace polyfem::solver
{
	namespace
	{
		class LocalThreadVecStorage
		{
		public:
			Eigen::MatrixXd vec;
			assembler::ElementAssemblyValues vals;
			QuadratureVector da;

			LocalThreadVecStorage(const int size)
			{
				vec.resize(size, 1);
				vec.setZero();
			}
		};

		double dot(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B) { return (A.array() * B.array()).sum(); }
	} // namespace
	
	ElasticForm::ElasticForm(const int n_bases,
							 const std::vector<basis::ElementBases> &bases,
							 const std::vector<basis::ElementBases> &geom_bases,
							 const assembler::Assembler &assembler,
							 const assembler::AssemblyValsCache &ass_vals_cache,
							 const double dt,
							 const bool is_volume)
		: n_bases_(n_bases),
		  bases_(bases),
		  geom_bases_(geom_bases),
		  assembler_(assembler),
		  ass_vals_cache_(ass_vals_cache),
		  dt_(dt),
		  is_volume_(is_volume)
	{
		if (assembler_.is_linear())
			compute_cached_stiffness();
	}

	double ElasticForm::value_unweighted(const Eigen::VectorXd &x) const
	{
		return assembler_.assemble_energy(is_volume_, bases_, geom_bases_,
										  ass_vals_cache_, dt_, x, x_prev_);
	}

	void ElasticForm::first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		Eigen::MatrixXd grad;
		assembler_.assemble_gradient(is_volume_, n_bases_, bases_, geom_bases_,
									 ass_vals_cache_, dt_, x, x_prev_, grad);
		gradv = grad;
	}

	void ElasticForm::second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const
	{
		POLYFEM_SCOPED_TIMER("elastic hessian");

		hessian.resize(x.size(), x.size());

		if (assembler_.is_linear())
		{
			assert(cached_stiffness_.rows() == x.size() && cached_stiffness_.cols() == x.size());
			hessian = cached_stiffness_;
		}
		else
		{
			// NOTE: mat_cache_ is marked as mutable so we can modify it here
			assembler_.assemble_hessian(
				is_volume_, n_bases_, project_to_psd_, bases_,
				geom_bases_, ass_vals_cache_, dt_, x, x_prev_, mat_cache_, hessian);
		}
	}

	bool ElasticForm::is_step_valid(const Eigen::VectorXd &, const Eigen::VectorXd &x1) const
	{
		Eigen::VectorXd grad;
		first_derivative(x1, grad);

		if (grad.array().isNaN().any())
			return false;

		// Check the scalar field in the output does not contain NANs.
		// WARNING: Does not work because the energy is not evaluated at the same quadrature points.
		//          This causes small step lengths in the LS.
		// TVector x1_full;
		// reduced_to_full(x1, x1_full);
		// return state_.check_scalar_value(x1_full, true, false);
		return true;
	}

	void ElasticForm::compute_cached_stiffness()
	{
		if (assembler_.is_linear() && cached_stiffness_.size() == 0)
		{
			assembler_.assemble(is_volume_, n_bases_, bases_, geom_bases_,
								ass_vals_cache_, cached_stiffness_);
		}
	}

	void ElasticForm::force_material_derivative(const Eigen::MatrixXd &x, const Eigen::MatrixXd &x_prev, const Eigen::MatrixXd &adjoint, Eigen::VectorXd &term)
	{
		const int dim = is_volume_ ? 3 : 2;

		const int n_elements = int(bases_.size());

		if (assembler_.name() == "ViscousDamping")
		{
			term.setZero(2);

			auto storage = utils::create_thread_storage(LocalThreadVecStorage(term.size()));

			utils::maybe_parallel_for(n_elements, [&](int start, int end, int thread_id) {
				LocalThreadVecStorage &local_storage = utils::get_local_thread_storage(storage, thread_id);

				for (int e = start; e < end; ++e)
				{
					assembler::ElementAssemblyValues &vals = local_storage.vals;
					ass_vals_cache_.compute(e, is_volume_, bases_[e], geom_bases_[e], vals);

					const quadrature::Quadrature &quadrature = vals.quadrature;
					local_storage.da = vals.det.array() * quadrature.weights.array();

					Eigen::MatrixXd u, grad_u, prev_u, prev_grad_u, p, grad_p;
					io::Evaluator::interpolate_at_local_vals(e, dim, dim, vals, x, u, grad_u);
					io::Evaluator::interpolate_at_local_vals(e, dim, dim, vals, x_prev, prev_u, prev_grad_u);
					io::Evaluator::interpolate_at_local_vals(e, dim, dim, vals, adjoint, p, grad_p);

					for (int q = 0; q < local_storage.da.size(); ++q)
					{
						Eigen::MatrixXd grad_p_i, grad_u_i, prev_grad_u_i;
						vector2matrix(grad_p.row(q), grad_p_i);
						vector2matrix(grad_u.row(q), grad_u_i);
						vector2matrix(prev_grad_u.row(q), prev_grad_u_i);

						Eigen::MatrixXd f_prime_dpsi, f_prime_dphi;
						assembler::ViscousDamping::compute_dstress_dpsi_dphi(e, dt_, quadrature.points.row(q), vals.val.row(q), grad_u_i, prev_grad_u_i, f_prime_dpsi, f_prime_dphi);

						// This needs to be a sum over material parameter basis.
						local_storage.vec(0) += -dot(f_prime_dpsi, grad_p_i) * local_storage.da(q);
						local_storage.vec(1) += -dot(f_prime_dphi, grad_p_i) * local_storage.da(q);
					}
				}
			});

			for (const LocalThreadVecStorage &local_storage : storage)
				term += local_storage.vec;
		}
		else
		{
			term.setZero(n_elements * 2, 1);

			auto storage = utils::create_thread_storage(LocalThreadVecStorage(term.size()));

			utils::maybe_parallel_for(n_elements, [&](int start, int end, int thread_id) {
				LocalThreadVecStorage &local_storage = utils::get_local_thread_storage(storage, thread_id);

				for (int e = start; e < end; ++e)
				{
					assembler::ElementAssemblyValues &vals = local_storage.vals;
					ass_vals_cache_.compute(e, is_volume_, bases_[e], geom_bases_[e], vals);

					const quadrature::Quadrature &quadrature = vals.quadrature;
					local_storage.da = vals.det.array() * quadrature.weights.array();

					Eigen::MatrixXd u, grad_u, p, grad_p;
					io::Evaluator::interpolate_at_local_vals(e, dim, dim, vals, x, u, grad_u);
					io::Evaluator::interpolate_at_local_vals(e, dim, dim, vals, adjoint, p, grad_p);

					for (int q = 0; q < local_storage.da.size(); ++q)
					{
						Eigen::MatrixXd grad_p_i, grad_u_i;
						vector2matrix(grad_p.row(q), grad_p_i);
						vector2matrix(grad_u.row(q), grad_u_i);

						Eigen::MatrixXd f_prime_dmu, f_prime_dlambda;
						assembler_.compute_dstress_dmu_dlambda(e, quadrature.points.row(q), vals.val.row(q), grad_u_i, f_prime_dmu, f_prime_dlambda);

						// This needs to be a sum over material parameter basis.
						local_storage.vec(e + n_elements) += -dot(f_prime_dmu, grad_p_i) * local_storage.da(q);
						local_storage.vec(e) += -dot(f_prime_dlambda, grad_p_i) * local_storage.da(q);
					}
				}
			});

			for (const LocalThreadVecStorage &local_storage : storage)
				term += local_storage.vec;
		}
	}

	void ElasticForm::force_shape_derivative(const int n_verts, const Eigen::MatrixXd &x, const Eigen::MatrixXd &x_prev, const Eigen::MatrixXd &adjoint, Eigen::VectorXd &term)
	{
		const int dim = is_volume_ ? 3 : 2;
		const int actual_dim = (assembler_.name() == "Laplacian") ? 1 : dim;

		const int n_elements = int(bases_.size());
		term.setZero(n_verts * dim, 1);

		auto storage = utils::create_thread_storage(LocalThreadVecStorage(term.size()));

		if (assembler_.name() == "ViscousDamping")
		{
			utils::maybe_parallel_for(n_elements, [&](int start, int end, int thread_id) {
				LocalThreadVecStorage &local_storage = utils::get_local_thread_storage(storage, thread_id);

				for (int e = start; e < end; ++e)
				{
					assembler::ElementAssemblyValues &vals = local_storage.vals;
					ass_vals_cache_.compute(e, is_volume_, bases_[e], geom_bases_[e], vals);
					assembler::ElementAssemblyValues gvals;
					gvals.compute(e, is_volume_, vals.quadrature.points, geom_bases_[e], geom_bases_[e]);

					const quadrature::Quadrature &quadrature = vals.quadrature;
					local_storage.da = vals.det.array() * quadrature.weights.array();

					Eigen::MatrixXd u, grad_u, prev_u, prev_grad_u, p, grad_p;
					io::Evaluator::interpolate_at_local_vals(e, dim, dim, vals, x, u, grad_u);
					io::Evaluator::interpolate_at_local_vals(e, dim, dim, vals, x_prev, prev_u, prev_grad_u);
					io::Evaluator::interpolate_at_local_vals(e, dim, dim, vals, adjoint, p, grad_p);

					Eigen::MatrixXd grad_u_i, grad_p_i, prev_grad_u_i;
					Eigen::MatrixXd grad_v_i;
					Eigen::MatrixXd stress_tensor, f_prime_gradu_gradv;
					Eigen::MatrixXd f_prev_prime_prev_gradu_gradv;

					for (int q = 0; q < local_storage.da.size(); ++q)
					{
						vector2matrix(grad_u.row(q), grad_u_i);
						vector2matrix(grad_p.row(q), grad_p_i);
						vector2matrix(prev_grad_u.row(q), prev_grad_u_i);

						for (auto &v : gvals.basis_values)
						{
							Eigen::MatrixXd stress_grad, stress_prev_grad;
							assembler_.compute_stress_grad(e, dt_, quadrature.points.row(q), vals.val.row(q), grad_u_i, prev_grad_u_i, stress_tensor, stress_grad);
							assembler_.compute_stress_prev_grad(e, dt_, quadrature.points.row(q), vals.val.row(q), grad_u_i, prev_grad_u_i, stress_prev_grad);
							for (int d = 0; d < dim; d++)
							{
								grad_v_i.setZero(dim, dim);
								grad_v_i.row(d) = v.grad_t_m.row(q);

								f_prime_gradu_gradv.setZero(dim, dim);
								Eigen::MatrixXd tmp = grad_u_i * grad_v_i;
								for (int i = 0; i < f_prime_gradu_gradv.rows(); i++)
									for (int j = 0; j < f_prime_gradu_gradv.cols(); j++)
										for (int k = 0; k < tmp.rows(); k++)
											for (int l = 0; l < tmp.cols(); l++)
												f_prime_gradu_gradv(i, j) += stress_grad(i * dim + j, k * dim + l) * tmp(k, l);

								f_prev_prime_prev_gradu_gradv.setZero(dim, dim);
								tmp = prev_grad_u_i * grad_v_i;
								for (int i = 0; i < f_prev_prime_prev_gradu_gradv.rows(); i++)
									for (int j = 0; j < f_prev_prime_prev_gradu_gradv.cols(); j++)
										for (int k = 0; k < tmp.rows(); k++)
											for (int l = 0; l < tmp.cols(); l++)
												f_prev_prime_prev_gradu_gradv(i, j) += stress_prev_grad(i * dim + j, k * dim + l) * tmp(k, l);

								tmp = grad_v_i - grad_v_i.trace() * Eigen::MatrixXd::Identity(dim, dim);
								local_storage.vec(v.global[0].index * dim + d) -= dot(f_prime_gradu_gradv + f_prev_prime_prev_gradu_gradv + stress_tensor * tmp.transpose(), grad_p_i) * local_storage.da(q);
							}
						}
					}
				}
			});
		}
		else
		{
			utils::maybe_parallel_for(n_elements, [&](int start, int end, int thread_id) {
				LocalThreadVecStorage &local_storage = utils::get_local_thread_storage(storage, thread_id);

				for (int e = start; e < end; ++e)
				{
					assembler::ElementAssemblyValues &vals = local_storage.vals;
					ass_vals_cache_.compute(e, is_volume_, bases_[e], geom_bases_[e], vals);
					assembler::ElementAssemblyValues gvals;
					gvals.compute(e, is_volume_, vals.quadrature.points, geom_bases_[e], geom_bases_[e]);

					const quadrature::Quadrature &quadrature = vals.quadrature;
					local_storage.da = vals.det.array() * quadrature.weights.array();

					Eigen::MatrixXd u, grad_u, p, grad_p; //, stiffnesses;
					io::Evaluator::interpolate_at_local_vals(e, dim, actual_dim, vals, x, u, grad_u);
					io::Evaluator::interpolate_at_local_vals(e, dim, actual_dim, vals, adjoint, p, grad_p);
					// assembler_.compute_stiffness_value(formulation_, vals, quadrature.points, x, stiffnesses);

					for (int q = 0; q < local_storage.da.size(); ++q)
					{
						Eigen::MatrixXd grad_u_i, grad_p_i, stiffness_i;
						if (actual_dim == 1)
						{
							grad_u_i = grad_u.row(q);
							grad_p_i = grad_p.row(q);
						}
						else
						{
							vector2matrix(grad_u.row(q), grad_u_i);
							vector2matrix(grad_p.row(q), grad_p_i);
						}

						// stiffness_i = utils::unflatten(stiffnesses.row(q).transpose(), actual_dim * dim);

						for (auto &v : gvals.basis_values)
						{
							for (int d = 0; d < dim; d++)
							{
								Eigen::MatrixXd grad_v_i;
								grad_v_i.setZero(dim, dim);
								grad_v_i.row(d) = v.grad_t_m.row(q);

								Eigen::MatrixXd stress_tensor, f_prime_gradu_gradv;
								assembler_.compute_stress_grad_multiply_mat(e, quadrature.points.row(q), vals.val.row(q), grad_u_i, grad_u_i * grad_v_i, stress_tensor, f_prime_gradu_gradv);
								// f_prime_gradu_gradv = utils::unflatten(stiffness_i * utils::flatten(grad_u_i * grad_v_i), dim);

								Eigen::MatrixXd tmp = grad_v_i - grad_v_i.trace() * Eigen::MatrixXd::Identity(dim, dim);
								local_storage.vec(v.global[0].index * dim + d) -= dot(f_prime_gradu_gradv + stress_tensor * tmp.transpose(), grad_p_i) * local_storage.da(q);
							}
						}
					}
				}
			});
		}

		for (const LocalThreadVecStorage &local_storage : storage)
			term += local_storage.vec;
	}
} // namespace polyfem::solver
