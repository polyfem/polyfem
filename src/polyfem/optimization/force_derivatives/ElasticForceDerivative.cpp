#include "ElasticForceDerivative.hpp"

#include <Eigen/Core>
#include <polyfem/solver/forms/ElasticForm.hpp>
#include <polyfem/assembler/ElementAssemblyValues.hpp>
#include <polyfem/assembler/MatParams.hpp>
#include <polyfem/assembler/ViscousDamping.hpp>
#include <polyfem/io/Evaluator.hpp>
#include <polyfem/quadrature/Quadrature.hpp>
#include <polyfem/utils/MatrixUtils.hpp>
#include <polyfem/utils/MaybeParallelFor.hpp>
#include <polyfem/utils/Types.hpp>

using namespace polyfem::assembler;

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
	} // namespace

	void ElasticForceDerivative::force_material_derivative(
		ElasticForm &form,
		const double t,
		const Eigen::MatrixXd &x,
		const Eigen::MatrixXd &x_prev,
		const Eigen::MatrixXd &adjoint,
		Eigen::VectorXd &term)
	{
		const int dim = form.is_volume_ ? 3 : 2;

		const int n_elements = int(form.bases_.size());

		if (form.assembler_.name() == "ViscousDamping")
		{
			term.setZero(2);

			auto storage = utils::create_thread_storage(LocalThreadVecStorage(term.size()));

			utils::maybe_parallel_for(n_elements, [&](int start, int end, int thread_id) {
				LocalThreadVecStorage &local_storage = utils::get_local_thread_storage(storage, thread_id);

				for (int e = start; e < end; ++e)
				{
					assembler::ElementAssemblyValues &vals = local_storage.vals;
					form.ass_vals_cache_.compute(e, form.is_volume_, form.bases_[e], form.geom_bases_[e], vals);

					const quadrature::Quadrature &quadrature = vals.quadrature;
					local_storage.da = vals.det.array() * quadrature.weights.array();

					Eigen::MatrixXd u, grad_u, prev_u, prev_grad_u, p, grad_p;
					io::Evaluator::interpolate_at_local_vals(e, dim, dim, vals, x, u, grad_u);
					io::Evaluator::interpolate_at_local_vals(e, dim, dim, vals, x_prev, prev_u, prev_grad_u);
					io::Evaluator::interpolate_at_local_vals(e, dim, dim, vals, adjoint, p, grad_p);

					for (int q = 0; q < local_storage.da.size(); ++q)
					{
						Eigen::MatrixXd grad_p_i, grad_u_i, prev_grad_u_i;
						utils::vector2matrix(grad_p.row(q), grad_p_i);
						utils::vector2matrix(grad_u.row(q), grad_u_i);
						utils::vector2matrix(prev_grad_u.row(q), prev_grad_u_i);

						Eigen::MatrixXd f_prime_dpsi, f_prime_dphi;
						assembler::ViscousDamping::compute_dstress_dpsi_dphi(
							OptAssemblerData(t, form.dt_, e, quadrature.points.row(q), vals.val.row(q), grad_u_i),
							prev_grad_u_i, f_prime_dpsi, f_prime_dphi);

						// This needs to be a sum over material parameter basis.
						local_storage.vec(0) += -utils::matrix_inner_product<double>(f_prime_dpsi, grad_p_i) * local_storage.da(q);
						local_storage.vec(1) += -utils::matrix_inner_product<double>(f_prime_dphi, grad_p_i) * local_storage.da(q);
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
					form.ass_vals_cache_.compute(e, form.is_volume_, form.bases_[e], form.geom_bases_[e], vals);

					const quadrature::Quadrature &quadrature = vals.quadrature;
					local_storage.da = vals.det.array() * quadrature.weights.array();

					Eigen::MatrixXd u, grad_u, p, grad_p;
					io::Evaluator::interpolate_at_local_vals(e, dim, dim, vals, x, u, grad_u);
					io::Evaluator::interpolate_at_local_vals(e, dim, dim, vals, adjoint, p, grad_p);

					for (int q = 0; q < local_storage.da.size(); ++q)
					{
						Eigen::MatrixXd grad_p_i, grad_u_i;
						utils::vector2matrix(grad_p.row(q), grad_p_i);
						utils::vector2matrix(grad_u.row(q), grad_u_i);

						Eigen::MatrixXd f_prime_dmu, f_prime_dlambda;
						form.assembler_.compute_dstress_dmu_dlambda(
							OptAssemblerData(t, form.dt_, e, quadrature.points.row(q), vals.val.row(q), grad_u_i),
							f_prime_dmu, f_prime_dlambda);

						// This needs to be a sum over material parameter basis.
						local_storage.vec(e + n_elements) += -utils::matrix_inner_product<double>(f_prime_dmu, grad_p_i) * local_storage.da(q);
						local_storage.vec(e) += -utils::matrix_inner_product<double>(f_prime_dlambda, grad_p_i) * local_storage.da(q);
					}
				}
			});

			for (const LocalThreadVecStorage &local_storage : storage)
				term += local_storage.vec;
		}
	}

	void ElasticForceDerivative::force_shape_derivative(
		ElasticForm &form,
		const double t,
		const int n_verts,
		const Eigen::MatrixXd &x,
		const Eigen::MatrixXd &x_prev,
		const Eigen::MatrixXd &adjoint,
		Eigen::VectorXd &term)
	{
		const int dim = form.is_volume_ ? 3 : 2;
		const int actual_dim = ((form.assembler_.name() == "Laplacian") || (form.assembler_.name() == "Electrostatics")) ? 1 : dim;

		const int n_elements = int(form.bases_.size());
		term.setZero(n_verts * dim, 1);

		auto storage = utils::create_thread_storage(LocalThreadVecStorage(term.size()));

		if (form.assembler_.name() == "ViscousDamping")
		{
			utils::maybe_parallel_for(n_elements, [&](int start, int end, int thread_id) {
				LocalThreadVecStorage &local_storage = utils::get_local_thread_storage(storage, thread_id);

				for (int e = start; e < end; ++e)
				{
					assembler::ElementAssemblyValues &vals = local_storage.vals;
					form.ass_vals_cache_.compute(e, form.is_volume_, form.bases_[e], form.geom_bases_[e], vals);
					assembler::ElementAssemblyValues gvals;
					gvals.compute(e, form.is_volume_, vals.quadrature.points, form.geom_bases_[e], form.geom_bases_[e]);

					const quadrature::Quadrature &quadrature = vals.quadrature;
					local_storage.da = vals.det.array() * quadrature.weights.array();

					Eigen::MatrixXd u, grad_u, prev_u, prev_grad_u, p, grad_p;
					io::Evaluator::interpolate_at_local_vals(e, dim, dim, vals, x, u, grad_u);
					io::Evaluator::interpolate_at_local_vals(e, dim, dim, vals, x_prev, prev_u, prev_grad_u);
					io::Evaluator::interpolate_at_local_vals(e, dim, dim, vals, adjoint, p, grad_p);

					Eigen::MatrixXd grad_u_i, prev_grad_u_i;
					Eigen::MatrixXd grad_v_i;
					Eigen::MatrixXd stress_tensor;
					Eigen::VectorXd f_prime_gradu_gradv, f_prev_prime_prev_gradu_gradv;

					for (int q = 0; q < local_storage.da.size(); ++q)
					{
						utils::vector2matrix(grad_u.row(q), grad_u_i);
						utils::vector2matrix(prev_grad_u.row(q), prev_grad_u_i);

						for (auto &v : gvals.basis_values)
						{
							Eigen::MatrixXd stress_grad, stress_prev_grad;
							form.assembler_.compute_stress_grad(
								OptAssemblerData(t, form.dt_, e, quadrature.points.row(q), vals.val.row(q), grad_u_i),
								prev_grad_u_i, stress_tensor, stress_grad);
							form.assembler_.compute_stress_prev_grad(
								OptAssemblerData(t, form.dt_, e, quadrature.points.row(q), vals.val.row(q), grad_u_i),
								prev_grad_u_i, stress_prev_grad);
							for (int d = 0; d < dim; d++)
							{
								grad_v_i.setZero(dim, dim);
								grad_v_i.row(d) = v.grad_t_m.row(q);

								f_prime_gradu_gradv = stress_grad * utils::flatten(grad_u_i * grad_v_i);
								f_prev_prime_prev_gradu_gradv = stress_prev_grad * utils::flatten(prev_grad_u_i * grad_v_i);

								Eigen::MatrixXd tmp = grad_v_i - grad_v_i.trace() * Eigen::MatrixXd::Identity(dim, dim);
								local_storage.vec(v.global[0].index * dim + d) -= grad_p.row(q).dot(f_prime_gradu_gradv + f_prev_prime_prev_gradu_gradv + utils::flatten(stress_tensor * tmp.transpose())) * local_storage.da(q);
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
					form.ass_vals_cache_.compute(e, form.is_volume_, form.bases_[e], form.geom_bases_[e], vals);
					assembler::ElementAssemblyValues gvals;
					gvals.compute(e, form.is_volume_, vals.quadrature.points, form.geom_bases_[e], form.geom_bases_[e]);

					const quadrature::Quadrature &quadrature = vals.quadrature;
					local_storage.da = vals.det.array() * quadrature.weights.array();

					Eigen::MatrixXd u, grad_u, p, grad_p;
					io::Evaluator::interpolate_at_local_vals(e, dim, actual_dim, vals, x, u, grad_u);
					io::Evaluator::interpolate_at_local_vals(e, dim, actual_dim, vals, adjoint, p, grad_p);

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
							utils::vector2matrix(grad_u.row(q), grad_u_i);
							utils::vector2matrix(grad_p.row(q), grad_p_i);
						}

						for (auto &v : gvals.basis_values)
						{
							for (int d = 0; d < dim; d++)
							{
								Eigen::MatrixXd grad_v_i;
								grad_v_i.setZero(dim, dim);
								grad_v_i.row(d) = v.grad_t_m.row(q);

								Eigen::MatrixXd stress_tensor, f_prime_gradu_gradv;
								form.assembler_.compute_stress_grad_multiply_mat(
									OptAssemblerData(t, form.dt_, e, quadrature.points.row(q), vals.val.row(q), grad_u_i),
									grad_u_i * grad_v_i, stress_tensor, f_prime_gradu_gradv);

								const Eigen::MatrixXd tmp = stress_tensor * grad_v_i.transpose() - grad_v_i.trace() * stress_tensor;
								local_storage.vec(v.global[0].index * dim + d) -= utils::matrix_inner_product<double>(f_prime_gradu_gradv + tmp, grad_p_i) * local_storage.da(q);
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
