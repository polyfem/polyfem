#include "BodyForceDerivative.hpp"

#include <cassert>
#include <vector>

#include <Eigen/Core>
#include <polyfem/assembler/AssemblyValues.hpp>
#include <polyfem/assembler/ElementAssemblyValues.hpp>
#include <polyfem/io/Evaluator.hpp>
#include <polyfem/optimization/AdjointTools.hpp>
#include <polyfem/quadrature/Quadrature.hpp>
#include <polyfem/solver/forms/BodyForm.hpp>
#include <polyfem/utils/BoundarySampler.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/MaybeParallelFor.hpp>
#include <polyfem/utils/Types.hpp>

namespace polyfem::solver
{
	namespace
	{
		class LocalThreadVecStorage
		{
		public:
			Eigen::MatrixXd vec;
			assembler::ElementAssemblyValues vals, gvals;
			QuadratureVector da;

			LocalThreadVecStorage(const int size)
			{
				vec.resize(size, 1);
				vec.setZero();
			}
		};
	} // namespace

	void BodyForceDerivative::force_shape_derivative(
		BodyForm &form,
		const int n_verts,
		const double t,
		const Eigen::MatrixXd &x,
		const Eigen::MatrixXd &adjoint,
		Eigen::VectorXd &term)
	{
		const auto &bases = form.rhs_assembler_.bases();
		const auto &gbases = form.rhs_assembler_.gbases();
		const int dim = form.rhs_assembler_.mesh().dimension();
		const int actual_dim = form.rhs_assembler_.problem().is_scalar() ? 1 : dim;

		const int n_elements = int(bases.size());
		term.setZero(n_verts * dim, 1);

		auto storage = utils::create_thread_storage(LocalThreadVecStorage(term.size()));

		utils::maybe_parallel_for(n_elements, [&](int start, int end, int thread_id) {
			LocalThreadVecStorage &local_storage = utils::get_local_thread_storage(storage, thread_id);

			for (int e = start; e < end; ++e)
			{
				assembler::ElementAssemblyValues &vals = local_storage.vals;
				form.rhs_assembler_.ass_vals_cache().compute(e, form.rhs_assembler_.mesh().is_volume(), bases[e], gbases[e], vals);
				assembler::ElementAssemblyValues &gvals = local_storage.gvals;
				gvals.compute(e, form.rhs_assembler_.mesh().is_volume(), vals.quadrature.points, gbases[e], gbases[e]);

				const quadrature::Quadrature &quadrature = vals.quadrature;
				local_storage.da = vals.det.array() * quadrature.weights.array();

				Eigen::MatrixXd p, grad_p;
				io::Evaluator::interpolate_at_local_vals(e, dim, actual_dim, vals, adjoint, p, grad_p);

				Eigen::MatrixXd rhs_function;
				form.rhs_assembler_.problem().rhs(form.rhs_assembler_.assembler(), vals.val, t, rhs_function);
				rhs_function *= -1;
				for (int q = 0; q < vals.val.rows(); q++)
				{
					const double rho = form.density_(quadrature.points.row(q), vals.val.row(q), t, e);
					rhs_function.row(q) *= rho;
				}

				for (int q = 0; q < local_storage.da.size(); ++q)
				{
					const double value = p.row(q).dot(rhs_function.row(q)) * local_storage.da(q);
					for (auto &v : gvals.basis_values)
					{
						local_storage.vec.block(v.global[0].index * dim, 0, dim, 1) += value * v.grad_t_m.row(q).transpose();
					}
				}
			}
		});

		// Zero entries in p that correspond to nodes lying between dirichlet and neumann surfaces
		// DO NOT PARALLELIZE since they both write to the same location
		Eigen::MatrixXd adjoint_zeroed = adjoint;
		adjoint_zeroed(form.boundary_nodes_, Eigen::all).setZero();

		utils::maybe_parallel_for(form.local_neumann_boundary_.size(), [&](int start, int end, int thread_id) {
			LocalThreadVecStorage &local_storage = utils::get_local_thread_storage(storage, thread_id);

			Eigen::MatrixXd uv, points, normals;
			Eigen::VectorXd weights;
			Eigen::VectorXi global_ids;
			assembler::ElementAssemblyValues vals, gvals;

			for (int lb_id = start; lb_id < end; ++lb_id)
			{
				const auto &lb = form.local_neumann_boundary_[lb_id];
				const int e = lb.element_id();

				for (int i = 0; i < lb.size(); i++)
				{
					const int global_primitive_id = lb.global_primitive_id(i);
					utils::BoundarySampler::boundary_quadrature(lb, form.n_boundary_samples_, form.rhs_assembler_.mesh(), i, false, uv, points, normals, weights);
					global_ids.setConstant(points.rows(), global_primitive_id);

					Eigen::MatrixXd reference_normals = normals;

					vals.compute(e, form.rhs_assembler_.mesh().is_volume(), points, bases[e], gbases[e]);
					gvals.compute(e, form.rhs_assembler_.mesh().is_volume(), points, gbases[e], gbases[e]);

					std::vector<std::vector<Eigen::MatrixXd>> grad_normal;
					for (int n = 0; n < gvals.jac_it.size(); ++n)
					{
						Eigen::MatrixXd trafo = gvals.jac_it[n].inverse();

						Eigen::MatrixXd deform_mat(dim, dim);
						deform_mat.setZero();
						Eigen::MatrixXd jac_mat(dim, gvals.basis_values.size());
						int b_idx = 0;
						for (const auto &b : gvals.basis_values)
						{
							jac_mat.col(b_idx++) = b.grad.row(n);

							for (const auto &g : b.global)
								for (int d = 0; d < dim; ++d)
									deform_mat.row(d) += x(g.index * dim + d) * b.grad.row(n);
						}

						trafo += deform_mat;
						trafo = trafo.inverse();

						Eigen::VectorXd displaced_normal = normals.row(n) * trafo;
						normals.row(n) = displaced_normal / displaced_normal.norm();

						std::vector<Eigen::MatrixXd> grad;
						{
							Eigen::MatrixXd vec = -(jac_mat.transpose() * trafo * reference_normals.row(n).transpose());
							// Gradient of the displaced normal computation
							for (int k = 0; k < dim; ++k)
							{
								Eigen::MatrixXd grad_i(jac_mat.rows(), jac_mat.cols());
								grad_i.setZero();
								for (int m = 0; m < jac_mat.rows(); ++m)
									for (int l = 0; l < jac_mat.cols(); ++l)
										grad_i(m, l) = -(reference_normals.row(n) * trafo)(m) * (jac_mat.transpose() * trafo)(l, k);
								grad.push_back(grad_i);
							}
						}

						{
							Eigen::MatrixXd normalization_chain_rule = (normals.row(n).transpose() * normals.row(n));
							normalization_chain_rule = Eigen::MatrixXd::Identity(dim, dim) - normalization_chain_rule;
							normalization_chain_rule /= displaced_normal.norm();

							Eigen::VectorXd vec(dim);
							b_idx = 0;
							for (const auto &b : gvals.basis_values)
							{
								for (int d = 0; d < dim; ++d)
								{
									for (int k = 0; k < dim; ++k)
										vec(k) = grad[k](d, b_idx);
									vec = normalization_chain_rule * vec;
									for (int k = 0; k < dim; ++k)
										grad[k](d, b_idx) = vec(k);
								}
								b_idx++;
							}
						}

						grad_normal.push_back(grad);
					}

					Eigen::MatrixXd neumann_val;
					form.rhs_assembler_.problem().neumann_bc(form.rhs_assembler_.mesh(), global_ids, uv, vals.val, normals, t, neumann_val);

					Eigen::MatrixXd p, grad_p;
					io::Evaluator::interpolate_at_local_vals(e, dim, actual_dim, vals, adjoint_zeroed, p, grad_p);

					const Eigen::VectorXi geom_nodes = gbases[e].local_nodes_for_primitive(global_primitive_id, form.rhs_assembler_.mesh());

					if (geom_nodes.size() != dim)
						log_and_throw_error("Only linear geometry is supported in shape derivative!");

					Eigen::VectorXd value = (p.array() * neumann_val.array()).rowwise().sum() * weights.array();

					Eigen::VectorXd pressure_bc;
					pressure_bc = (neumann_val.array() * normals.array()).rowwise().sum();

					for (long n = 0; n < geom_nodes.size(); ++n)
					{
						const assembler::AssemblyValues &v = gvals.basis_values[geom_nodes(n)];

						Eigen::MatrixXd velocity_div_mat;
						{
							if (form.rhs_assembler_.mesh().is_volume())
							{
								Eigen::MatrixXd V(3, 3);
								V.row(0) = gbases[e].bases[geom_nodes(0)].global()[0].node;
								V.row(1) = gbases[e].bases[geom_nodes(1)].global()[0].node;
								V.row(2) = gbases[e].bases[geom_nodes(2)].global()[0].node;

								velocity_div_mat = AdjointTools::face_velocity_divergence(V);
							}
							else
							{
								Eigen::MatrixXd V(2, 2);
								V.row(0) = gbases[e].bases[geom_nodes(0)].global()[0].node;
								V.row(1) = gbases[e].bases[geom_nodes(1)].global()[0].node;

								velocity_div_mat = AdjointTools::edge_velocity_divergence(V);
							}
						}

						// integrate j * div(gbases) over the whole boundary
						for (int d = 0; d < dim; d++)
						{
							assert(v.global.size() == 1);
							const int g_index = v.global[0].index * dim + d;

							Eigen::MatrixXd gvals(v.val.size(), dim);
							gvals.setZero();
							gvals.col(d) = v.val;

							local_storage.vec(g_index) += value.sum() * velocity_div_mat(n, d);
							const bool is_pressure = form.rhs_assembler_.problem().is_boundary_pressure(form.rhs_assembler_.mesh().get_boundary_id(global_primitive_id));
							// if (is_pressure)
							// {
							// 	for (int q = 0; q < vals.jac_it.size(); ++q)
							// 	{
							// 		Eigen::MatrixXd grad_x_f(dim, dim);
							// 		for (int ki = 0; ki < dim; ++ki)
							// 			for (int kj = 0; kj < dim; ++kj)
							// 				grad_x_f(ki, kj) = grad_normal[q][ki](kj, geom_nodes(n));
							// 		local_storage.vec(g_index) += pressure_bc(q) * (grad_x_f * gvals.row(q).transpose()).dot(p.row(q)) * weights(q);
							// 	}
							// }
						}
					}
				}
			}
		});

		for (const LocalThreadVecStorage &local_storage : storage)
			term += local_storage.vec;

		term *= -1;

		if (actual_dim == dim)
		{
			StiffnessMatrix hess;
			form.hessian_wrt_u_prev(x, t, hess);
			if (hess.cols() == term.size())
				term += hess.transpose() * adjoint_zeroed;
			// TODO: fix me for P2 basis
		}
	}
} // namespace polyfem::solver
