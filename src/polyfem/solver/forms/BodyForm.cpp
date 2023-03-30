#include "BodyForm.hpp"

#include <polyfem/io/Evaluator.hpp>
#include <polyfem/utils/MaybeParallelFor.hpp>
#include <polyfem/utils/BoundarySampler.hpp>

#include <polyfem/basis/ElementBases.hpp>
#include <polyfem/assembler/AssemblerUtils.hpp>
#include <polyfem/assembler/AssemblyValsCache.hpp>

#include <polyfem/autogen/auto_p_bases.hpp>

#include <polyfem/solver/AdjointTools.hpp>

#include <polyfem/utils/Logger.hpp>

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
	BodyForm::BodyForm(const int ndof,
					   const int n_pressure_bases,
					   const std::vector<int> &boundary_nodes,
					   const std::vector<mesh::LocalBoundary> &local_boundary,
					   const std::vector<mesh::LocalBoundary> &local_neumann_boundary,
					   const int n_boundary_samples,
					   const Eigen::MatrixXd &rhs,
					   const assembler::RhsAssembler &rhs_assembler,
					   const assembler::Density &density,
					   const bool apply_DBC,
					   const bool is_formulation_mixed,
					   const bool is_time_dependent)
		: ndof_(ndof),
		  n_pressure_bases_(n_pressure_bases),
		  boundary_nodes_(boundary_nodes),
		  local_boundary_(local_boundary),
		  local_neumann_boundary_(local_neumann_boundary),
		  n_boundary_samples_(n_boundary_samples),
		  rhs_(rhs),
		  rhs_assembler_(rhs_assembler),
		  density_(density),
		  apply_DBC_(apply_DBC),
		  is_formulation_mixed_(is_formulation_mixed)
	{
		t_ = 0;
		if (!is_time_dependent)
			update_current_rhs(Eigen::VectorXd());
	}

	double BodyForm::value_unweighted(const Eigen::VectorXd &x) const
	{
		return rhs_assembler_.compute_energy(x, local_neumann_boundary_, density_, n_boundary_samples_, t_);
	}

	void BodyForm::first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		// REMEMBER -!!!!!
		gradv = -current_rhs_;
	}

	void BodyForm::update_quantities(const double t, const Eigen::VectorXd &x)
	{
		this->t_ = t;
		update_current_rhs(x);
	}

	void BodyForm::update_current_rhs(const Eigen::VectorXd &x)
	{
		rhs_assembler_.compute_energy_grad(
			local_boundary_, boundary_nodes_, density_,
			n_boundary_samples_, local_neumann_boundary_,
			rhs_, t_, current_rhs_);

		if (is_formulation_mixed_ && current_rhs_.size() < ndof_)
		{
			current_rhs_.conservativeResize(
				current_rhs_.rows() + n_pressure_bases_, current_rhs_.cols());
			current_rhs_.bottomRows(n_pressure_bases_).setZero();
		}

		// Apply Neumann boundary conditions
		rhs_assembler_.set_bc(
			std::vector<mesh::LocalBoundary>(), std::vector<int>(),
			n_boundary_samples_, local_neumann_boundary_,
			current_rhs_, x, t_);

		// Apply Dirichlet boundary conditions
		if (apply_DBC_)
			rhs_assembler_.set_bc(
				local_boundary_, boundary_nodes_,
				n_boundary_samples_, std::vector<mesh::LocalBoundary>(),
				current_rhs_, x, t_);
	}

	void BodyForm::force_shape_derivative(const int n_verts, const double t, const Eigen::MatrixXd &x, const Eigen::MatrixXd &adjoint, Eigen::VectorXd &term)
	{
		const auto &bases = rhs_assembler_.bases();
		const auto &gbases = rhs_assembler_.gbases();
		const int dim = rhs_assembler_.mesh().dimension();
		const int actual_dim = rhs_assembler_.problem().is_scalar() ? 1 : dim;

		const int n_elements = int(bases.size());
		term.setZero(n_verts * dim, 1);

		auto storage = utils::create_thread_storage(LocalThreadVecStorage(term.size()));

		utils::maybe_parallel_for(n_elements, [&](int start, int end, int thread_id) {
			LocalThreadVecStorage &local_storage = utils::get_local_thread_storage(storage, thread_id);

			for (int e = start; e < end; ++e)
			{
				assembler::ElementAssemblyValues &vals = local_storage.vals;
				rhs_assembler_.ass_vals_cache().compute(e, rhs_assembler_.mesh().is_volume(), bases[e], gbases[e], vals);
				assembler::ElementAssemblyValues &gvals = local_storage.gvals;
				gvals.compute(e, rhs_assembler_.mesh().is_volume(), vals.quadrature.points, gbases[e], gbases[e]);

				const quadrature::Quadrature &quadrature = vals.quadrature;
				local_storage.da = vals.det.array() * quadrature.weights.array();

				Eigen::MatrixXd p, grad_p;
				io::Evaluator::interpolate_at_local_vals(e, dim, actual_dim, vals, adjoint, p, grad_p);

				Eigen::MatrixXd rhs_function;
				rhs_assembler_.problem().rhs(rhs_assembler_.assembler(), rhs_assembler_.formulation(), vals.val, t, rhs_function);
				rhs_function *= -1;
				for (int q = 0; q < vals.val.rows(); q++)
				{
					const double rho = density_(quadrature.points.row(q), vals.val.row(q), e);
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
		adjoint_zeroed(boundary_nodes_, Eigen::all).setZero();

		utils::maybe_parallel_for(local_neumann_boundary_.size(), [&](int start, int end, int thread_id) {
			LocalThreadVecStorage &local_storage = utils::get_local_thread_storage(storage, thread_id);

			Eigen::MatrixXd uv, points, normals;
			Eigen::VectorXd weights;
			Eigen::VectorXi global_ids;
			assembler::ElementAssemblyValues vals;

			for (int lb_id = start; lb_id < end; ++lb_id)
			{
				const auto &lb = local_neumann_boundary_[lb_id];
				const int e = lb.element_id();

				for (int i = 0; i < lb.size(); i++)
				{
					const int global_primitive_id = lb.global_primitive_id(i);

					utils::BoundarySampler::boundary_quadrature(lb, n_boundary_samples_, rhs_assembler_.mesh(), i, false, uv, points, normals, weights);
					global_ids.setConstant(points.rows(), 1, global_primitive_id);

					vals.compute(e, rhs_assembler_.mesh().is_volume(), points, gbases[e], gbases[e]);

					for (int n = 0; n < vals.jac_it.size(); ++n)
					{
						Eigen::MatrixXd ppp(1, dim);
						ppp = vals.val.row(n);
						Eigen::MatrixXd trafo = vals.jac_it[n];

						if (actual_dim == dim)
						{
							Eigen::MatrixXd deform_mat(dim, dim);
							deform_mat.setZero();
							for (const auto &b : vals.basis_values)
							{
								for (const auto &g : b.global)
								{
									for (int d = 0; d < dim; ++d)
									{
										deform_mat.col(d) += x(g.index * dim + d) * b.grad_t_m.row(n);

										ppp(d) += x(g.index * dim + d) * b.val(n);
									}
								}
							}

							trafo += deform_mat;
						}
						normals.row(n) = normals.row(n) * trafo;
						normals.row(n).normalize();
					}

					Eigen::MatrixXd neumann_val;
					rhs_assembler_.problem().neumann_bc(rhs_assembler_.mesh(), global_ids, uv, vals.val, normals, t, neumann_val);

					Eigen::MatrixXd p, grad_p;
					io::Evaluator::interpolate_at_local_vals(e, dim, actual_dim, vals, adjoint_zeroed, p, grad_p);

					const auto nodes = gbases[e].local_nodes_for_primitive(global_primitive_id, rhs_assembler_.mesh());

					Eigen::MatrixXd node_points;
					if (rhs_assembler_.mesh().is_volume())
					{
						autogen::p_nodes_3d(1, node_points);
						node_points = node_points(nodes, Eigen::all);
					}
					else
					{
						autogen::p_nodes_2d(1, node_points);
						node_points = node_points(nodes, Eigen::all);
					}
					Eigen::MatrixXd u, grad_u;
					io::Evaluator::interpolate_at_local_vals(rhs_assembler_.mesh(), actual_dim, bases, gbases, e, node_points, x, u, grad_u);

					if (nodes.size() != dim)
						log_and_throw_error("Only linear geometry is supported in shape derivative!");

					Eigen::VectorXd value = (p.array() * neumann_val.array()).rowwise().sum() * weights.array();
					Eigen::VectorXd pressure_value = (p.array() * vals.val.array()).rowwise().sum() * weights.array();

					for (long n = 0; n < nodes.size(); ++n)
					{
						const assembler::AssemblyValues &v = vals.basis_values[nodes(n)];

						Eigen::VectorXd grad_pressure_bc;
						Eigen::MatrixXd velocity_div_mat;
						{
							double pressure_bc = neumann_val.row(0).norm();
							if (rhs_assembler_.mesh().is_volume())
							{
								Eigen::MatrixXd V(3, 3);
								V.row(0) = gbases[e].bases[nodes(0)].global()[0].node;
								V.row(1) = gbases[e].bases[nodes(1)].global()[0].node;
								V.row(2) = gbases[e].bases[nodes(2)].global()[0].node;

								velocity_div_mat = AdjointTools::face_velocity_divergence(V);

								for (int k = 0; k < 3; ++k)
									V.row(k) += u.row(k);

								auto grad = AdjointTools::face_normal_gradient(V);
								if (n == 0)
									grad_pressure_bc = grad.block(0, 0, 3, 3).rowwise().sum();
								else if (n == 1)
									grad_pressure_bc = grad.block(0, 3, 3, 3).rowwise().sum();
								else if (n == 2)
									grad_pressure_bc = grad.block(0, 6, 3, 3).rowwise().sum();
								else
									assert(false);
							}
							else
							{
								Eigen::MatrixXd V(2, 2);
								V.row(0) = gbases[e].bases[nodes(0)].global()[0].node;
								V.row(1) = gbases[e].bases[nodes(1)].global()[0].node;

								velocity_div_mat = AdjointTools::edge_velocity_divergence(V);

								for (int k = 0; k < 2; ++k)
									V.row(k) += u.row(k);

								auto grad = AdjointTools::edge_normal_gradient(V);
								if (n == 0)
									grad_pressure_bc = grad.block(0, 0, 2, 2).rowwise().sum();
								else if (n == 1)
									grad_pressure_bc = grad.block(0, 2, 2, 2).rowwise().sum();
								else
									assert(false);
							}
							grad_pressure_bc *= pressure_bc;
						}

						// integrate j * div(gbases) over the whole boundary
						for (int d = 0; d < dim; d++)
						{
							assert(v.global.size() == 1);
							const int g_index = v.global[0].index * dim + d;
							local_storage.vec(g_index) += value.sum() * velocity_div_mat(n, d);
							const bool is_pressure = rhs_assembler_.problem().is_boundary_pressure(rhs_assembler_.mesh().get_boundary_id(global_primitive_id));
							// if (is_pressure)
							// 	local_storage.vec(g_index) += grad_pressure_bc(d) * pressure_value.sum();
						}
					}
				}
			}
		});

		for (const LocalThreadVecStorage &local_storage : storage)
			term += local_storage.vec;
	}
} // namespace polyfem::solver
