#include "AdjointTools.hpp"

#include <polyfem/utils/MaybeParallelFor.hpp>
#include <polyfem/utils/Timer.hpp>
#include <polyfem/io/Evaluator.hpp>
#include <polyfem/utils/AutodiffTypes.hpp>
#include <polyfem/State.hpp>

#include <polyfem/utils/IntegrableFunctional.hpp>
#include <polyfem/utils/BoundarySampler.hpp>

#include <polyfem/solver/forms/ElasticForm.hpp>
#include <polyfem/solver/forms/ContactForm.hpp>
#include <polyfem/solver/forms/FrictionForm.hpp>
#include <polyfem/solver/forms/BodyForm.hpp>
#include <polyfem/solver/forms/InertiaForm.hpp>

#include <polyfem/solver/NLProblem.hpp>
#include <polyfem/solver/forms/parametrization/SDFParametrizations.hpp>

#include <polyfem/time_integrator/BDF.hpp>

/*
Reminders:

	1. Due to Dirichlet boundary, any force vector at dirichlet indices should be zero, so \partial_q h and \partial_u h should be set zero at dirichlet rows.

*/

using namespace polyfem::utils;

namespace
{

	int get_bdf_order(const polyfem::State &state)
	{
		if (state.args["time"]["integrator"].is_string())
			return 1;
		if (state.args["time"]["integrator"]["type"] == "ImplicitEuler")
			return 1;
		if (state.args["time"]["integrator"]["type"] == "BDF")
			return state.args["time"]["integrator"]["steps"].get<int>();

		polyfem::log_and_throw_error("Integrator type not supported for differentiability.");
		return -1;
	}
} // namespace

namespace polyfem::solver
{
	namespace
	{
		double dot(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B) { return (A.array() * B.array()).sum(); }

		class LocalThreadScalarStorage
		{
		public:
			double val;
			assembler::ElementAssemblyValues vals;
			QuadratureVector da;

			LocalThreadScalarStorage()
			{
				val = 0;
			}
		};

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

		typedef DScalar1<double, Eigen::Matrix<double, Eigen::Dynamic, 1>> Diff;

		template <typename T>
		T triangle_area(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &V)
		{
			Eigen::Matrix<T, Eigen::Dynamic, 1> l1 = V.row(1) - V.row(0);
			Eigen::Matrix<T, Eigen::Dynamic, 1> l2 = V.row(2) - V.row(0);
			T area = 0.5 * sqrt(pow(l1(1) * l2(2) - l1(2) * l2(1), 2) + pow(l1(0) * l2(2) - l1(2) * l2(0), 2) + pow(l1(1) * l2(0) - l1(0) * l2(1), 2));
			return area;
		}

		Eigen::MatrixXd triangle_area_grad(const Eigen::MatrixXd &F)
		{
			DiffScalarBase::setVariableCount(F.size());
			Eigen::Matrix<Diff, Eigen::Dynamic, Eigen::Dynamic> full_diff(F.rows(), F.cols());
			for (int i = 0; i < F.rows(); i++)
				for (int j = 0; j < F.cols(); j++)
					full_diff(i, j) = Diff(i + j * F.rows(), F(i, j));
			auto reduced_diff = triangle_area(full_diff);

			Eigen::MatrixXd grad(F.rows(), F.cols());
			for (int i = 0; i < F.rows(); ++i)
				for (int j = 0; j < F.cols(); ++j)
					grad(i, j) = reduced_diff.getGradient()(i + j * F.rows());

			return grad;
		}

		template <typename T>
		T line_length(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &V)
		{
			Eigen::Matrix<T, Eigen::Dynamic, 1> L = V.row(1) - V.row(0);
			T area = L.norm();
			return area;
		}

		Eigen::MatrixXd line_length_grad(const Eigen::MatrixXd &F)
		{
			DiffScalarBase::setVariableCount(F.size());
			Eigen::Matrix<Diff, Eigen::Dynamic, Eigen::Dynamic> full_diff(F.rows(), F.cols());
			for (int i = 0; i < F.rows(); i++)
				for (int j = 0; j < F.cols(); j++)
					full_diff(i, j) = Diff(i + j * F.rows(), F(i, j));
			auto reduced_diff = line_length(full_diff);

			Eigen::MatrixXd grad(F.rows(), F.cols());
			for (int i = 0; i < F.rows(); ++i)
				for (int j = 0; j < F.cols(); ++j)
					grad(i, j) = reduced_diff.getGradient()(i + j * F.rows());

			return grad;
		}

		template <typename T>
		Eigen::Matrix<T, 2, 1> edge_normal(const Eigen::Matrix<T, 4, 1> &V)
		{
			Eigen::Matrix<T, 2, 1> v1 = V.segment(0, 2);
			Eigen::Matrix<T, 2, 1> v2 = V.segment(2, 2);
			Eigen::Matrix<T, 2, 1> normal = v1 - v2;
			normal(0) *= -1;
			normal = normal / normal.norm();
			return normal;
		}

		template <typename T>
		Eigen::Matrix<T, 3, 1> face_normal(const Eigen::Matrix<T, 9, 1> &V)
		{
			Eigen::Matrix<T, 3, 1> v1 = V.segment(0, 3);
			Eigen::Matrix<T, 3, 1> v2 = V.segment(3, 3);
			Eigen::Matrix<T, 3, 1> v3 = V.segment(6, 3);
			Eigen::Matrix<T, 3, 1> normal = (v2 - v1).cross(v3 - v1);
			normal = normal / normal.norm();
			return normal;
		}
	} // namespace

	void AdjointTools::dJ_macro_strain_adjoint_term(
		const State &state,
		const Eigen::MatrixXd &sol,
		const Eigen::MatrixXd &adjoint,
		Eigen::VectorXd &one_form)
	{
		const int dim = state.mesh->dimension();
		const auto &bases = state.bases;
		const auto &gbases = state.geom_bases();

		one_form.setZero(dim * dim);
		auto storage = utils::create_thread_storage(LocalThreadVecStorage(one_form.size()));
		utils::maybe_parallel_for(bases.size(), [&](int start, int end, int thread_id) {
			LocalThreadVecStorage &local_storage = utils::get_local_thread_storage(storage, thread_id);
			Eigen::MatrixXd stiffnesses;
			Eigen::MatrixXd p, grad_p;
			for (int e = start; e < end; ++e)
			{
				assembler::ElementAssemblyValues &vals = local_storage.vals;
				state.ass_vals_cache.compute(e, dim == 3, bases[e], gbases[e], vals);

				const quadrature::Quadrature &quadrature = vals.quadrature;
				local_storage.da = vals.det.array() * quadrature.weights.array();

				state.assembler->compute_stiffness_value(vals, quadrature.points, sol, stiffnesses);
				stiffnesses.array().colwise() *= local_storage.da.array();

				io::Evaluator::interpolate_at_local_vals(e, dim, dim, vals, adjoint, p, grad_p);

				for (int a = 0; a < dim; a++)
					for (int b = 0; b < dim; b++)
					{
						int X = a * dim + b;
						local_storage.vec(X) -= dot(stiffnesses.block(0, X * dim * dim, local_storage.da.size(), dim * dim), grad_p);
					}
			}
		});

		for (const LocalThreadVecStorage &local_storage : storage)
			one_form += local_storage.vec;
	}

	double AdjointTools::integrate_objective(
		const State &state,
		const IntegrableFunctional &j,
		const Eigen::MatrixXd &solution,
		const std::set<int> &interested_ids, // either body id or surface id
		const SpatialIntegralType spatial_integral_type,
		const int cur_step) // current time step
	{
		const auto &bases = state.bases;
		const auto &gbases = state.geom_bases();

		const int dim = state.mesh->dimension();
		const int actual_dim = state.problem->is_scalar() ? 1 : dim;
		const int n_elements = int(bases.size());
		const double dt = state.problem->is_time_dependent() ? state.args["time"]["dt"].get<double>() : 0.0;

		double integral = 0;
		if (spatial_integral_type == SpatialIntegralType::volume)
		{
			auto storage = utils::create_thread_storage(LocalThreadScalarStorage());
			utils::maybe_parallel_for(n_elements, [&](int start, int end, int thread_id) {
				LocalThreadScalarStorage &local_storage = utils::get_local_thread_storage(storage, thread_id);

				json params = {};
				params["t"] = dt * cur_step;
				params["step"] = cur_step;

				Eigen::MatrixXd u, grad_u;
				Eigen::MatrixXd lambda, mu;
				Eigen::MatrixXd result;

				for (int e = start; e < end; ++e)
				{
					if (interested_ids.size() != 0 && interested_ids.find(state.mesh->get_body_id(e)) == interested_ids.end())
						continue;

					assembler::ElementAssemblyValues &vals = local_storage.vals;
					state.ass_vals_cache.compute(e, state.mesh->is_volume(), bases[e], gbases[e], vals);
					io::Evaluator::interpolate_at_local_vals(e, dim, actual_dim, vals, solution, u, grad_u);

					const quadrature::Quadrature &quadrature = vals.quadrature;
					local_storage.da = vals.det.array() * quadrature.weights.array();

					params["elem"] = e;
					params["body_id"] = state.mesh->get_body_id(e);
					j.evaluate(state.assembler->parameters(), quadrature.points, vals.val, u, grad_u, Eigen::MatrixXd::Zero(0, 0) /*Not used*/, vals, params, result);

					local_storage.val += dot(result, local_storage.da);
				}
			});
			for (const LocalThreadScalarStorage &local_storage : storage)
				integral += local_storage.val;
		}
		else if (spatial_integral_type == SpatialIntegralType::surface)
		{
			auto storage = utils::create_thread_storage(LocalThreadScalarStorage());
			utils::maybe_parallel_for(state.total_local_boundary.size(), [&](int start, int end, int thread_id) {
				LocalThreadScalarStorage &local_storage = utils::get_local_thread_storage(storage, thread_id);

				Eigen::MatrixXd uv, samples, gtmp;
				Eigen::MatrixXd points, normal;
				Eigen::VectorXd weights;

				Eigen::MatrixXd u, grad_u;
				Eigen::MatrixXd lambda, mu;
				Eigen::MatrixXd result;
				json params = {};
				params["t"] = dt * cur_step;
				params["step"] = cur_step;

				for (int lb_id = start; lb_id < end; ++lb_id)
				{
					const auto &lb = state.total_local_boundary[lb_id];
					const int e = lb.element_id();

					for (int i = 0; i < lb.size(); i++)
					{
						const int global_primitive_id = lb.global_primitive_id(i);
						if (interested_ids.size() != 0 && interested_ids.find(state.mesh->get_boundary_id(global_primitive_id)) == interested_ids.end())
							continue;

						utils::BoundarySampler::boundary_quadrature(lb, state.n_boundary_samples(), *state.mesh, i, false, uv, points, normal, weights);

						assembler::ElementAssemblyValues &vals = local_storage.vals;
						vals.compute(e, state.mesh->is_volume(), points, bases[e], gbases[e]);
						io::Evaluator::interpolate_at_local_vals(e, dim, actual_dim, vals, solution, u, grad_u);

						// normal = normal * vals.jac_it[0]; // assuming linear geometry

						params["elem"] = e;
						params["body_id"] = state.mesh->get_body_id(e);
						params["boundary_id"] = state.mesh->get_boundary_id(global_primitive_id);
						j.evaluate(state.assembler->parameters(), points, vals.val, u, grad_u, normal, vals, params, result);

						local_storage.val += dot(result, weights);
					}
				}
			});
			for (const LocalThreadScalarStorage &local_storage : storage)
				integral += local_storage.val;
		}
		else if (spatial_integral_type == SpatialIntegralType::vertex_sum)
		{
			std::vector<bool> traversed(state.n_bases, false);
			json params = {};
			params["t"] = dt * cur_step;
			params["step"] = cur_step;
			for (int e = 0; e < bases.size(); e++)
			{
				const auto &bs = bases[e];
				for (int i = 0; i < bs.bases.size(); i++)
				{
					const auto &b = bs.bases[i];
					assert(b.global().size() == 1);
					const auto &g = b.global()[0];
					if (traversed[g.index])
						continue;

					params["node"] = g.index;
					params["elem"] = e;
					params["body_id"] = state.mesh->get_body_id(e);
					params["boundary_id"] = -1;
					Eigen::MatrixXd val;
					j.evaluate(state.assembler->parameters(), Eigen::MatrixXd::Zero(1, dim) /*Not used*/, g.node, solution.block(g.index * dim, 0, dim, 1).transpose(), Eigen::MatrixXd::Zero(1, dim * actual_dim) /*Not used*/, Eigen::MatrixXd::Zero(0, 0) /*Not used*/, assembler::ElementAssemblyValues(), params, val);
					integral += val(0);
					traversed[g.index] = true;
				}
			}
		}

		return integral;
	}

	void AdjointTools::compute_shape_derivative_functional_term(
		const State &state,
		const Eigen::MatrixXd &solution,
		const IntegrableFunctional &j,
		const std::set<int> &interested_ids, // either body id or surface id
		const SpatialIntegralType spatial_integral_type,
		Eigen::VectorXd &term,
		const int cur_time_step)
	{
		const auto &gbases = state.geom_bases();
		const auto &bases = state.bases;
		const int dim = state.mesh->dimension();
		const int actual_dim = state.problem->is_scalar() ? 1 : dim;

		const int n_elements = int(bases.size());
		term.setZero(state.n_geom_bases * dim, 1);

		auto storage = utils::create_thread_storage(LocalThreadVecStorage(term.size()));

		// Eigen::MatrixXd global_positions;
		// global_positions.setZero(state.n_geom_bases * dim, 1);

		// for (int e = 0; e < gbases.size(); ++e)
		// {
		// 	const auto &gbs = gbases[e].bases;
		// 	const Eigen::MatrixXd pos = gbases[e].nodes();
		// 	for (int i = 0; i < gbs.size(); ++i)
		// 		global_positions.block(gbs[i].global()[0].index * dim, 0, dim, 1) = pos.row(i).transpose();
		// }

		if (spatial_integral_type == SpatialIntegralType::volume)
		{
			utils::maybe_parallel_for(n_elements, [&](int start, int end, int thread_id) {
				LocalThreadVecStorage &local_storage = utils::get_local_thread_storage(storage, thread_id);

				Eigen::MatrixXd u, grad_u, j_val, dj_dgradu, dj_dx, lambda, mu;

				json params = {};
				params["step"] = cur_time_step;

				for (int e = start; e < end; ++e)
				{
					if (interested_ids.size() != 0 && interested_ids.find(state.mesh->get_body_id(e)) == interested_ids.end())
						continue;

					assembler::ElementAssemblyValues &vals = local_storage.vals;
					state.ass_vals_cache.compute(e, state.mesh->is_volume(), bases[e], gbases[e], vals);
					io::Evaluator::interpolate_at_local_vals(e, dim, actual_dim, vals, solution, u, grad_u);

					assembler::ElementAssemblyValues gvals;
					gvals.compute(e, state.mesh->is_volume(), vals.quadrature.points, gbases[e], gbases[e]);

					const quadrature::Quadrature &quadrature = vals.quadrature;
					local_storage.da = vals.det.array() * quadrature.weights.array();

					params["elem"] = e;
					params["body_id"] = state.mesh->get_body_id(e);

					j.evaluate(state.assembler->parameters(), quadrature.points, vals.val, u, grad_u, Eigen::MatrixXd::Zero(0, 0) /*Not used*/, vals, params, j_val);

					if (j.depend_on_gradu())
						j.dj_dgradu(state.assembler->parameters(), quadrature.points, vals.val, u, grad_u, Eigen::MatrixXd::Zero(0, 0) /*Not used*/, vals, params, dj_dgradu);

					if (j.depend_on_x())
						j.dj_dx(state.assembler->parameters(), quadrature.points, vals.val, u, grad_u, Eigen::MatrixXd::Zero(0, 0) /*Not used*/, vals, params, dj_dx);

					Eigen::MatrixXd tau_q, grad_u_q;
					for (auto &v : gvals.basis_values)
					{
						for (int q = 0; q < local_storage.da.size(); ++q)
						{
							local_storage.vec.block(v.global[0].index * dim, 0, dim, 1) += (j_val(q) * local_storage.da(q)) * v.grad_t_m.row(q).transpose();

							if (j.depend_on_x())
								local_storage.vec.block(v.global[0].index * dim, 0, dim, 1) += (v.val(q) * local_storage.da(q)) * dj_dx.row(q).transpose();

							if (j.depend_on_gradu())
							{
								if (dim == actual_dim) // Elasticity PDE
								{
									vector2matrix(dj_dgradu.row(q), tau_q);
									vector2matrix(grad_u.row(q), grad_u_q);
								}
								else // Laplacian PDE
								{
									tau_q = dj_dgradu.row(q);
									grad_u_q = grad_u.row(q);
								}
								for (int d = 0; d < dim; d++)
									local_storage.vec(v.global[0].index * dim + d) += -dot(tau_q, grad_u_q.col(d) * v.grad_t_m.row(q)) * local_storage.da(q);
							}
						}
					}
				}
			});
		}
		else if (spatial_integral_type == SpatialIntegralType::surface)
		{
			utils::maybe_parallel_for(state.total_local_boundary.size(), [&](int start, int end, int thread_id) {
				LocalThreadVecStorage &local_storage = utils::get_local_thread_storage(storage, thread_id);

				Eigen::MatrixXd uv, points, normal;
				Eigen::VectorXd &weights = local_storage.da;

				Eigen::MatrixXd u, grad_u, x, grad_x, j_val, dj_dgradu, dj_dgradx, dj_dx, lambda, mu;

				json params = {};
				params["step"] = cur_time_step;

				for (int lb_id = start; lb_id < end; ++lb_id)
				{
					const auto &lb = state.total_local_boundary[lb_id];
					const int e = lb.element_id();

					for (int i = 0; i < lb.size(); i++)
					{
						const int global_primitive_id = lb.global_primitive_id(i);
						if (interested_ids.size() != 0 && interested_ids.find(state.mesh->get_boundary_id(global_primitive_id)) == interested_ids.end())
							continue;

						utils::BoundarySampler::boundary_quadrature(lb, state.n_boundary_samples(), *state.mesh, i, false, uv, points, normal, weights);

						assembler::ElementAssemblyValues &vals = local_storage.vals;
						io::Evaluator::interpolate_at_local_vals(*state.mesh, state.problem->is_scalar(), bases, gbases, e, points, solution, u, grad_u);
						// io::Evaluator::interpolate_at_local_vals(*state.mesh, state.problem->is_scalar(), gbases, gbases, e, points, global_positions, x, grad_x);

						vals.compute(e, state.mesh->is_volume(), points, gbases[e], gbases[e]);

						// normal = normal * vals.jac_it[0]; // assuming linear geometry

						const int n_loc_bases_ = int(vals.basis_values.size());

						params["elem"] = e;
						params["body_id"] = state.mesh->get_body_id(e);
						params["boundary_id"] = state.mesh->get_boundary_id(global_primitive_id);

						j.evaluate(state.assembler->parameters(), points, vals.val, u, grad_u, normal, vals, params, j_val);
						j_val = j_val.array().colwise() * weights.array();

						if (j.depend_on_gradu())
						{
							j.dj_dgradu(state.assembler->parameters(), points, vals.val, u, grad_u, normal, vals, params, dj_dgradu);
							dj_dgradu = dj_dgradu.array().colwise() * weights.array();
						}

						if (j.depend_on_gradx())
						{
							j.dj_dgradx(state.assembler->parameters(), points, vals.val, u, grad_u, normal, vals, params, dj_dgradx);
							dj_dgradx = dj_dgradx.array().colwise() * weights.array();
						}

						if (j.depend_on_x())
						{
							j.dj_dx(state.assembler->parameters(), points, vals.val, u, grad_u, normal, vals, params, dj_dx);
							dj_dx = dj_dx.array().colwise() * weights.array();
						}

						const auto nodes = gbases[e].local_nodes_for_primitive(lb.global_primitive_id(i), *state.mesh);

						if (nodes.size() != dim)
							log_and_throw_error("Only linear geometry is supported in differentiable surface integral functional!");

						Eigen::MatrixXd velocity_div_mat;
						if (state.mesh->is_volume())
						{
							Eigen::Matrix3d V;
							for (int d = 0; d < 3; d++)
								V.row(d) = gbases[e].bases[nodes(d)].global()[0].node;
							velocity_div_mat = face_velocity_divergence(V);
						}
						else
						{
							Eigen::Matrix2d V;
							for (int d = 0; d < 2; d++)
								V.row(d) = gbases[e].bases[nodes(d)].global()[0].node;
							velocity_div_mat = edge_velocity_divergence(V);
						}

						Eigen::MatrixXd grad_u_q, tau_q, grad_x_q;
						for (long n = 0; n < nodes.size(); ++n)
						{
							const assembler::AssemblyValues &v = vals.basis_values[nodes(n)];

							local_storage.vec.block(v.global[0].index * dim, 0, dim, 1) += j_val.sum() * velocity_div_mat.row(n).transpose();
						}

						for (long n = 0; n < n_loc_bases_; ++n)
						{
							const assembler::AssemblyValues &v = vals.basis_values[n];

							if (j.depend_on_x())
								local_storage.vec.block(v.global[0].index * dim, 0, dim, 1) += dj_dx.transpose() * v.val;

							// integrate j * div(gbases) over the whole boundary
							if (j.depend_on_gradu())
								for (int q = 0; q < weights.size(); ++q)
								{
									if (dim == actual_dim) // Elasticity PDE
									{
										vector2matrix(grad_u.row(q), grad_u_q);
										vector2matrix(dj_dgradu.row(q), tau_q);
									}
									else // Laplacian PDE
									{
										grad_u_q = grad_u.row(q);
										tau_q = dj_dgradu.row(q);
									}

									for (int d = 0; d < dim; d++)
										local_storage.vec(v.global[0].index * dim + d) += -dot(tau_q, grad_u_q.col(d) * v.grad_t_m.row(q));
								}

							if (j.depend_on_gradx())
							{
								for (int d = 0; d < dim; d++)
								{
									for (int q = 0; q < weights.size(); ++q)
										local_storage.vec(v.global[0].index * dim + d) += dot(dj_dgradx.block(q, d * dim, 1, dim), v.grad.row(q));
								}
							}
						}
					}
				}
			});
		}
		else if (spatial_integral_type == SpatialIntegralType::vertex_sum)
		{
			log_and_throw_error("Shape derivative of vertex sum type functional is not implemented!");
		}
		for (const LocalThreadVecStorage &local_storage : storage)
			term += local_storage.vec;

		term = utils::flatten(utils::unflatten(term, dim)(state.primitive_to_node(), Eigen::all));
	}

	void AdjointTools::compute_macro_strain_derivative_functional_term(
		const State &state,
		const Eigen::MatrixXd &solution,
		const IntegrableFunctional &j,
		const std::set<int> &interested_ids, // either body id or surface id
		const SpatialIntegralType spatial_integral_type,
		Eigen::VectorXd &term,
		const int cur_time_step)
	{
		const auto &gbases = state.geom_bases();
		const auto &bases = state.bases;
		const int dim = state.mesh->dimension();
		const int actual_dim = state.problem->is_scalar() ? 1 : dim;

		const int n_elements = int(bases.size());
		term.setZero(dim * dim, 1);

		if (!j.depend_on_gradu())
			return;

		auto storage = utils::create_thread_storage(LocalThreadVecStorage(term.size()));

		if (spatial_integral_type == SpatialIntegralType::volume)
		{
			utils::maybe_parallel_for(n_elements, [&](int start, int end, int thread_id) {
				LocalThreadVecStorage &local_storage = utils::get_local_thread_storage(storage, thread_id);

				Eigen::MatrixXd u, grad_u, dj_du;

				json params = {};
				params["step"] = cur_time_step;

				for (int e = start; e < end; ++e)
				{
					if (interested_ids.size() != 0 && interested_ids.find(state.mesh->get_body_id(e)) == interested_ids.end())
						continue;

					assembler::ElementAssemblyValues &vals = local_storage.vals;
					state.ass_vals_cache.compute(e, state.mesh->is_volume(), bases[e], gbases[e], vals);
					io::Evaluator::interpolate_at_local_vals(e, dim, actual_dim, vals, solution, u, grad_u);

					const quadrature::Quadrature &quadrature = vals.quadrature;
					local_storage.da = vals.det.array() * quadrature.weights.array();

					params["elem"] = e;
					params["body_id"] = state.mesh->get_body_id(e);

					j.dj_dgradu(state.assembler->parameters(), quadrature.points, vals.val, u, grad_u, Eigen::MatrixXd::Zero(0, 0) /*Not used*/, vals, params, dj_du);

					local_storage.vec += dj_du.transpose() * local_storage.da;
				}
			});
		}
		else
			log_and_throw_error("Not implemented!");

		for (const LocalThreadVecStorage &local_storage : storage)
			term += local_storage.vec;
	}

	void AdjointTools::dJ_shape_static_adjoint_term(
		const State &state,
		const Eigen::MatrixXd &sol,
		const Eigen::MatrixXd &adjoint,
		Eigen::VectorXd &one_form)
	{
		Eigen::VectorXd elasticity_term, rhs_term, contact_term;

		one_form.setZero(state.n_geom_bases * state.mesh->dimension());

		// if (j.depend_on_u() || j.depend_on_gradu())
		{
			state.solve_data.elastic_form->force_shape_derivative(state.n_geom_bases, sol, sol, adjoint, elasticity_term);
			if (state.solve_data.body_form)
				state.solve_data.body_form->force_shape_derivative(state.n_geom_bases, 0, sol, adjoint, rhs_term);
			else
				rhs_term.setZero(one_form.size());

			if (state.is_contact_enabled())
			{
				state.solve_data.contact_form->force_shape_derivative(state.diff_cached.contact_set(0), sol, adjoint, contact_term);
				contact_term = state.gbasis_nodes_to_basis_nodes * contact_term;
			}
			else
				contact_term.setZero(elasticity_term.size());
			one_form -= elasticity_term + rhs_term + contact_term;
		}

		one_form = utils::flatten(utils::unflatten(one_form, state.mesh->dimension())(state.primitive_to_node(), Eigen::all));
	}

	void AdjointTools::dJ_shape_transient_adjoint_term(
		const State &state,
		const Eigen::MatrixXd &adjoint_nu,
		const Eigen::MatrixXd &adjoint_p,
		Eigen::VectorXd &one_form)
	{
		const double t0 = state.args["time"]["t0"];
		const double dt = state.args["time"]["dt"];
		const int time_steps = state.args["time"]["time_steps"];
		const int bdf_order = get_bdf_order(state);

		Eigen::VectorXd elasticity_term, rhs_term, damping_term, mass_term, contact_term, friction_term;
		one_form.setZero(state.n_geom_bases * state.mesh->dimension());

		Eigen::VectorXd cur_p, cur_nu;
		for (int i = time_steps; i > 0; --i)
		{
			const int real_order = std::min(bdf_order, i);
			double beta = time_integrator::BDF::betas(real_order - 1);
			double beta_dt = beta * dt;

			Eigen::MatrixXd velocity = state.diff_cached.v(i);

			cur_p = adjoint_p.col(i);
			cur_nu = adjoint_nu.col(i);
			cur_p(state.boundary_nodes).setZero();
			cur_nu(state.boundary_nodes).setZero();

			{
				state.solve_data.inertia_form->force_shape_derivative(state.mesh->is_volume(), state.n_geom_bases, state.bases, state.geom_bases(), *(state.mass_matrix_assembler), state.mass_ass_vals_cache, velocity, cur_nu, mass_term);
				state.solve_data.elastic_form->force_shape_derivative(state.n_geom_bases, state.diff_cached.u(i), state.diff_cached.u(i), cur_p, elasticity_term);
				state.solve_data.body_form->force_shape_derivative(state.n_geom_bases, t0 + i * dt, state.diff_cached.u(i - 1), cur_p, rhs_term);

				if (state.solve_data.damping_form)
					state.solve_data.damping_form->force_shape_derivative(state.n_geom_bases, state.diff_cached.u(i), state.diff_cached.u(i - 1), cur_p, damping_term);
				else
					damping_term.setZero(mass_term.size());

				if (state.is_contact_enabled())
				{
					state.solve_data.contact_form->force_shape_derivative(state.diff_cached.contact_set(i), state.diff_cached.u(i), cur_p, contact_term);
					contact_term = state.gbasis_nodes_to_basis_nodes * contact_term;
					// contact_term /= beta_dt * beta_dt;
				}
				else
					contact_term.setZero(mass_term.size());

				if (state.solve_data.friction_form)
				{
					state.solve_data.friction_form->force_shape_derivative(state.diff_cached.u(i - 1), state.diff_cached.u(i), cur_p, state.diff_cached.friction_constraint_set(i), friction_term);
					friction_term = state.gbasis_nodes_to_basis_nodes * (friction_term / beta);
					// friction_term /= beta_dt * beta_dt;
				}
				else
					friction_term.setZero(mass_term.size());
			}

			one_form += beta_dt * (elasticity_term + rhs_term + damping_term + contact_term + friction_term + mass_term);
		}

		// time step 0
		Eigen::VectorXd sum_alpha_p;
		{
			sum_alpha_p.setZero(adjoint_p.rows());
			int num = std::min(bdf_order, time_steps);
			for (int j = 0; j < num; ++j)
			{
				int order = std::min(bdf_order - 1, j);
				sum_alpha_p -= time_integrator::BDF::alphas(order)[j] * adjoint_p.col(j + 1);
			}
		}
		sum_alpha_p(state.boundary_nodes).setZero();
		state.solve_data.inertia_form->force_shape_derivative(state.mesh->is_volume(), state.n_geom_bases, state.bases, state.geom_bases(), *(state.mass_matrix_assembler), state.mass_ass_vals_cache, state.diff_cached.v(0), sum_alpha_p, mass_term);

		one_form += mass_term;

		one_form = utils::flatten(utils::unflatten(one_form, state.mesh->dimension())(state.primitive_to_node(), Eigen::all));
	}

	void AdjointTools::dJ_material_static_adjoint_term(
		const State &state,
		const Eigen::MatrixXd &sol,
		const Eigen::MatrixXd &adjoint,
		Eigen::VectorXd &one_form)
	{
		state.solve_data.elastic_form->force_material_derivative(sol, sol, adjoint, one_form);
	}

	void AdjointTools::dJ_material_transient_adjoint_term(
		const State &state,
		const Eigen::MatrixXd &adjoint_nu,
		const Eigen::MatrixXd &adjoint_p,
		Eigen::VectorXd &one_form)
	{
		const double dt = state.args["time"]["dt"];
		const int time_steps = state.args["time"]["time_steps"];
		const int bdf_order = get_bdf_order(state);

		one_form.setZero(state.bases.size() * 2);

		auto storage = utils::create_thread_storage(LocalThreadVecStorage(one_form.size()));

		utils::maybe_parallel_for(time_steps, [&](int start, int end, int thread_id) {
			LocalThreadVecStorage &local_storage = utils::get_local_thread_storage(storage, thread_id);
			Eigen::VectorXd elasticity_term;
			for (int i_aux = start; i_aux < end; ++i_aux)
			{
				const int i = time_steps - i_aux;
				const int real_order = std::min(bdf_order, i);
				double beta_dt = time_integrator::BDF::betas(real_order - 1) * dt;

				Eigen::VectorXd cur_p = adjoint_p.col(i);
				cur_p(state.boundary_nodes).setZero();

				state.solve_data.elastic_form->force_material_derivative(state.diff_cached.u(i), state.diff_cached.u(i - 1), -cur_p, elasticity_term);
				local_storage.vec += beta_dt * elasticity_term;
			}
		});

		for (const LocalThreadVecStorage &local_storage : storage)
			one_form += local_storage.vec;
	}

	void AdjointTools::dJ_friction_transient_adjoint_term(
		const State &state,
		const Eigen::MatrixXd &adjoint_nu,
		const Eigen::MatrixXd &adjoint_p,
		Eigen::VectorXd &one_form)
	{
		const double dt = state.args["time"]["dt"];
		const double mu = state.solve_data.friction_form->mu();
		const int time_steps = state.args["time"]["time_steps"];
		const int dim = state.mesh->dimension();
		const int bdf_order = get_bdf_order(state);

		one_form.setZero(1);

		auto storage = utils::create_thread_storage(LocalThreadScalarStorage());

		utils::maybe_parallel_for(time_steps, [&](int start, int end, int thread_id) {
			LocalThreadScalarStorage &local_storage = utils::get_local_thread_storage(storage, thread_id);
			for (int t_aux = start; t_aux < end; ++t_aux)
			{
				const int t = time_steps - t_aux;
				const int real_order = std::min(bdf_order, t);
				double beta = time_integrator::BDF::betas(real_order - 1);

				const Eigen::MatrixXd surface_solution_prev = state.collision_mesh.vertices(utils::unflatten(state.diff_cached.u(t - 1), dim));
				const Eigen::MatrixXd surface_solution = state.collision_mesh.vertices(utils::unflatten(state.diff_cached.u(t), dim));

				// TODO: use the time integration to compute the velocity
				const Eigen::MatrixXd surface_velocities = (surface_solution - surface_solution_prev) / dt;

				Eigen::MatrixXd force = state.collision_mesh.to_full_dof(
					-state.diff_cached.friction_constraint_set(t)
						 .compute_force(
							 state.collision_mesh,
							 state.collision_mesh.rest_positions(),
							 /*lagged_displacements=*/surface_solution_prev,
							 surface_velocities,
							 state.solve_data.contact_form->dhat(),
							 state.solve_data.contact_form->barrier_stiffness(),
							 state.solve_data.friction_form->epsv()));

				Eigen::VectorXd cur_p = adjoint_p.col(t);
				cur_p(state.boundary_nodes).setZero();

				local_storage.val += dot(cur_p, force) / (beta * mu * dt);
			}
		});

		for (const LocalThreadScalarStorage &local_storage : storage)
			one_form(0) += local_storage.val;
	}

	void AdjointTools::dJ_damping_transient_adjoint_term(
		const State &state,
		const Eigen::MatrixXd &adjoint_nu,
		const Eigen::MatrixXd &adjoint_p,
		Eigen::VectorXd &one_form)
	{
		const double dt = state.args["time"]["dt"];
		const int time_steps = state.args["time"]["time_steps"];
		const int bdf_order = get_bdf_order(state);

		one_form.setZero(2);

		auto storage = utils::create_thread_storage(LocalThreadVecStorage(one_form.size()));

		utils::maybe_parallel_for(time_steps, [&](int start, int end, int thread_id) {
			LocalThreadVecStorage &local_storage = utils::get_local_thread_storage(storage, thread_id);
			Eigen::VectorXd damping_term;
			for (int t_aux = start; t_aux < end; ++t_aux)
			{
				const int t = time_steps - t_aux;
				const int real_order = std::min(bdf_order, t);
				const double beta = time_integrator::BDF::betas(real_order - 1);

				Eigen::VectorXd cur_p = adjoint_p.col(t);
				cur_p(state.boundary_nodes).setZero();

				state.solve_data.damping_form->force_material_derivative(state.diff_cached.u(t), state.diff_cached.u(t - 1), -cur_p, damping_term);
				local_storage.vec += (beta * dt) * damping_term;
			}
		});

		for (const LocalThreadVecStorage &local_storage : storage)
			one_form += local_storage.vec;
	}

	void AdjointTools::dJ_initial_condition_adjoint_term(
		const State &state,
		const Eigen::MatrixXd &adjoint_nu,
		const Eigen::MatrixXd &adjoint_p,
		Eigen::VectorXd &one_form)
	{
		const int ndof = state.ndof();
		one_form.setZero(ndof * 2); // half for initial solution, half for initial velocity

		// \partial_q \hat{J}^0 - p_0^T \partial_q g^v - \mu_0^T \partial_q g^u
		one_form.segment(0, ndof) = -adjoint_nu.col(0); // adjoint_nu[0] actually stores adjoint_mu[0]
		one_form.segment(ndof, ndof) = -adjoint_p.col(0);

		for (int b : state.boundary_nodes)
		{
			one_form(b) = 0;
			one_form(ndof + b) = 0;
		}
	}

	void AdjointTools::dJ_dirichlet_transient_adjoint_term(
		const State &state,
		const Eigen::MatrixXd &adjoint_nu,
		const Eigen::MatrixXd &adjoint_p,
		Eigen::VectorXd &one_form)
	{
		const double dt = state.args["time"]["dt"];
		const int time_steps = state.args["time"]["time_steps"];
		const int bdf_order = get_bdf_order(state);
		const int n_dirichlet_dof = state.boundary_nodes.size();

		one_form.setZero(time_steps * n_dirichlet_dof);
		for (int i = 1; i <= time_steps; ++i)
		{
			const int real_order = std::min(bdf_order, i);
			const double beta_dt = time_integrator::BDF::betas(real_order - 1) * dt;

			one_form.segment((i - 1) * n_dirichlet_dof, n_dirichlet_dof) = -(1. / beta_dt) * adjoint_p(state.boundary_nodes, i);
		}
	}

	void AdjointTools::dJ_du_step(
		const State &state,
		const IntegrableFunctional &j,
		const Eigen::MatrixXd &solution,
		const std::set<int> &interested_ids,
		const SpatialIntegralType spatial_integral_type,
		const int cur_step,
		Eigen::VectorXd &term)
	{
		const auto &bases = state.bases;
		const auto &gbases = state.geom_bases();

		const int dim = state.mesh->dimension();
		const int actual_dim = state.problem->is_scalar() ? 1 : dim;
		const int n_elements = int(bases.size());
		const double dt = state.problem->is_time_dependent() ? state.args["time"]["dt"].get<double>() : 0.0;

		term = Eigen::MatrixXd::Zero(state.n_bases * actual_dim, 1);

		if (!j.depend_on_u() && !j.depend_on_gradu() && !j.depend_on_gradu_local())
			return;

		if (spatial_integral_type == SpatialIntegralType::volume)
		{
			auto storage = utils::create_thread_storage(LocalThreadVecStorage(term.size()));
			utils::maybe_parallel_for(n_elements, [&](int start, int end, int thread_id) {
				LocalThreadVecStorage &local_storage = utils::get_local_thread_storage(storage, thread_id);

				Eigen::MatrixXd u, grad_u;
				Eigen::MatrixXd lambda, mu;
				Eigen::MatrixXd dj_du, dj_dgradu, dj_dgradx;

				json params = {};
				params["t"] = dt * cur_step;
				params["step"] = cur_step;

				for (int e = start; e < end; ++e)
				{
					if (interested_ids.size() != 0 && interested_ids.find(state.mesh->get_body_id(e)) == interested_ids.end())
						continue;

					assembler::ElementAssemblyValues &vals = local_storage.vals;
					state.ass_vals_cache.compute(e, state.mesh->is_volume(), bases[e], gbases[e], vals);

					const quadrature::Quadrature &quadrature = vals.quadrature;
					local_storage.da = vals.det.array() * quadrature.weights.array();

					const int n_loc_bases_ = int(vals.basis_values.size());

					io::Evaluator::interpolate_at_local_vals(e, dim, actual_dim, vals, solution, u, grad_u);

					params["elem"] = e;
					params["body_id"] = state.mesh->get_body_id(e);

					dj_dgradu.resize(0, 0);
					if (j.depend_on_gradu())
					{
						j.dj_dgradu(state.assembler->parameters(), quadrature.points, vals.val, u, grad_u, Eigen::MatrixXd::Zero(0, 0) /*Not used*/, vals, params, dj_dgradu);
						for (int q = 0; q < dj_dgradu.rows(); q++)
							dj_dgradu.row(q) *= local_storage.da(q);
					}

					dj_du.resize(0, 0);
					if (j.depend_on_u())
					{
						j.dj_du(state.assembler->parameters(), quadrature.points, vals.val, u, grad_u, Eigen::MatrixXd::Zero(0, 0) /*Not used*/, vals, params, dj_du);
						for (int q = 0; q < dj_du.rows(); q++)
							dj_du.row(q) *= local_storage.da(q);
					}

					for (int i = 0; i < n_loc_bases_; ++i)
					{
						const assembler::AssemblyValues &v = vals.basis_values[i];
						assert(v.global.size() == 1);
						for (int d = 0; d < actual_dim; d++)
						{
							double val = 0;

							// j = j(..., grad u)
							if (j.depend_on_gradu())
							{
								for (int q = 0; q < local_storage.da.size(); ++q)
									val += dot(dj_dgradu.block(q, d * dim, 1, dim), v.grad_t_m.row(q));
							}

							// j = j(..., u)
							if (j.depend_on_u())
							{
								for (int q = 0; q < local_storage.da.size(); ++q)
									val += dj_du(q, d) * v.val(q);
							}
							local_storage.vec(v.global[0].index * actual_dim + d) += val;
						}
					}
				}
			});
			for (const LocalThreadVecStorage &local_storage : storage)
				term += local_storage.vec;
		}
		else if (spatial_integral_type == SpatialIntegralType::surface)
		{
			auto storage = utils::create_thread_storage(LocalThreadVecStorage(term.size()));
			utils::maybe_parallel_for(state.total_local_boundary.size(), [&](int start, int end, int thread_id) {
				LocalThreadVecStorage &local_storage = utils::get_local_thread_storage(storage, thread_id);

				Eigen::MatrixXd uv, samples, gtmp;
				Eigen::MatrixXd points, normal;
				Eigen::VectorXd weights;

				Eigen::MatrixXd u, grad_u;
				Eigen::MatrixXd lambda, mu;
				Eigen::MatrixXd dj_du, dj_dgradu, dj_dgradu_local;
				json params = {};
				params["t"] = dt * cur_step;
				params["step"] = cur_step;

				for (int lb_id = start; lb_id < end; ++lb_id)
				{
					const auto &lb = state.total_local_boundary[lb_id];
					const int e = lb.element_id();

					for (int i = 0; i < lb.size(); i++)
					{
						const int global_primitive_id = lb.global_primitive_id(i);
						if (interested_ids.size() != 0 && interested_ids.find(state.mesh->get_boundary_id(global_primitive_id)) == interested_ids.end())
							continue;

						utils::BoundarySampler::boundary_quadrature(lb, state.n_boundary_samples(), *state.mesh, i, false, uv, points, normal, weights);

						assembler::ElementAssemblyValues &vals = local_storage.vals;
						vals.compute(e, state.mesh->is_volume(), points, bases[e], gbases[e]);
						io::Evaluator::interpolate_at_local_vals(e, dim, actual_dim, vals, solution, u, grad_u);

						// normal = normal * vals.jac_it[0]; // assuming linear geometry

						const int n_loc_bases_ = int(vals.basis_values.size());

						params["elem"] = e;
						params["body_id"] = state.mesh->get_body_id(e);
						params["boundary_id"] = state.mesh->get_boundary_id(global_primitive_id);

						dj_dgradu.resize(0, 0);
						if (j.depend_on_gradu())
						{
							j.dj_dgradu(state.assembler->parameters(), points, vals.val, u, grad_u, normal, vals, params, dj_dgradu);
							for (int q = 0; q < dj_dgradu.rows(); q++)
								dj_dgradu.row(q) *= weights(q);
						}

						dj_dgradu_local.resize(0, 0);
						if (j.depend_on_gradu_local())
						{
							j.dj_dgradu_local(state.assembler->parameters(), points, vals.val, u, grad_u, normal, vals, params, dj_dgradu_local);
							for (int q = 0; q < dj_dgradu_local.rows(); q++)
								dj_dgradu_local.row(q) *= weights(q);
						}

						dj_du.resize(0, 0);
						if (j.depend_on_u())
						{
							j.dj_du(state.assembler->parameters(), points, vals.val, u, grad_u, normal, vals, params, dj_du);
							for (int q = 0; q < dj_du.rows(); q++)
								dj_du.row(q) *= weights(q);
						}

						for (int l = 0; l < lb.size(); ++l)
						{
							const auto nodes = bases[e].local_nodes_for_primitive(lb.global_primitive_id(l), *state.mesh);

							for (long n = 0; n < nodes.size(); ++n)
							{
								const assembler::AssemblyValues &v = vals.basis_values[nodes(n)];
								assert(v.global.size() == 1);
								for (int d = 0; d < actual_dim; d++)
								{
									double val = 0;

									// j = j(x, grad u)
									if (j.depend_on_gradu())
									{
										for (int q = 0; q < weights.size(); ++q)
											val += dot(dj_dgradu.block(q, d * dim, 1, dim), v.grad_t_m.row(q));
									}
									// j = j(x, grad u)
									if (j.depend_on_gradu_local())
									{
										for (int q = 0; q < weights.size(); ++q)
											val += dot(dj_dgradu_local.block(q, d * dim, 1, dim), v.grad.row(q));
									}
									// j = j(x, u)
									if (j.depend_on_u())
									{
										for (int q = 0; q < weights.size(); ++q)
											val += dj_du(q, d) * v.val(q);
									}
									local_storage.vec(v.global[0].index * actual_dim + d) += val;
								}
							}
						}
					}
				}
			});
			for (const LocalThreadVecStorage &local_storage : storage)
				term += local_storage.vec;
		}
		else if (spatial_integral_type == SpatialIntegralType::vertex_sum)
		{
			std::vector<bool> traversed(state.n_bases, false);
			json params = {};
			params["t"] = dt * cur_step;
			params["step"] = cur_step;
			for (int e = 0; e < bases.size(); e++)
			{
				const auto &bs = bases[e];
				for (int i = 0; i < bs.bases.size(); i++)
				{
					const auto &b = bs.bases[i];
					assert(b.global().size() == 1);
					const auto &g = b.global()[0];
					if (traversed[g.index])
						continue;

					params["node"] = g.index;
					params["elem"] = e;
					params["body_id"] = state.mesh->get_body_id(e);
					params["boundary_id"] = -1;
					Eigen::MatrixXd val;
					j.dj_du(state.assembler->parameters(), Eigen::MatrixXd::Zero(1, dim) /*Not used*/, g.node, solution.block(g.index * dim, 0, dim, 1).transpose(), Eigen::MatrixXd::Zero(1, dim * actual_dim) /*Not used*/, Eigen::MatrixXd::Zero(0, 0) /*Not used*/, assembler::ElementAssemblyValues(), params, val);
					term.block(g.index * actual_dim, 0, actual_dim, 1) += val.transpose();
					traversed[g.index] = true;
				}
			}
		}
	}

	Eigen::VectorXd AdjointTools::map_primitive_to_node_order(const State &state, const Eigen::VectorXd &primitives)
	{
		int dim = state.mesh->dimension();
		assert(primitives.size() == (state.n_geom_bases * dim));
		Eigen::VectorXd nodes(primitives.size());
		auto map = state.primitive_to_node();
		for (int v = 0; v < state.n_geom_bases; ++v)
			nodes.segment(map[v] * dim, dim) = primitives.segment(v * dim, dim);
		return nodes;
	}

	Eigen::VectorXd AdjointTools::map_node_to_primitive_order(const State &state, const Eigen::VectorXd &nodes)
	{
		int dim = state.mesh->dimension();
		assert(nodes.size() == (state.n_geom_bases * dim));
		Eigen::VectorXd primitives(nodes.size());
		auto map = state.node_to_primitive();
		for (int v = 0; v < state.n_geom_bases; ++v)
			primitives.segment(map[v] * dim, dim) = nodes.segment(v * dim, dim);
		return primitives;
	}

	Eigen::MatrixXd AdjointTools::edge_normal_gradient(const Eigen::MatrixXd &V)
	{
		DiffScalarBase::setVariableCount(4);
		Eigen::Matrix<Diff, 4, 1> full_diff(4, 1);
		for (int i = 0; i < 2; i++)
			for (int j = 0; j < 2; j++)
				full_diff(i * 2 + j) = Diff(i * 2 + j, V(i, j));
		auto reduced_diff = edge_normal(full_diff);

		Eigen::MatrixXd grad(2, 4);
		for (int i = 0; i < 2; ++i)
			grad.row(i) = reduced_diff[i].getGradient();

		return grad;
	}

	Eigen::MatrixXd AdjointTools::face_normal_gradient(const Eigen::MatrixXd &V)
	{
		DiffScalarBase::setVariableCount(9);
		Eigen::Matrix<Diff, 9, 1> full_diff(9, 1);
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++)
				full_diff(i * 3 + j) = Diff(i * 3 + j, V(i, j));
		auto reduced_diff = face_normal(full_diff);

		Eigen::MatrixXd grad(3, 9);
		for (int i = 0; i < 3; ++i)
			grad.row(i) = reduced_diff[i].getGradient();

		return grad;
	}

	Eigen::MatrixXd AdjointTools::edge_velocity_divergence(const Eigen::MatrixXd &V)
	{
		return line_length_grad(V) / line_length<double>(V);
	}

	Eigen::MatrixXd AdjointTools::face_velocity_divergence(const Eigen::MatrixXd &V)
	{
		return triangle_area_grad(V) / triangle_area<double>(V);
	}
} // namespace polyfem::solver