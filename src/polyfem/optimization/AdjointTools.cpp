#include "AdjointTools.hpp"

#include <polyfem/utils/MaybeParallelFor.hpp>
#include <polyfem/utils/Timer.hpp>
#include <polyfem/io/Evaluator.hpp>
#include <polyfem/utils/AutodiffTypes.hpp>
#include <polyfem/State.hpp>

#include <polyfem/utils/IntegrableFunctional.hpp>
#include <polyfem/utils/BoundarySampler.hpp>
#include <polyfem/time_integrator/ImplicitTimeIntegrator.hpp>

#include <polyfem/solver/forms/ElasticForm.hpp>
#include <polyfem/solver/forms/ContactForm.hpp>
#include <polyfem/solver/forms/BarrierContactForm.hpp>
#include <polyfem/solver/forms/SmoothContactForm.hpp>
#include <polyfem/solver/forms/PeriodicContactForm.hpp>
#include <polyfem/solver/forms/NormalAdhesionForm.hpp>
#include <polyfem/solver/forms/TangentialAdhesionForm.hpp>
#include <polyfem/solver/forms/FrictionForm.hpp>
#include <polyfem/solver/forms/BodyForm.hpp>
#include <polyfem/solver/forms/PressureForm.hpp>
#include <polyfem/solver/forms/InertiaForm.hpp>
#include <polyfem/optimization/force_derivatives/ElasticForceDerivative.hpp>
#include <polyfem/optimization/force_derivatives/BodyForceDerivative.hpp>
#include <polyfem/optimization/force_derivatives/PressureForceDerivative.hpp>
#include <polyfem/optimization/force_derivatives/InertiaForceDerivative.hpp>
#include <polyfem/optimization/force_derivatives/BarrierContactForceDerivative.hpp>
#include <polyfem/optimization/force_derivatives/SmoothContactForceDerivative.hpp>
#include <polyfem/optimization/force_derivatives/NormalAdhesionForceDerivative.hpp>
#include <polyfem/optimization/force_derivatives/PeriodicContactForceDerivative.hpp>
#include <polyfem/optimization/force_derivatives/FrictionForceDerivative.hpp>
#include <polyfem/optimization/force_derivatives/TangentialAdhesionForceDerivative.hpp>
#include <polyfem/optimization/parametrization/PeriodicMeshToMesh.hpp>

#include <polyfem/solver/NLProblem.hpp>
#include <polyfem/solver/NLHomoProblem.hpp>

#include <polyfem/time_integrator/BDF.hpp>

/*
Reminders:

	1. Due to Dirichlet boundary, any force vector at dirichlet indices should be zero, so \partial_q h and \partial_u h should be set zero at dirichlet rows.

*/

using namespace polyfem::utils;

namespace polyfem::solver
{
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

			polyfem::log_and_throw_adjoint_error("Integrator type not supported for differentiability.");
			return -1;
		}

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

		Eigen::MatrixXd extract_lame_params(const std::map<std::string, Assembler::ParamFunc> &lame_params, const int e, const int t, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts)
		{
			Eigen::MatrixXd params = Eigen::MatrixXd::Zero(local_pts.rows(), 2);

			auto search_lambda = lame_params.find("lambda");
			auto search_mu = lame_params.find("mu");

			if (search_lambda == lame_params.end() || search_mu == lame_params.end())
				return params;

			for (int p = 0; p < local_pts.rows(); p++)
			{
				params(p, 0) = search_lambda->second(local_pts.row(p), pts.row(p), t, e);
				params(p, 1) = search_mu->second(local_pts.row(p), pts.row(p), t, e);
			}

			return params;
		}
	} // namespace

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
		const double t0 = state.problem->is_time_dependent() ? state.args["time"]["t0"].get<double>() : 0.0;
		const double dt = state.problem->is_time_dependent() ? state.args["time"]["dt"].get<double>() : 0.0;

		double integral = 0;
		if (spatial_integral_type == SpatialIntegralType::Volume)
		{
			auto storage = utils::create_thread_storage(LocalThreadScalarStorage());
			utils::maybe_parallel_for(n_elements, [&](int start, int end, int thread_id) {
				LocalThreadScalarStorage &local_storage = utils::get_local_thread_storage(storage, thread_id);

				IntegrableFunctional::ParameterType params;
				params.t = dt * cur_step + t0;
				params.step = cur_step;

				Eigen::MatrixXd u, grad_u;
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

					const Eigen::MatrixXd lame_params = extract_lame_params(state.assembler->parameters(), e, params.t, quadrature.points, vals.val);

					params.elem = e;
					params.body_id = state.mesh->get_body_id(e);
					j.evaluate(lame_params, quadrature.points, vals.val, u, grad_u, Eigen::MatrixXd::Zero(0, 0) /*Not used*/, vals, params, result);

					local_storage.val += dot(result, local_storage.da);
				}
			});
			for (const LocalThreadScalarStorage &local_storage : storage)
				integral += local_storage.val;
		}
		else if (spatial_integral_type == SpatialIntegralType::Surface)
		{
			auto storage = utils::create_thread_storage(LocalThreadScalarStorage());
			utils::maybe_parallel_for(state.total_local_boundary.size(), [&](int start, int end, int thread_id) {
				LocalThreadScalarStorage &local_storage = utils::get_local_thread_storage(storage, thread_id);

				Eigen::MatrixXd uv;
				Eigen::MatrixXd points, normal;
				Eigen::VectorXd weights;

				Eigen::MatrixXd u, grad_u;
				Eigen::MatrixXd result;
				IntegrableFunctional::ParameterType params;
				params.t = dt * cur_step + t0;
				params.step = cur_step;

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

						const Eigen::MatrixXd lame_params = extract_lame_params(state.assembler->parameters(), e, params.t, points, vals.val);

						params.elem = e;
						params.body_id = state.mesh->get_body_id(e);
						params.boundary_id = state.mesh->get_boundary_id(global_primitive_id);
						j.evaluate(lame_params, points, vals.val, u, grad_u, normal, vals, params, result);

						local_storage.val += dot(result, weights);
					}
				}
			});
			for (const LocalThreadScalarStorage &local_storage : storage)
				integral += local_storage.val;
		}
		else if (spatial_integral_type == SpatialIntegralType::VertexSum)
		{
			std::vector<bool> traversed(state.n_bases, false);
			IntegrableFunctional::ParameterType params;
			params.t = dt * cur_step + t0;
			params.step = cur_step;
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

					const Eigen::MatrixXd lame_params = extract_lame_params(state.assembler->parameters(), e, params.t, Eigen::MatrixXd::Zero(1, dim) /*Not used*/, g.node);

					params.node = g.index;
					params.elem = e;
					params.body_id = state.mesh->get_body_id(e);
					Eigen::MatrixXd val;
					j.evaluate(lame_params, Eigen::MatrixXd::Zero(1, dim) /*Not used*/, g.node, solution.block(g.index * dim, 0, dim, 1).transpose(), Eigen::MatrixXd::Zero(1, dim * actual_dim) /*Not used*/, Eigen::MatrixXd::Zero(0, 0) /*Not used*/, assembler::ElementAssemblyValues(), params, val);
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
		const double t0 = state.problem->is_time_dependent() ? state.args["time"]["t0"].get<double>() : 0.0;
		const double dt = state.problem->is_time_dependent() ? state.args["time"]["dt"].get<double>() : 0.0;

		const int n_elements = int(bases.size());
		term.setZero(state.n_geom_bases * dim, 1);

		auto storage = utils::create_thread_storage(LocalThreadVecStorage(term.size()));

		if (spatial_integral_type == SpatialIntegralType::Volume)
		{
			utils::maybe_parallel_for(n_elements, [&](int start, int end, int thread_id) {
				LocalThreadVecStorage &local_storage = utils::get_local_thread_storage(storage, thread_id);

				Eigen::MatrixXd u, grad_u, j_val, dj_dgradu, dj_dx;

				IntegrableFunctional::ParameterType params;
				params.t = cur_time_step * dt + t0;
				params.step = cur_time_step;

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

					const Eigen::MatrixXd lame_params = extract_lame_params(state.assembler->parameters(), e, params.t, quadrature.points, vals.val);

					params.elem = e;
					params.body_id = state.mesh->get_body_id(e);

					j.evaluate(lame_params, quadrature.points, vals.val, u, grad_u, Eigen::MatrixXd::Zero(0, 0) /*Not used*/, vals, params, j_val);

					if (j.depend_on_gradu())
						j.dj_dgradu(lame_params, quadrature.points, vals.val, u, grad_u, Eigen::MatrixXd::Zero(0, 0) /*Not used*/, vals, params, dj_dgradu);

					if (j.depend_on_x())
						j.dj_dx(lame_params, quadrature.points, vals.val, u, grad_u, Eigen::MatrixXd::Zero(0, 0) /*Not used*/, vals, params, dj_dx);

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
		else if (spatial_integral_type == SpatialIntegralType::Surface)
		{
			utils::maybe_parallel_for(state.total_local_boundary.size(), [&](int start, int end, int thread_id) {
				LocalThreadVecStorage &local_storage = utils::get_local_thread_storage(storage, thread_id);

				Eigen::MatrixXd uv, points, normal;
				Eigen::VectorXd &weights = local_storage.da;

				Eigen::MatrixXd u, grad_u, x, grad_x, j_val, dj_dgradu, dj_dgradx, dj_dx;

				IntegrableFunctional::ParameterType params;
				params.t = cur_time_step * dt + t0;
				params.step = cur_time_step;

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

						const Eigen::MatrixXd lame_params = extract_lame_params(state.assembler->parameters(), e, params.t, points, vals.val);

						params.elem = e;
						params.body_id = state.mesh->get_body_id(e);
						params.boundary_id = state.mesh->get_boundary_id(global_primitive_id);

						j.evaluate(lame_params, points, vals.val, u, grad_u, normal, vals, params, j_val);
						j_val = j_val.array().colwise() * weights.array();

						if (j.depend_on_gradu())
						{
							j.dj_dgradu(lame_params, points, vals.val, u, grad_u, normal, vals, params, dj_dgradu);
							dj_dgradu = dj_dgradu.array().colwise() * weights.array();
						}

						if (j.depend_on_gradx())
						{
							j.dj_dgradx(lame_params, points, vals.val, u, grad_u, normal, vals, params, dj_dgradx);
							dj_dgradx = dj_dgradx.array().colwise() * weights.array();
						}

						if (j.depend_on_x())
						{
							j.dj_dx(lame_params, points, vals.val, u, grad_u, normal, vals, params, dj_dx);
							dj_dx = dj_dx.array().colwise() * weights.array();
						}

						const auto nodes = gbases[e].local_nodes_for_primitive(lb.global_primitive_id(i), *state.mesh);

						if (nodes.size() != dim)
							log_and_throw_adjoint_error("Only linear geometry is supported in differentiable surface integral functional!");

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
							{
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
		else if (spatial_integral_type == SpatialIntegralType::VertexSum)
		{
			log_and_throw_adjoint_error("Shape derivative of vertex sum type functional is not implemented!");
		}
		for (const LocalThreadVecStorage &local_storage : storage)
			term += local_storage.vec;

		term = utils::flatten(utils::unflatten(term, dim)(state.primitive_to_node(), Eigen::all));
	}

	void AdjointTools::dJ_shape_static_adjoint_term(
		const State &state,
		const Eigen::MatrixXd &sol,
		const Eigen::MatrixXd &adjoint,
		Eigen::VectorXd &one_form)
	{
		Eigen::VectorXd elasticity_term, rhs_term, pressure_term, contact_term, adhesion_term;

		one_form.setZero(state.n_geom_bases * state.mesh->dimension());
		Eigen::MatrixXd adjoint_zeroed = adjoint;
		adjoint_zeroed(state.boundary_nodes, Eigen::all).setZero();

		// if (j.depend_on_u() || j.depend_on_gradu())
		{
			ElasticForceDerivative::force_shape_derivative(*state.solve_data.elastic_form, 0, state.n_geom_bases, sol, sol, adjoint_zeroed, elasticity_term);
			if (state.solve_data.body_form)
				BodyForceDerivative::force_shape_derivative(*state.solve_data.body_form, state.n_geom_bases, 0, sol, adjoint_zeroed, rhs_term);
			else
				rhs_term.setZero(one_form.size());

			if (state.solve_data.pressure_form)
			{
				PressureForceDerivative::force_shape_derivative(*state.solve_data.pressure_form, state.n_geom_bases, 0, sol, adjoint_zeroed, pressure_term);
				pressure_term = state.basis_nodes_to_gbasis_nodes * pressure_term;
			}
			else
				pressure_term.setZero(one_form.size());

			if (state.is_contact_enabled())
			{
				if (const auto barrier_contact = dynamic_cast<const BarrierContactForm *>(state.solve_data.contact_form.get()))
				{
					BarrierContactForceDerivative::force_shape_derivative(*barrier_contact, state.diff_cached.collision_set(0), sol, adjoint_zeroed, contact_term);
				}
				else if (const auto smooth_contact = dynamic_cast<const SmoothContactForm *>(state.solve_data.contact_form.get()))
				{
					SmoothContactForceDerivative::force_shape_derivative(*smooth_contact, state.diff_cached.smooth_collision_set(0), sol, adjoint_zeroed, contact_term);
				}

				contact_term = state.basis_nodes_to_gbasis_nodes * contact_term;
			}
			else
				contact_term.setZero(elasticity_term.size());

			if (state.is_adhesion_enabled())
			{
				NormalAdhesionForceDerivative::force_shape_derivative(*state.solve_data.normal_adhesion_form, state.diff_cached.normal_adhesion_collision_set(0), sol, adjoint, adhesion_term);
				adhesion_term = state.basis_nodes_to_gbasis_nodes * adhesion_term;
			}
			else
			{
				adhesion_term.setZero(elasticity_term.size());
			}
		}

		one_form -= elasticity_term + rhs_term + pressure_term + contact_term + adhesion_term;
		one_form = utils::flatten(utils::unflatten(one_form, state.mesh->dimension())(state.primitive_to_node(), Eigen::all));
	}

	void AdjointTools::dJ_shape_homogenization_adjoint_term(
		const State &state,
		const Eigen::MatrixXd &sol,
		const Eigen::MatrixXd &adjoint,
		Eigen::VectorXd &one_form)
	{
		Eigen::VectorXd elasticity_term, contact_term, adhesion_term;

		std::shared_ptr<NLHomoProblem> homo_problem = std::dynamic_pointer_cast<NLHomoProblem>(state.solve_data.nl_problem);
		assert(homo_problem);

		const int dim = state.mesh->dimension();
		one_form.setZero(state.n_geom_bases * dim);

		const Eigen::MatrixXd affine_adjoint = homo_problem->reduced_to_disp_grad(adjoint, true);
		const Eigen::VectorXd full_adjoint = homo_problem->NLProblem::reduced_to_full(adjoint.topRows(homo_problem->reduced_size())) + io::Evaluator::generate_linear_field(state.n_bases, state.mesh_nodes, affine_adjoint);

		ElasticForceDerivative::force_shape_derivative(*state.solve_data.elastic_form, 0, state.n_geom_bases, sol, sol, full_adjoint, elasticity_term);

		if (state.solve_data.contact_form)
		{
			if (const auto barrier_contact = dynamic_cast<const BarrierContactForm *>(state.solve_data.contact_form.get()))
			{
				BarrierContactForceDerivative::force_shape_derivative(*barrier_contact, state.diff_cached.collision_set(0), sol, full_adjoint, contact_term);
			}
			else if (const auto smooth_contact = dynamic_cast<const SmoothContactForm *>(state.solve_data.contact_form.get()))
			{
				SmoothContactForceDerivative::force_shape_derivative(*smooth_contact, state.diff_cached.smooth_collision_set(0), sol, full_adjoint, contact_term);
			}

			contact_term = state.basis_nodes_to_gbasis_nodes * contact_term;
		}
		else
			contact_term.setZero(elasticity_term.size());

		if (state.is_adhesion_enabled())
		{
			NormalAdhesionForceDerivative::force_shape_derivative(*state.solve_data.normal_adhesion_form, state.diff_cached.normal_adhesion_collision_set(0), sol, full_adjoint, adhesion_term);
			adhesion_term = state.basis_nodes_to_gbasis_nodes * adhesion_term;
		}
		else
		{
			adhesion_term.setZero(elasticity_term.size());
		}

		one_form = -(elasticity_term + contact_term + adhesion_term);

		Eigen::VectorXd force;
		homo_problem->FullNLProblem::gradient(sol, force);
		one_form -= state.basis_nodes_to_gbasis_nodes * utils::flatten(utils::unflatten(force, dim) * affine_adjoint);

		one_form = utils::flatten(utils::unflatten(one_form, dim)(state.primitive_to_node(), Eigen::all));
	}

	void AdjointTools::dJ_periodic_shape_adjoint_term(
		const State &state,
		const PeriodicMeshToMesh &periodic_mesh_map,
		const Eigen::VectorXd &periodic_mesh_representation,
		const Eigen::MatrixXd &sol,
		const Eigen::MatrixXd &adjoint,
		Eigen::VectorXd &one_form)
	{
		std::shared_ptr<NLHomoProblem> homo_problem = std::dynamic_pointer_cast<NLHomoProblem>(state.solve_data.nl_problem);
		assert(homo_problem);

		const Eigen::MatrixXd reduced_sol = homo_problem->full_to_reduced(sol, state.diff_cached.disp_grad());
		const Eigen::VectorXd extended_sol = homo_problem->reduced_to_extended(reduced_sol);

		const Eigen::VectorXd extended_adjoint = homo_problem->reduced_to_extended(adjoint, true);
		const Eigen::MatrixXd affine_adjoint = homo_problem->reduced_to_disp_grad(adjoint, true);
		const Eigen::VectorXd full_adjoint = homo_problem->NLProblem::reduced_to_full(adjoint.topRows(homo_problem->reduced_size())) + io::Evaluator::generate_linear_field(state.n_bases, state.mesh_nodes, affine_adjoint);

		const int dim = state.mesh->dimension();

		dJ_shape_homogenization_adjoint_term(state, sol, adjoint, one_form);

		StiffnessMatrix hessian;
		homo_problem->set_project_to_psd(false);
		homo_problem->FullNLProblem::hessian(sol, hessian);
		Eigen::VectorXd partial_term = full_adjoint.transpose() * hessian;
		partial_term = state.basis_nodes_to_gbasis_nodes * utils::flatten(utils::unflatten(partial_term, dim) * state.diff_cached.disp_grad());
		one_form -= utils::flatten(utils::unflatten(partial_term, dim)(state.primitive_to_node(), Eigen::all));

		one_form = periodic_mesh_map.apply_jacobian(one_form, periodic_mesh_representation);

		if (state.solve_data.periodic_contact_form)
		{
			Eigen::VectorXd contact_term;
			PeriodicContactForceDerivative::force_shape_derivative(*state.solve_data.periodic_contact_form, state, periodic_mesh_map, periodic_mesh_representation, state.solve_data.periodic_contact_form->collision_set(), extended_sol, extended_adjoint, contact_term);

			one_form -= contact_term;
		}
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

		Eigen::VectorXd elasticity_term, rhs_term, pressure_term, damping_term, mass_term, contact_term, friction_term, adhesion_term, tangential_adhesion_term;
		one_form.setZero(state.n_geom_bases * state.mesh->dimension());

		Eigen::VectorXd cur_p, cur_nu;
		for (int i = time_steps; i > 0; --i)
		{
			const int real_order = std::min(bdf_order, i);
			double beta = time_integrator::BDF::betas(real_order - 1);
			double beta_dt = beta * dt;
			const double t = i * dt + t0;

			Eigen::MatrixXd velocity = state.diff_cached.v(i);

			cur_p = adjoint_p.col(i);
			cur_nu = adjoint_nu.col(i);
			cur_p(state.boundary_nodes).setZero();
			cur_nu(state.boundary_nodes).setZero();

			{
				InertiaForceDerivative::force_shape_derivative(*state.solve_data.inertia_form, state.mesh->is_volume(), state.n_geom_bases, t, state.bases, state.geom_bases(), *(state.mass_matrix_assembler), state.mass_ass_vals_cache, velocity, cur_nu, mass_term);
				ElasticForceDerivative::force_shape_derivative(*state.solve_data.elastic_form, t, state.n_geom_bases, state.diff_cached.u(i), state.diff_cached.u(i), cur_p, elasticity_term);
				BodyForceDerivative::force_shape_derivative(*state.solve_data.body_form, state.n_geom_bases, t, state.diff_cached.u(i - 1), cur_p, rhs_term);
				PressureForceDerivative::force_shape_derivative(*state.solve_data.pressure_form, state.n_geom_bases, t, state.diff_cached.u(i), cur_p, pressure_term);
				pressure_term = state.basis_nodes_to_gbasis_nodes * pressure_term;

				if (state.solve_data.damping_form)
					ElasticForceDerivative::force_shape_derivative(*state.solve_data.damping_form, t, state.n_geom_bases, state.diff_cached.u(i), state.diff_cached.u(i - 1), cur_p, damping_term);
				else
					damping_term.setZero(mass_term.size());

				if (state.is_contact_enabled())
				{
					if (const auto barrier_contact = dynamic_cast<const BarrierContactForm *>(state.solve_data.contact_form.get()))
					{
						BarrierContactForceDerivative::force_shape_derivative(*barrier_contact, state.diff_cached.collision_set(i), state.diff_cached.u(i), cur_p, contact_term);
					}
					else if (const auto smooth_contact = dynamic_cast<const SmoothContactForm *>(state.solve_data.contact_form.get()))
					{
						SmoothContactForceDerivative::force_shape_derivative(*smooth_contact, state.diff_cached.smooth_collision_set(i), state.diff_cached.u(i), cur_p, contact_term);
					}
					contact_term = state.basis_nodes_to_gbasis_nodes * contact_term;
					// contact_term /= beta_dt * beta_dt;
				}
				else
					contact_term.setZero(mass_term.size());

				if (state.solve_data.friction_form)
				{
					FrictionForceDerivative::force_shape_derivative(*state.solve_data.friction_form, state.diff_cached.u(i - 1), state.diff_cached.u(i), cur_p, state.diff_cached.friction_collision_set(i), friction_term);
					friction_term = state.basis_nodes_to_gbasis_nodes * (friction_term / beta);
					// friction_term /= beta_dt * beta_dt;
				}
				else
					friction_term.setZero(mass_term.size());

				if (state.is_adhesion_enabled())
				{
					NormalAdhesionForceDerivative::force_shape_derivative(*state.solve_data.normal_adhesion_form, state.diff_cached.normal_adhesion_collision_set(i), state.diff_cached.u(i), cur_p, adhesion_term);
					adhesion_term = state.basis_nodes_to_gbasis_nodes * adhesion_term;
				}
				else
				{
					adhesion_term.setZero(mass_term.size());
				}

				if (state.solve_data.tangential_adhesion_form)
				{
					TangentialAdhesionForceDerivative::force_shape_derivative(*state.solve_data.tangential_adhesion_form, state.diff_cached.u(i - 1), state.diff_cached.u(i), cur_p, state.diff_cached.tangential_adhesion_collision_set(i), tangential_adhesion_term);
					tangential_adhesion_term = state.basis_nodes_to_gbasis_nodes * (tangential_adhesion_term / beta);
					// friction_term /= beta_dt * beta_dt;
				}
				else
					tangential_adhesion_term.setZero(mass_term.size());
			}

			one_form += beta_dt * (elasticity_term + rhs_term + pressure_term + damping_term + contact_term + friction_term + mass_term + adhesion_term + tangential_adhesion_term);
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
		InertiaForceDerivative::force_shape_derivative(*state.solve_data.inertia_form, state.mesh->is_volume(), state.n_geom_bases, t0, state.bases, state.geom_bases(), *(state.mass_matrix_assembler), state.mass_ass_vals_cache, state.diff_cached.v(0), sum_alpha_p, mass_term);

		one_form += mass_term;

		one_form = utils::flatten(utils::unflatten(one_form, state.mesh->dimension())(state.primitive_to_node(), Eigen::all));
	}

	void AdjointTools::dJ_material_static_adjoint_term(
		const State &state,
		const Eigen::MatrixXd &sol,
		const Eigen::MatrixXd &adjoint,
		Eigen::VectorXd &one_form)
	{
		Eigen::MatrixXd adjoint_zeroed = adjoint;
		adjoint_zeroed(state.boundary_nodes, Eigen::all).setZero();
		ElasticForceDerivative::force_material_derivative(*state.solve_data.elastic_form, 0, sol, sol, adjoint_zeroed, one_form);
	}

	void AdjointTools::dJ_material_transient_adjoint_term(
		const State &state,
		const Eigen::MatrixXd &adjoint_nu,
		const Eigen::MatrixXd &adjoint_p,
		Eigen::VectorXd &one_form)
	{
		const double t0 = state.args["time"]["t0"];
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

				ElasticForceDerivative::force_material_derivative(*state.solve_data.elastic_form, t0 + dt * i, state.diff_cached.u(i), state.diff_cached.u(i - 1), -cur_p, elasticity_term);
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

		std::shared_ptr<time_integrator::ImplicitTimeIntegrator> time_integrator =
			time_integrator::ImplicitTimeIntegrator::construct_time_integrator(state.args["time"]["integrator"]);
		{
			Eigen::MatrixXd solution, velocity, acceleration;
			solution = state.diff_cached.u(0);
			state.initial_velocity(velocity);
			state.initial_acceleration(acceleration);
			if (state.initial_vel_update.size() == state.ndof())
				velocity = state.initial_vel_update;
			const double dt = state.args["time"]["dt"];
			time_integrator->init(solution, velocity, acceleration, dt);
		}

		for (int t = 1; t <= time_steps; ++t)
		{
			const int real_order = std::min(bdf_order, t);
			double beta = time_integrator::BDF::betas(real_order - 1);

			const Eigen::MatrixXd surface_solution_prev = state.collision_mesh.vertices(utils::unflatten(state.diff_cached.u(t - 1), dim));
			// const Eigen::MatrixXd surface_solution = state.collision_mesh.vertices(utils::unflatten(state.diff_cached.u(t), dim));

			const Eigen::MatrixXd surface_velocities = state.collision_mesh.map_displacements(utils::unflatten(time_integrator->compute_velocity(state.diff_cached.u(t)), state.collision_mesh.dim()));
			time_integrator->update_quantities(state.diff_cached.u(t));

			if (const auto barrier_contact = dynamic_cast<const BarrierContactForm *>(state.solve_data.contact_form.get()))
			{
				Eigen::MatrixXd force = state.collision_mesh.to_full_dof(
					-state.solve_data.friction_form->friction_potential().force(
						state.diff_cached.friction_collision_set(t),
						state.collision_mesh,
						state.collision_mesh.rest_positions(),
						/*lagged_displacements=*/surface_solution_prev,
						surface_velocities,
						barrier_contact->barrier_potential(),
						barrier_contact->barrier_stiffness(),
						0., true));

				Eigen::VectorXd cur_p = adjoint_p.col(t);
				cur_p(state.boundary_nodes).setZero();

				one_form(0) += dot(cur_p, force) * beta * dt;
			}
		}
	}

	void AdjointTools::dJ_damping_transient_adjoint_term(
		const State &state,
		const Eigen::MatrixXd &adjoint_nu,
		const Eigen::MatrixXd &adjoint_p,
		Eigen::VectorXd &one_form)
	{
		const double t0 = state.args["time"]["t0"];
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

				ElasticForceDerivative::force_material_derivative(*state.solve_data.damping_form, t * dt + t0, state.diff_cached.u(t), state.diff_cached.u(t - 1), -cur_p, damping_term);
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

	void AdjointTools::dJ_dirichlet_static_adjoint_term(
		const State &state,
		const Eigen::MatrixXd &adjoint,
		Eigen::VectorXd &one_form)
	{
		const int n_dirichlet_dof = state.boundary_nodes.size();
		StiffnessMatrix gradd_h = state.diff_cached.gradu_h(0);
		std::set<int> boundary_nodes_set(state.boundary_nodes.begin(), state.boundary_nodes.end());
		gradd_h.prune([&boundary_nodes_set](const Eigen::Index &row, const Eigen::Index &col, const FullNLProblem::Scalar &value) {
			if (row != col)
				return value;
			if (boundary_nodes_set.find(row) == boundary_nodes_set.end())
				return value;
			return 0.0;
		});
		one_form.setZero(state.ndof());
		one_form(state.boundary_nodes) -= adjoint(state.boundary_nodes, 0);
		one_form(state.boundary_nodes) -= (gradd_h.transpose() * adjoint.col(0))(state.boundary_nodes);
		one_form = utils::flatten(utils::unflatten(one_form, state.mesh->dimension())(state.primitive_to_node(), Eigen::all));
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

		// Map dirichlet gradient on each node to dirichlet gradient on boundary ids

		one_form.setZero(time_steps * n_dirichlet_dof);
		for (int i = 1; i <= time_steps; ++i)
		{
			const int real_order = std::min(bdf_order, i);
			const double beta_dt = time_integrator::BDF::betas(real_order - 1) * dt;

			one_form.segment((i - 1) * n_dirichlet_dof, n_dirichlet_dof) = -(1. / beta_dt) * adjoint_p(state.boundary_nodes, i);
		}
	}

	void AdjointTools::dJ_pressure_static_adjoint_term(
		const State &state,
		const std::vector<int> &boundary_ids,
		const Eigen::MatrixXd &sol,
		const Eigen::MatrixXd &adjoint,
		Eigen::VectorXd &one_form)
	{
		const int n_pressure_dof = boundary_ids.size();

		one_form.setZero(n_pressure_dof);

		for (int i = 0; i < boundary_ids.size(); ++i)
		{
			double pressure_term = PressureForceDerivative::force_pressure_derivative(
				*state.solve_data.pressure_form,
				state.n_geom_bases,
				0,
				boundary_ids[i],
				sol,
				adjoint);
			one_form(i) = pressure_term;
		}
	}

	void AdjointTools::dJ_pressure_transient_adjoint_term(
		const State &state,
		const std::vector<int> &boundary_ids,
		const Eigen::MatrixXd &adjoint_nu,
		const Eigen::MatrixXd &adjoint_p,
		Eigen::VectorXd &one_form)
	{
		const double t0 = state.args["time"]["t0"];
		const double dt = state.args["time"]["dt"];
		const int time_steps = state.args["time"]["time_steps"];
		const int bdf_order = get_bdf_order(state);

		const int n_pressure_dof = boundary_ids.size();

		one_form.setZero(time_steps * n_pressure_dof);
		Eigen::VectorXd cur_p, cur_nu;
		for (int i = time_steps; i > 0; --i)
		{
			const int real_order = std::min(bdf_order, i);
			double beta = time_integrator::BDF::betas(real_order - 1);
			double beta_dt = beta * dt;
			const double t = i * dt + t0;

			cur_p = adjoint_p.col(i);
			cur_nu = adjoint_nu.col(i);
			cur_p(state.boundary_nodes).setZero();
			cur_nu(state.boundary_nodes).setZero();

			for (int b = 0; b < boundary_ids.size(); ++b)
			{
				double pressure_term = PressureForceDerivative::force_pressure_derivative(
					*state.solve_data.pressure_form,
					state.n_geom_bases,
					t,
					boundary_ids[b],
					state.diff_cached.u(i),
					cur_p);
				one_form((i - 1) * n_pressure_dof + b) = -beta_dt * pressure_term;
			}
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
		const double t0 = state.problem->is_time_dependent() ? state.args["time"]["t0"].get<double>() : 0.0;
		const double dt = state.problem->is_time_dependent() ? state.args["time"]["dt"].get<double>() : 0.0;

		term = Eigen::MatrixXd::Zero(state.n_bases * actual_dim, 1);

		if (!j.depend_on_u() && !j.depend_on_gradu() && !j.depend_on_gradu_local())
			return;

		if (spatial_integral_type == SpatialIntegralType::Volume)
		{
			auto storage = utils::create_thread_storage(LocalThreadVecStorage(term.size()));
			utils::maybe_parallel_for(n_elements, [&](int start, int end, int thread_id) {
				LocalThreadVecStorage &local_storage = utils::get_local_thread_storage(storage, thread_id);

				Eigen::MatrixXd u, grad_u;
				Eigen::MatrixXd lambda, mu;
				Eigen::MatrixXd dj_du, dj_dgradu, dj_dgradx;

				IntegrableFunctional::ParameterType params;
				params.t = dt * cur_step + t0;
				params.step = cur_step;

				for (int e = start; e < end; ++e)
				{
					if (interested_ids.size() != 0 && interested_ids.find(state.mesh->get_body_id(e)) == interested_ids.end())
						continue;

					assembler::ElementAssemblyValues &vals = local_storage.vals;
					state.ass_vals_cache.compute(e, state.mesh->is_volume(), bases[e], gbases[e], vals);

					const quadrature::Quadrature &quadrature = vals.quadrature;
					local_storage.da = vals.det.array() * quadrature.weights.array();

					const Eigen::MatrixXd lame_params = extract_lame_params(state.assembler->parameters(), e, params.t, quadrature.points, vals.val);

					const int n_loc_bases_ = int(vals.basis_values.size());

					io::Evaluator::interpolate_at_local_vals(e, dim, actual_dim, vals, solution, u, grad_u);

					params.elem = e;
					params.body_id = state.mesh->get_body_id(e);

					dj_dgradu.resize(0, 0);
					if (j.depend_on_gradu())
					{
						j.dj_dgradu(lame_params, quadrature.points, vals.val, u, grad_u, Eigen::MatrixXd::Zero(0, 0) /*Not used*/, vals, params, dj_dgradu);
						for (int q = 0; q < dj_dgradu.rows(); q++)
							dj_dgradu.row(q) *= local_storage.da(q);
					}

					dj_du.resize(0, 0);
					if (j.depend_on_u())
					{
						j.dj_du(lame_params, quadrature.points, vals.val, u, grad_u, Eigen::MatrixXd::Zero(0, 0) /*Not used*/, vals, params, dj_du);
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
		else if (spatial_integral_type == SpatialIntegralType::Surface)
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

				IntegrableFunctional::ParameterType params;
				params.t = dt * cur_step + t0;
				params.step = cur_step;

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

						const Eigen::MatrixXd lame_params = extract_lame_params(state.assembler->parameters(), e, params.t, points, vals.val);

						// normal = normal * vals.jac_it[0]; // assuming linear geometry

						const int n_loc_bases_ = int(vals.basis_values.size());

						params.elem = e;
						params.body_id = state.mesh->get_body_id(e);
						params.boundary_id = state.mesh->get_boundary_id(global_primitive_id);

						dj_dgradu.resize(0, 0);
						if (j.depend_on_gradu())
						{
							j.dj_dgradu(lame_params, points, vals.val, u, grad_u, normal, vals, params, dj_dgradu);
							for (int q = 0; q < dj_dgradu.rows(); q++)
								dj_dgradu.row(q) *= weights(q);
						}

						dj_dgradu_local.resize(0, 0);
						if (j.depend_on_gradu_local())
						{
							j.dj_dgradu_local(lame_params, points, vals.val, u, grad_u, normal, vals, params, dj_dgradu_local);
							for (int q = 0; q < dj_dgradu_local.rows(); q++)
								dj_dgradu_local.row(q) *= weights(q);
						}

						dj_du.resize(0, 0);
						if (j.depend_on_u())
						{
							j.dj_du(lame_params, points, vals.val, u, grad_u, normal, vals, params, dj_du);
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
		else if (spatial_integral_type == SpatialIntegralType::VertexSum)
		{
			std::vector<bool> traversed(state.n_bases, false);
			IntegrableFunctional::ParameterType params;
			params.t = dt * cur_step + t0;
			params.step = cur_step;
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

					const Eigen::MatrixXd lame_params = extract_lame_params(state.assembler->parameters(), e, params.t, Eigen::MatrixXd::Zero(1, dim) /*Not used*/, g.node);

					params.node = g.index;
					params.elem = e;
					params.body_id = state.mesh->get_body_id(e);
					Eigen::MatrixXd val;
					j.dj_du(lame_params, Eigen::MatrixXd::Zero(1, dim) /*Not used*/, g.node, solution.block(g.index * dim, 0, dim, 1).transpose(), Eigen::MatrixXd::Zero(1, dim * actual_dim) /*Not used*/, Eigen::MatrixXd::Zero(0, 0) /*Not used*/, assembler::ElementAssemblyValues(), params, val);
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

	void AdjointTools::scaled_jacobian(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, Eigen::VectorXd &quality)
	{
		const int dim = F.cols() - 1;

		quality.setZero(F.rows());
		if (dim == 2)
		{
			for (int i = 0; i < F.rows(); i++)
			{
				Eigen::RowVector3d e0;
				e0(2) = 0;
				e0.head(2) = V.row(F(i, 2)) - V.row(F(i, 1));
				Eigen::RowVector3d e1;
				e1(2) = 0;
				e1.head(2) = V.row(F(i, 0)) - V.row(F(i, 2));
				Eigen::RowVector3d e2;
				e2(2) = 0;
				e2.head(2) = V.row(F(i, 1)) - V.row(F(i, 0));

				double l0 = e0.norm();
				double l1 = e1.norm();
				double l2 = e2.norm();

				double A = 0.5 * (e0.cross(e1)).norm();
				double Lmax = std::max(l0 * l1, std::max(l1 * l2, l0 * l2));

				quality(i) = 2 * A * (2 / sqrt(3)) / Lmax;
			}
		}
		else
		{
			for (int i = 0; i < F.rows(); i++)
			{
				Eigen::RowVector3d e0 = V.row(F(i, 1)) - V.row(F(i, 0));
				Eigen::RowVector3d e1 = V.row(F(i, 2)) - V.row(F(i, 1));
				Eigen::RowVector3d e2 = V.row(F(i, 0)) - V.row(F(i, 2));
				Eigen::RowVector3d e3 = V.row(F(i, 3)) - V.row(F(i, 0));
				Eigen::RowVector3d e4 = V.row(F(i, 3)) - V.row(F(i, 1));
				Eigen::RowVector3d e5 = V.row(F(i, 3)) - V.row(F(i, 2));

				double l0 = e0.norm();
				double l1 = e1.norm();
				double l2 = e2.norm();
				double l3 = e3.norm();
				double l4 = e4.norm();
				double l5 = e5.norm();

				double J = std::abs((e0.cross(e3)).dot(e2));

				double a1 = l0 * l2 * l3;
				double a2 = l0 * l1 * l4;
				double a3 = l1 * l2 * l5;
				double a4 = l3 * l4 * l5;

				double a = std::max({a1, a2, a3, a4, J});
				quality(i) = J * sqrt(2) / a;
			}
		}
	}

} // namespace polyfem::solver
