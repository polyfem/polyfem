#include "AdjointForm.hpp"

#include <polyfem/utils/MaybeParallelFor.hpp>
#include <polyfem/utils/Timer.hpp>
#include <polyfem/io/Evaluator.hpp>
#include <polyfem/State.hpp>

#include <polyfem/utils/IntegrableFunctional.hpp>
#include <polyfem/utils/BoundarySampler.hpp>

#include <polyfem/solver/forms/ElasticForm.hpp>
#include <polyfem/solver/forms/ContactForm.hpp>
#include <polyfem/solver/forms/FrictionForm.hpp>
#include <polyfem/solver/forms/BodyForm.hpp>
#include <polyfem/solver/forms/InertiaForm.hpp>

#include <polyfem/time_integrator/BDF.hpp>

namespace polyfem::solver {
namespace { 
	double dot(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B) { return (A.array() * B.array()).sum(); }

	void get_bdf_parts(
		const int bdf_order,
		const int index,
		const std::vector<Eigen::MatrixXd> &adjoint_p,
		const std::vector<Eigen::MatrixXd> &adjoint_nu,
		Eigen::MatrixXd &sum_adjoint_p,
		Eigen::MatrixXd &sum_adjoint_nu,
		double &beta)
	{
		sum_adjoint_p.setZero(adjoint_p[0].size(), 1);
		sum_adjoint_nu.setZero(adjoint_nu[0].size(), 1);

		int num = std::min(bdf_order, int(adjoint_p.size()) - 2 - index);
		for (int j = 1; j <= num; ++j)
		{
			int order = std::min(bdf_order, index + j);
			sum_adjoint_p += -time_integrator::BDF::alphas(order - 1)[j - 1] * adjoint_p[index + j];
			sum_adjoint_nu += -time_integrator::BDF::alphas(order - 1)[j - 1] * adjoint_nu[index + j];
		}
		int order = std::min(bdf_order, index);
		if (order >= 1)
			beta = time_integrator::BDF::betas(order - 1);
		else
			beta = std::nan("");
	}

    void get_transient_quadrature_weights(const std::string &transient_integral_type, const int n, const double dt, std::vector<double> &weights)
    {
        weights.assign(n + 1, dt);
        if (transient_integral_type == "uniform")
        {
            weights[0] = 0;
        }
        else if (transient_integral_type == "trapezoidal")
        {
            weights[0] = dt / 2.;
            weights[weights.size() - 1] = dt / 2.;
        }
        else if (transient_integral_type == "simpson")
        {
            weights[0] = dt / 3.;
            weights[weights.size() - 1] = dt / 3.;
            for (int i = 1; i < weights.size() - 1; i++)
            {
                if (i % 2)
                    weights[i] = dt * 4. / 3.;
                else
                    weights[i] = dt * 2. / 4.;
            }
        }
        else if (transient_integral_type == "final")
        {
            weights.assign(n + 1, 0);
            weights[n] = 1;
        }
        else if (transient_integral_type.find("step_") != std::string::npos)
        {
            weights.assign(n + 1, 0);
            int step = std::stoi(transient_integral_type.substr(5));
            assert(step > 0 && step < weights.size());
            weights[step] = 1;
        }
        else
            assert(false);
    }

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

	void vector2matrix(const Eigen::VectorXd &vec, Eigen::MatrixXd &mat)
	{
		int size = sqrt(vec.size());
		assert(size * size == vec.size());

		mat.resize(size, size);
		for (int i = 0; i < size; i++)
			for (int j = 0; j < size; j++)
				mat(i, j) = vec(i * size + j);
	}
}

	double AdjointForm::value(
		const State &state,
		const IntegrableFunctional &j,
		const std::set<int> &interested_ids, // either body id or surface id
		const SpatialIntegralType spatial_integral_type,
		const std::string &transient_integral_type)
	{
		if (state.problem->is_time_dependent())
			return integrate_objective_transient(state, j, interested_ids, spatial_integral_type, transient_integral_type);
		else
			return integrate_objective(state, j, state.diff_cached[0].u, interested_ids, spatial_integral_type);
	}

	void AdjointForm::gradient(
		const State &state,
		const IntegrableFunctional &j,
		const std::string &param,
		Eigen::VectorXd &grad,
		const std::set<int> &interested_ids, // either body id or surface id
		const SpatialIntegralType spatial_integral_type,
		const std::string &transient_integral_type)
	{
		if (state.problem->is_time_dependent())
		{
			std::vector<Eigen::MatrixXd> adjoint_nu, adjoint_p;
			std::vector<Eigen::VectorXd> adjoint_rhs;
			dJ_du_transient(state, j, interested_ids, spatial_integral_type, transient_integral_type, adjoint_rhs);
			state.solve_transient_adjoint(adjoint_rhs, adjoint_nu, adjoint_p, param == "dirichlet");
			if (param == "material")
				dJ_material_transient(state, adjoint_nu, adjoint_p, grad);
			else if (param == "shape")
				dJ_shape_transient(state, adjoint_nu, adjoint_p, j, interested_ids, spatial_integral_type, transient_integral_type, grad);
			else if (param == "friction")
				dJ_friction_transient(state, adjoint_nu, adjoint_p, grad);
			else if (param == "damping")
				dJ_damping_transient(state, adjoint_nu, adjoint_p, grad);
			else if (param == "initial")
				dJ_initial_condition(state, adjoint_nu, adjoint_p, grad);
			else if (param == "dirichlet")
				dJ_dirichlet_transient(state, adjoint_nu, adjoint_p, grad);
			else
				log_and_throw_error("Unknown design parameter!");
		}
		else
		{
			Eigen::MatrixXd adjoint;
			Eigen::VectorXd adjoint_rhs;
			dJ_du_step(state, j, state.diff_cached[0].u, interested_ids, spatial_integral_type, 0, adjoint_rhs);
			state.solve_adjoint(adjoint_rhs, adjoint);
			if (param == "material")
				dJ_material_static(state, state.diff_cached[0].u, adjoint, grad);
			else if (param == "shape")
				dJ_shape_static(state, state.diff_cached[0].u, adjoint, j, interested_ids, spatial_integral_type, grad);
			else if (param == "topology")
				dJ_topology_static(state, state.diff_cached[0].u, adjoint, j, interested_ids, spatial_integral_type, grad);
			else if (param == "friction")
				log_and_throw_error("Static friction coefficient grad not implemented!");
			else if (param == "dirichlet")
				log_and_throw_error("Static Dirichlet BC grad not implemented!");
			else
				log_and_throw_error("Unknown design parameter!");
		}
	}

    double AdjointForm::integrate_objective(
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
		if (spatial_integral_type == SpatialIntegralType::VOLUME)
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
                    if (interested_ids.size() != 0 && interested_ids.find(e) == interested_ids.end())
                        continue;

                    assembler::ElementAssemblyValues &vals = local_storage.vals;
                    state.ass_vals_cache.compute(e, state.mesh->is_volume(), bases[e], gbases[e], vals);
                    io::Evaluator::interpolate_at_local_vals(e, dim, actual_dim, vals, solution, u, grad_u);

					const quadrature::Quadrature &quadrature = vals.quadrature;
					local_storage.da = vals.det.array() * quadrature.weights.array();

					params["elem"] = e;
					params["body_id"] = state.mesh->get_body_id(e);
					j.evaluate(state.assembler.lame_params(), quadrature.points, vals.val, u, grad_u, params, result);

                    local_storage.val += dot(result, local_storage.da);
                }
            });
			for (const LocalThreadScalarStorage &local_storage : storage)
				integral += local_storage.val;
		}
		else if (spatial_integral_type == SpatialIntegralType::SURFACE)
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

                        normal = normal * vals.jac_it[0]; // assuming linear geometry

                        params["elem"] = e;
                        params["body_id"] = state.mesh->get_body_id(e);
                        params["boundary_id"] = state.mesh->get_boundary_id(global_primitive_id);
						j.evaluate(state.assembler.lame_params(), points, vals.val, u, grad_u, params, result);

                        local_storage.val += dot(result, weights);
                    }
				}
			});
			for (const LocalThreadScalarStorage &local_storage : storage)
				integral += local_storage.val;
		}
		else if (spatial_integral_type == SpatialIntegralType::VERTEX_SUM)
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
					params["elem"] = -1;
					params["body_id"] = state.mesh->get_body_id(e);
					params["boundary_id"] = -1;
					Eigen::MatrixXd val;
					j.evaluate(state.assembler.lame_params(), Eigen::MatrixXd::Zero(1, dim) /*Not used*/, g.node, solution.block(g.index * dim, 0, dim, 1).transpose(), Eigen::MatrixXd::Zero(1, dim * actual_dim) /*Not used*/, params, val);
					integral += val(0);
					traversed[g.index] = true;
				}
			}
		}
            
		return integral;
	}

    double AdjointForm::integrate_objective_transient(
        const State &state, 
        const IntegrableFunctional &j, 
        const std::set<int> &interested_ids,
        const SpatialIntegralType spatial_integral_type,
        const std::string &transient_integral_type)
    {
		const double dt = state.args["time"]["dt"];
		const int n_steps = state.args["time"]["time_steps"];
		double result = 0;

		std::vector<double> weights;
		get_transient_quadrature_weights(transient_integral_type, n_steps, dt, weights);
		for (int i = 0; i <= n_steps; ++i)
		{
			if (weights[i] == 0)
				continue;
			result += weights[i] * integrate_objective(state, j, state.diff_cached[i].u, interested_ids, spatial_integral_type, i);
		}

		return result;
    }

	void AdjointForm::compute_shape_derivative_functional_term(
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

		if (spatial_integral_type == SpatialIntegralType::VOLUME)
		{
			utils::maybe_parallel_for(n_elements, [&](int start, int end, int thread_id) {
				LocalThreadVecStorage &local_storage = utils::get_local_thread_storage(storage, thread_id);

				Eigen::MatrixXd u, grad_u, j_val, dj_du, dj_dx, lambda, mu;

				json params = {};
				params["step"] = cur_time_step;

				for (int e = start; e < end; ++e)
				{
                    if (interested_ids.size() != 0 && interested_ids.find(e) == interested_ids.end())
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
					
					j.evaluate(state.assembler.lame_params(), quadrature.points, vals.val, u, grad_u, params, j_val);

					if (j.depend_on_gradu() || j.depend_on_u())
						dj_du = j.grad_j(state.assembler.lame_params(), quadrature.points, vals.val, u, grad_u, params);

					if (j.depend_on_x())
						j.dj_dx(state.assembler.lame_params(), quadrature.points, vals.val, u, grad_u, params, dj_dx);

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
									vector2matrix(dj_du.row(q), tau_q);
									vector2matrix(grad_u.row(q), grad_u_q);
								}
								else // Laplacian PDE
								{
									tau_q = dj_du.row(q);
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
		else if (spatial_integral_type == SpatialIntegralType::SURFACE)
		{
			utils::maybe_parallel_for(state.total_local_boundary.size(), [&](int start, int end, int thread_id) {
				LocalThreadVecStorage &local_storage = utils::get_local_thread_storage(storage, thread_id);

				Eigen::MatrixXd uv, points, normal;
				Eigen::VectorXd &weights = local_storage.da;

				Eigen::MatrixXd u, grad_u, j_val, dj_du, dj_dx, lambda, mu;

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

						vals.compute(e, state.mesh->is_volume(), points, gbases[e], gbases[e]);

						normal = normal * vals.jac_it[0]; // assuming linear geometry

                        params["elem"] = e;
                        params["body_id"] = state.mesh->get_body_id(e);
                        params["boundary_id"] = state.mesh->get_boundary_id(global_primitive_id);
						
						j.evaluate(state.assembler.lame_params(), points, vals.val, u, grad_u, params, j_val);
						j_val = j_val.array().colwise() * weights.array();

						if (j.depend_on_gradu())
						{
							dj_du = j.grad_j(state.assembler.lame_params(), points, vals.val, u, grad_u, params);
							dj_du = dj_du.array().colwise() * weights.array();
						}

						if (j.depend_on_x())
						{
							j.dj_dx(state.assembler.lame_params(), points, vals.val, u, grad_u, params, dj_dx);
							dj_dx = dj_dx.array().colwise() * weights.array();
						}

						const auto nodes = gbases[e].local_nodes_for_primitive(lb.global_primitive_id(i), *state.mesh);

						if (nodes.size() != dim)
							log_and_throw_error("Only linear geometry is supported in differentiable surface integral functional!");

						Eigen::MatrixXd grad_u_q, tau_q;
						for (long n = 0; n < nodes.size(); ++n)
						{
							const assembler::AssemblyValues &v = vals.basis_values[nodes(n)];
							// integrate j * div(gbases) over the whole boundary
							for (int d = 0; d < dim; d++)
							{
								double velocity_div = 0;
								if (state.mesh->is_volume())
								{
									Eigen::Vector3d dr_du = gbases[e].bases[nodes(1)].global()[0].node - gbases[e].bases[nodes(0)].global()[0].node;
									Eigen::Vector3d dr_dv = gbases[e].bases[nodes(2)].global()[0].node - gbases[e].bases[nodes(0)].global()[0].node;

									// compute dtheta
									Eigen::Vector3d dtheta_du, dtheta_dv;
									dtheta_du.setZero();
									dtheta_dv.setZero();
									if (0 == n)
									{
										dtheta_du(d) = -1;
										dtheta_dv(d) = -1;
									}
									else if (1 == n)
										dtheta_du(d) = 1;
									else if (2 == n)
										dtheta_dv(d) = 1;
									else
										assert(false);

									velocity_div = (dr_du.cross(dr_dv)).dot(dtheta_du.cross(dr_dv) + dr_du.cross(dtheta_dv)) / (dr_du.cross(dr_dv)).squaredNorm();
								}
								else
								{
									Eigen::VectorXd dr = gbases[e].bases[nodes(1)].global()[0].node - gbases[e].bases[nodes(0)].global()[0].node;

									// compute dtheta
									Eigen::VectorXd dtheta;
									dtheta.setZero(dr.rows(), dr.cols());
									if (0 == n)
										dtheta(d) = -1;
									else if (1 == n)
										dtheta(d) = 1;
									else
										assert(false);

									velocity_div = dr.dot(dtheta) / dr.squaredNorm();
								}

								for (int q = 0; q < weights.size(); ++q)
								{
									local_storage.vec(v.global[0].index * dim + d) += j_val(q) * velocity_div;

									if (j.depend_on_x())
										local_storage.vec(v.global[0].index * dim + d) += v.val(q) * dj_dx(q, d);

									if (j.depend_on_gradu())
									{
										if (dim == actual_dim) // Elasticity PDE
										{
											vector2matrix(grad_u.row(q), grad_u_q);
											vector2matrix(dj_du.row(q), tau_q);
										}
										else // Laplacian PDE
										{
											grad_u_q = grad_u.row(q);
											tau_q = dj_du.row(q);
										}
											
										local_storage.vec(v.global[0].index * dim + d) += -dot(tau_q, grad_u_q.col(d) * v.grad_t_m.row(q));
									}
								}
							}
						}
					}
				}
			});
		}
		else if (spatial_integral_type == SpatialIntegralType::VERTEX_SUM)
		{
			log_and_throw_error("Shape derivative of vertex sum type functional is not implemented!");
		}
		for (const LocalThreadVecStorage &local_storage : storage)
			term += local_storage.vec;
	}

	void AdjointForm::dJ_shape_static(
		const State &state,
		const Eigen::MatrixXd &sol,
		const Eigen::MatrixXd &adjoint,
		const IntegrableFunctional &j,
		const std::set<int> &interested_ids,
		const SpatialIntegralType spatial_integral_type,
		Eigen::VectorXd &one_form)
	{
		Eigen::VectorXd functional_term, elasticity_term, rhs_term, contact_term, friction_term;

		compute_shape_derivative_functional_term(state, sol, j, interested_ids, spatial_integral_type, functional_term, 0);
		one_form = functional_term;

		if (j.depend_on_u() || j.depend_on_gradu())
		{
			state.solve_data.elastic_form->force_shape_derivative(state.n_geom_bases, sol, sol, adjoint, elasticity_term);
			state.solve_data.body_form->force_shape_derivative(state.n_geom_bases, sol, adjoint, rhs_term);
			
			if (state.is_contact_enabled())
			{
				state.solve_data.contact_form->force_shape_derivative(state.solve_data.contact_form->get_constraint_set(), sol, adjoint, contact_term);
				contact_term = state.down_sampling_mat * contact_term;
			}
			else
				contact_term.setZero(elasticity_term.size());
			one_form += elasticity_term + rhs_term + contact_term;
		}
	}

	void AdjointForm::dJ_shape_transient(
		const State &state,
		const std::vector<Eigen::MatrixXd> &adjoint_nu,
		const std::vector<Eigen::MatrixXd> &adjoint_p,
		const IntegrableFunctional &j,
		const std::set<int> &interested_ids,
		const SpatialIntegralType spatial_integral_type,
        const std::string &transient_integral_type,
		Eigen::VectorXd &one_form)
	{
		const double dt = state.args["time"]["dt"];
		const int time_steps = state.args["time"]["time_steps"];
		const int bdf_order = state.get_bdf_order();

		Eigen::VectorXd elasticity_term, rhs_term, damping_term, mass_term, contact_term, friction_term, functional_term;
		one_form.setZero(state.n_geom_bases * state.mesh->dimension());

		std::vector<double> weights;
		get_transient_quadrature_weights(transient_integral_type, time_steps, dt, weights);
		for (int i = time_steps; i > 0; --i)
		{
			const int real_order = std::min(bdf_order, i);
			double beta = time_integrator::BDF::betas(real_order - 1);
			double beta_dt = beta * dt;

			Eigen::MatrixXd velocity;
			{
				velocity = state.diff_cached[i].u;
				for (int o = 1; o <= real_order; o++)
					velocity -= time_integrator::BDF::alphas(real_order - 1)[o - 1] * state.diff_cached[i - o].u;
				velocity /= beta_dt;
			}

			if (weights[i] == 0)
				functional_term.setZero(one_form.rows(), one_form.cols());
			else
				compute_shape_derivative_functional_term(state, state.diff_cached[i].u, j, interested_ids, spatial_integral_type, functional_term, i);

			if (j.depend_on_u() || j.depend_on_gradu())
			{
				state.solve_data.inertia_form->force_shape_derivative(state.mesh->is_volume(), state.n_geom_bases, state.bases, state.geom_bases(), state.assembler, state.mass_ass_vals_cache, velocity, adjoint_nu[i], mass_term);
				state.solve_data.elastic_form->force_shape_derivative(state.n_geom_bases, state.diff_cached[i].u, state.diff_cached[i].u, -adjoint_p[i], elasticity_term);
				state.solve_data.body_form->force_shape_derivative(state.n_geom_bases, state.diff_cached[i].u, -adjoint_p[i], rhs_term);
				
				if (state.solve_data.damping_form)
					state.solve_data.damping_form->force_shape_derivative(state.n_geom_bases, state.diff_cached[i].u, state.diff_cached[i - 1].u, -adjoint_p[i], damping_term);
				else
					damping_term.setZero(mass_term.size());
				
				if (state.is_contact_enabled())
				{
					state.solve_data.contact_form->force_shape_derivative(state.diff_cached[i].contact_set, state.diff_cached[i].u, -adjoint_p[i], contact_term);
					contact_term = state.down_sampling_mat * contact_term;
					contact_term /= beta_dt * beta_dt;
				}
				else
					contact_term.setZero(mass_term.size());
				
				if (state.solve_data.friction_form)
				{
					state.solve_data.friction_form->force_shape_derivative(state.diff_cached[i - 1].u, state.diff_cached[i].u, -adjoint_p[i], state.diff_cached[i].friction_constraint_set, friction_term);
					friction_term = state.down_sampling_mat * friction_term;
					friction_term /= beta_dt * beta_dt;
				}
				else
					friction_term.setZero(mass_term.size());
			}
			else
			{
				mass_term.setZero(one_form.rows(), one_form.cols());
				elasticity_term.setZero(one_form.rows(), one_form.cols());
				rhs_term.setZero(one_form.rows(), one_form.cols());
				damping_term.setZero(one_form.rows(), one_form.cols());
				contact_term.setZero(one_form.rows(), one_form.cols());
				friction_term.setZero(one_form.rows(), one_form.cols());
			}

			one_form += weights[i] * functional_term + beta_dt * (elasticity_term + rhs_term + damping_term + contact_term + friction_term + mass_term);
		}
		
		// time step 0
		if (j.depend_on_u() || j.depend_on_gradu())
		{
			double beta;
			Eigen::MatrixXd sum_alpha_p, sum_alpha_nu;
			get_bdf_parts(bdf_order, 0, adjoint_p, adjoint_nu, sum_alpha_p, sum_alpha_nu, beta);
			state.solve_data.inertia_form->force_shape_derivative(state.mesh->is_volume(), state.n_geom_bases, state.bases, state.geom_bases(), state.assembler, state.mass_ass_vals_cache, state.initial_velocity_cache, sum_alpha_p, mass_term);
		}
		else
			mass_term.setZero(one_form.rows(), one_form.cols());

		compute_shape_derivative_functional_term(state, state.diff_cached[0].u, j, interested_ids, spatial_integral_type, functional_term, 0);
		one_form += weights[0] * functional_term + mass_term;
	}

	void AdjointForm::dJ_material_static(
		const State &state,
		const Eigen::MatrixXd &sol,
		const Eigen::MatrixXd &adjoint,
		Eigen::VectorXd &one_form)
	{
		state.solve_data.elastic_form->foce_material_derivative(sol, sol, adjoint, one_form);
	}

	void AdjointForm::dJ_material_transient(
		const State &state,
		const std::vector<Eigen::MatrixXd> &adjoint_nu,
		const std::vector<Eigen::MatrixXd> &adjoint_p,
		Eigen::VectorXd &one_form)
	{
		const double dt = state.args["time"]["dt"];
		const int time_steps = state.args["time"]["time_steps"];
		const int bdf_order = state.get_bdf_order();

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
				
				state.solve_data.elastic_form->foce_material_derivative(state.diff_cached[i].u, state.diff_cached[i - 1].u, -adjoint_p[i], elasticity_term);
				local_storage.vec += beta_dt * elasticity_term;
			}
		});

        for (const LocalThreadVecStorage &local_storage : storage)
            one_form += local_storage.vec;
	}

	void AdjointForm::dJ_friction_transient(
		const State &state,
		const std::vector<Eigen::MatrixXd> &adjoint_nu,
		const std::vector<Eigen::MatrixXd> &adjoint_p,
		Eigen::VectorXd &one_form)
	{
		const double dt = state.args["time"]["dt"];
		const double mu = state.solve_data.friction_form->mu();
		const int time_steps = state.args["time"]["time_steps"];
		const int dim = state.mesh->dimension();
		const int bdf_order = state.get_bdf_order();

		one_form.setZero(1);

		auto storage = utils::create_thread_storage(LocalThreadScalarStorage());

		utils::maybe_parallel_for(time_steps, [&](int start, int end, int thread_id) {
			LocalThreadScalarStorage &local_storage = utils::get_local_thread_storage(storage, thread_id);
			for (int t_aux = start; t_aux < end; ++t_aux)
			{
				const int t = time_steps - t_aux;
				const int real_order = std::min(bdf_order, t);
				double beta = time_integrator::BDF::betas(real_order - 1);

				Eigen::MatrixXd surface_solution_prev = state.collision_mesh.vertices(utils::unflatten(state.diff_cached[t - 1].u, dim));
				Eigen::MatrixXd surface_solution = state.collision_mesh.vertices(utils::unflatten(state.diff_cached[t].u, dim));

				auto force = -ipc::compute_friction_force(state.collision_mesh, state.collision_mesh.vertices_at_rest(), surface_solution_prev, surface_solution, state.diff_cached[t].friction_constraint_set, state.solve_data.contact_form->dhat(), state.solve_data.contact_form->barrier_stiffness(), state.solve_data.friction_form->epsv_dt());

				local_storage.val += dot(adjoint_p[t], state.collision_mesh.to_full_dof(force)) / (beta * mu * dt);
			}
		});

		for (const LocalThreadScalarStorage &local_storage : storage)
			one_form(0) += local_storage.val;
	}

	void AdjointForm::dJ_damping_transient(
		const State &state,
		const std::vector<Eigen::MatrixXd> &adjoint_nu,
		const std::vector<Eigen::MatrixXd> &adjoint_p,
		Eigen::VectorXd &one_form)
	{
		const double dt = state.args["time"]["dt"];
		const int time_steps = state.args["time"]["time_steps"];
		const int bdf_order = state.get_bdf_order();

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

				state.solve_data.damping_form->foce_material_derivative(state.diff_cached[t].u, state.diff_cached[t - 1].u, -adjoint_p[t], damping_term);
				local_storage.vec += (beta * dt) * damping_term;
			}
		});

		for (const LocalThreadVecStorage &local_storage : storage)
			one_form += local_storage.vec;
	}

	void AdjointForm::dJ_initial_condition(
		const State &state,
		const std::vector<Eigen::MatrixXd> &adjoint_nu,
		const std::vector<Eigen::MatrixXd> &adjoint_p,
		Eigen::VectorXd &one_form)
	{
		const int ndof = state.n_bases * state.mesh->dimension();
		one_form.setZero(ndof * 2); // half for initial solution, half for initial velocity

		// \partial_q \hat{J}^0 - p_0^T \partial_q g^v - \mu_0^T \partial_q g^u
		one_form.segment(0, ndof) = -adjoint_nu[0]; // adjoint_nu[0] actually stores adjoint_mu[0]
		one_form.segment(ndof, ndof) = -adjoint_p[0];
	}

	void AdjointForm::dJ_dirichlet_transient(
		const State &state,
		const std::vector<Eigen::MatrixXd> &adjoint_nu,
		const std::vector<Eigen::MatrixXd> &adjoint_p,
		Eigen::VectorXd &one_form)
	{
		const double dt = state.args["time"]["dt"];
		const int time_steps = state.args["time"]["time_steps"];
		const int bdf_order = state.get_bdf_order();

		one_form.setZero(time_steps * state.boundary_nodes.size());
		for (int i = 1; i <= time_steps; ++i)
		{
			const int real_order = std::min(bdf_order, i);
			const double beta_dt = time_integrator::BDF::betas(real_order - 1) * dt;

			for (int b = 0; b < state.boundary_nodes.size(); ++b)
				one_form((i - 1) * state.boundary_nodes.size() + b) = -(1. / beta_dt) * adjoint_p[i](state.boundary_nodes[b]);
		}
	}

	void AdjointForm::dJ_du_step(
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

		if (!j.depend_on_u() && !j.depend_on_gradu())
			return;

		if (spatial_integral_type == SpatialIntegralType::VOLUME)
		{
			auto storage = utils::create_thread_storage(LocalThreadVecStorage(term.size()));
			utils::maybe_parallel_for(n_elements, [&](int start, int end, int thread_id) {
				LocalThreadVecStorage &local_storage = utils::get_local_thread_storage(storage, thread_id);

                Eigen::MatrixXd u, grad_u;
                Eigen::MatrixXd lambda, mu;
                Eigen::MatrixXd result;

                json params = {};
                params["t"] = dt * cur_step;
                params["step"] = cur_step;

				for (int e = start; e < end; ++e)
				{
					assembler::ElementAssemblyValues &vals = local_storage.vals;
					state.ass_vals_cache.compute(e, state.mesh->is_volume(), bases[e], gbases[e], vals);

					const quadrature::Quadrature &quadrature = vals.quadrature;
					local_storage.da = vals.det.array() * quadrature.weights.array();

					const int n_loc_bases_ = int(vals.basis_values.size());
					
					io::Evaluator::interpolate_at_local_vals(e, dim, actual_dim, vals, solution, u, grad_u);
					
					params["elem"] = e;
					params["body_id"] = state.mesh->get_body_id(e);
					result = j.grad_j(state.assembler.lame_params(), quadrature.points, vals.val, u, grad_u, params);
					for (int q = 0; q < result.rows(); q++)
						result.row(q) *= local_storage.da(q);

					for (int i = 0; i < n_loc_bases_; ++i)
					{
						const assembler::AssemblyValues &v = vals.basis_values[i];
						assert(v.global.size() == 1);
						for (int d = 0; d < actual_dim; d++)
						{
							double val = 0;

							// j = j(x, grad u)
							if (result.cols() == grad_u.cols())
							{
								for (int q = 0; q < local_storage.da.size(); ++q)
									val += dot(result.block(q, d * dim, 1, dim), v.grad_t_m.row(q));
							}
							// j = j(x, u)
							else
							{
								for (int q = 0; q < local_storage.da.size(); ++q)
									val += result(q, d) * v.val(q);
							}
							local_storage.vec(v.global[0].index * actual_dim + d) += val;
						}
					}
				}
			});
			for (const LocalThreadVecStorage &local_storage : storage)
				term += local_storage.vec;
		}
		else if (spatial_integral_type == SpatialIntegralType::SURFACE)
		{
			auto storage = utils::create_thread_storage(LocalThreadVecStorage(term.size()));
            utils::maybe_parallel_for(state.total_local_boundary.size(), [&](int start, int end, int thread_id) {
                LocalThreadVecStorage &local_storage = utils::get_local_thread_storage(storage, thread_id);

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

                        normal = normal * vals.jac_it[0]; // assuming linear geometry

						const int n_loc_bases_ = int(vals.basis_values.size());

                        params["elem"] = e;
                        params["body_id"] = state.mesh->get_body_id(e);
                        params["boundary_id"] = state.mesh->get_boundary_id(global_primitive_id);
						result = j.grad_j(state.assembler.lame_params(), points, vals.val, u, grad_u, params);
						for (int q = 0; q < result.rows(); q++)
							result.row(q) *= weights(q);

						for (int j = 0; j < lb.size(); ++j)
						{
							const auto nodes = bases[e].local_nodes_for_primitive(lb.global_primitive_id(j), *state.mesh);

							for (long n = 0; n < nodes.size(); ++n)
							{
								const assembler::AssemblyValues &v = vals.basis_values[nodes(n)];
								assert(v.global.size() == 1);
								for (int d = 0; d < actual_dim; d++)
								{
									double val = 0;

									// j = j(x, grad u)
									if (result.cols() == grad_u.cols())
									{
										for (int q = 0; q < weights.size(); ++q)
										{
											Eigen::Matrix<double, -1, -1, Eigen::RowMajor> grad_phi;
											grad_phi.setZero(actual_dim, dim);
											grad_phi.row(d) = v.grad_t_m.row(q);
											for (int k = d * dim; k < (d + 1) * dim; k++)
												val += result(q, k) * grad_phi(k);
										}
									}
									// j = j(x, u)
									else
									{
										for (int q = 0; q < weights.size(); ++q)
											val += result(q, d) * v.val(q);
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
		else if (spatial_integral_type == SpatialIntegralType::VERTEX_SUM)
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
					params["elem"] = -1;
					params["body_id"] = state.mesh->get_body_id(e);
					params["boundary_id"] = -1;
					Eigen::MatrixXd val;
					j.dj_du(state.assembler.lame_params(), Eigen::MatrixXd::Zero(1, dim) /*Not used*/, g.node, solution.block(g.index * dim, 0, dim, 1).transpose(), Eigen::MatrixXd::Zero(1, dim * actual_dim) /*Not used*/, params, val);
					term.block(g.index * actual_dim, 0, actual_dim, 1) += val.transpose();
					traversed[g.index] = true;
				}
			}
		}
	}

	void AdjointForm::dJ_du_transient(
		const State &state,
		const IntegrableFunctional &j,
		const std::set<int> &interested_ids,
		const SpatialIntegralType spatial_integral_type,
		const std::string &transient_integral_type,
		std::vector<Eigen::VectorXd> &terms)
	{
		const double dt = state.args["time"]["dt"];
		const int time_steps = state.args["time"]["time_steps"];

		terms.resize(time_steps + 1);

		std::vector<double> weights;
		get_transient_quadrature_weights(transient_integral_type, time_steps, dt, weights);
		for (int i = time_steps; i >= 0; --i)
		{
			dJ_du_step(state, j, state.diff_cached[i].u, interested_ids, spatial_integral_type, i, terms[i]);
			terms[i] *= weights[i];
		}
	}

	void AdjointForm::compute_topology_derivative_functional_term(
		const State &state,
		const Eigen::MatrixXd &solution, 
		const IntegrableFunctional &j, 
		const std::set<int> &interested_ids, // either body id or surface id 
		const SpatialIntegralType spatial_integral_type,
		Eigen::VectorXd &term)
	{
		const auto &bases = state.bases;
		const auto &gbases = state.geom_bases();
		const int dim = state.mesh->dimension();

		assert(spatial_integral_type == SpatialIntegralType::VOLUME);

		term.setZero(bases.size());
		if (j.get_name() == "Mass")
		{
			for (int e = 0; e < bases.size(); e++)
			{
				assembler::ElementAssemblyValues vals;
				state.ass_vals_cache.compute(e, state.mesh->is_volume(), bases[e], gbases[e], vals);

				const quadrature::Quadrature &quadrature = vals.quadrature;

				term(e) += (quadrature.weights.array() * vals.det.array()).sum();
			}
		}
		else if (j.get_name() == "Compliance")
		{
			const LameParameters &params = state.assembler.lame_params();
			const auto &density_mat = params.density_mat_;
			const double density_power = params.density_power_;
			for (int e = 0; e < bases.size(); e++)
			{
				assembler::ElementAssemblyValues vals;
				state.ass_vals_cache.compute(e, state.mesh->is_volume(), bases[e], gbases[e], vals);

				const quadrature::Quadrature &quadrature = vals.quadrature;

				for (int q = 0; q < quadrature.weights.size(); q++)
				{
					double lambda, mu;
					params.lambda_mu(quadrature.points.row(q), vals.val.row(q), e, lambda, mu, false);

					Eigen::MatrixXd grad_u_q(dim, dim);
					grad_u_q.setZero();
					for (const auto &v : vals.basis_values)
						for (int d = 0; d < dim; d++)
						{
							double coeff = 0;
							for (const auto &g : v.global)
								coeff += solution(g.index * dim + d) * g.val;
							grad_u_q.row(d) += v.grad_t_m.row(q) * coeff;
						}

					Eigen::MatrixXd stress;
					if (state.formulation() == "LinearElasticity")
						stress = mu * (grad_u_q + grad_u_q.transpose()) + lambda * grad_u_q.trace() * Eigen::MatrixXd::Identity(grad_u_q.rows(), grad_u_q.cols());
					else
						logger().error("Unknown formulation!");

					term(e) += density_power * pow(density_mat(e), density_power - 1) * (stress.array() * grad_u_q.array()).sum() * quadrature.weights(q) * vals.det(q);
				}
			}
		}
		else
			logger().error("Not supported functional type in topology optimization!");
	}

	void AdjointForm::dJ_topology_static(
		const State &state,
		const Eigen::MatrixXd &sol,
		const Eigen::MatrixXd &adjoint,
		const IntegrableFunctional &j,
		const std::set<int> &interested_ids,
		const SpatialIntegralType spatial_integral_type,
		Eigen::VectorXd &one_form)
	{
		Eigen::VectorXd elasticity_term, functional_term;
		state.solve_data.elastic_form->foce_topology_derivative(state.diff_cached[0].u, adjoint, elasticity_term);
		compute_topology_derivative_functional_term(state, state.diff_cached[0].u, j, interested_ids, spatial_integral_type, functional_term);

		one_form = elasticity_term + functional_term;
	}
}