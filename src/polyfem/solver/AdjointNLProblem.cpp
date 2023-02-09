#include "AdjointNLProblem.hpp"
#include <polyfem/utils/MaybeParallelFor.hpp>
#include <polyfem/utils/Timer.hpp>
#include <polyfem/io/OBJWriter.hpp>
#include <igl/boundary_facets.h>
#include <igl/writeOBJ.h>
#include "ShapeParameter.hpp"

namespace polyfem::solver
{
	double AdjointNLProblem::value(const Eigen::VectorXd &x, const bool only_elastic)
	{
		return obj_->value();
	}

	double AdjointNLProblem::value(const Eigen::VectorXd &x)
	{
		return value(x, false);
	}

	void AdjointNLProblem::gradient(const Eigen::VectorXd &x, Eigen::VectorXd &gradv)
	{
		if (cur_grad.size() == x.size())
			gradv = cur_grad;
		else
		{
			gradv.setZero(optimization_dim_);

			{
				POLYFEM_SCOPED_TIMER("adjoint solve", adjoint_solve_time);
				for (int i = 0; i < all_states_.size(); i++)
					all_states_[i]->solve_adjoint_cached(obj_->compute_adjoint_rhs(x, all_states_[i])); // caches inside state
			}

			{
				POLYFEM_SCOPED_TIMER("gradient assembly", grad_assembly_time);
				obj_->gradient(x, gradv);
			}

			cur_grad = gradv;
		}
	}

	bool AdjointNLProblem::smoothing(const Eigen::VectorXd &x, const Eigen::VectorXd &new_x, Eigen::VectorXd &smoothed_x)
	{
		return true;
	}

	bool AdjointNLProblem::remesh(Eigen::VectorXd &x)
	{
		bool remesh = false;
		// TODO: remesh changes size of parameters, need to be careful
		// int cumulative = 0;
		// for (const auto &p : parameters_)
		// {
		// 	Eigen::VectorXd tmp = x.segment(cumulative, p->optimization_dim());
		// 	remesh |= p->remesh(tmp);
		// 	assert(tmp.size() == p->optimization_dim());
		// 	x.segment(cumulative, p->optimization_dim()) = tmp;
		// 	cumulative += p->optimization_dim();
		// }
		return remesh;
	}

	bool AdjointNLProblem::is_step_valid(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const
	{
		return obj_->is_step_valid(x0, x1);
	}

	bool AdjointNLProblem::is_step_collision_free(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const
	{
		return obj_->is_step_collision_free(x0, x1);
	}

	double AdjointNLProblem::max_step_size(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const
	{
		return obj_->max_step_size(x0, x1);
	}

	void AdjointNLProblem::line_search_begin(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1)
	{
		obj_->line_search_begin(x0, x1);
	}

	void AdjointNLProblem::line_search_end()
	{
		return obj_->line_search_end();
	}

	void AdjointNLProblem::post_step(const int iter_num, const Eigen::VectorXd &x)
	{
		iter++;
		return obj_->post_step(iter_num, x);
	}

	void AdjointNLProblem::save_to_file(const Eigen::VectorXd &x0)
	{
		logger().info("Saving iter {}", iter);
		int id = 0;
		if (iter % save_freq != 0)
			return;
		for (const auto &state : all_states_)
		{
			bool save_vtu = false;
			bool save_rest_mesh = false;
			for (const auto &p : parameters_)
				if (p->contains_state(*state))
				{
					save_vtu = true;
					if (p->name() == "shape")
						save_rest_mesh = true;
				}

			std::string vis_mesh_path = state->resolve_output_path(fmt::format("opt_state_{:d}_iter_{:d}.vtu", id, iter));
			std::string rest_mesh_path = state->resolve_output_path(fmt::format("opt_state_{:d}_iter_{:d}.obj", id, iter));
			id++;

			if (!save_vtu)
				continue;
			logger().debug("Save final vtu to file {} ...", vis_mesh_path);

			double tend = state->args.value("tend", 1.0);
			double dt = 1;
			if (!state->args["time"].is_null())
				dt = state->args["time"]["dt"];

			Eigen::MatrixXd sol;
			if (state->args["time"].is_null())
				sol = state->diff_cached[0].u;
			else
				sol = state->diff_cached[state->diff_cached.size() - 1].u;

			state->out_geom.save_vtu(
				vis_mesh_path,
				*state,
				sol,
				Eigen::MatrixXd::Zero(state->n_pressure_bases, 1),
				tend, dt,
				io::OutGeometryData::ExportOptions(state->args, state->mesh->is_linear(), state->problem->is_scalar(), state->solve_export_to_file),
				state->is_contact_enabled(),
				state->solution_frames);

			if (!save_rest_mesh)
				continue;
			logger().debug("Save rest mesh to file {} ...", rest_mesh_path);

			// If shape opt, save rest meshes as well
			Eigen::MatrixXd V;
			Eigen::MatrixXi F;
			state->get_vf(V, F);
			if (state->mesh->dimension() == 3)
				F = igl::boundary_facets<Eigen::MatrixXi, Eigen::MatrixXi>(F);

			io::OBJWriter::write(rest_mesh_path, V, F);
		}
	}

	Eigen::VectorXd AdjointNLProblem::initial_guess() const
	{
		Eigen::VectorXd x;
		x.setZero(full_size());
		int cumulative = 0;
		for (const auto &p : parameters_)
		{
			x.segment(cumulative, p->optimization_dim()) = p->initial_guess();
			cumulative += p->optimization_dim();
		}
		return x;
	}

	Eigen::VectorXd AdjointNLProblem::get_lower_bound(const Eigen::VectorXd &x) const
	{
		Eigen::VectorXd min;
		min.setZero(optimization_dim_);
		int cumulative = 0;
		for (const auto &p : parameters_)
		{
			min.segment(cumulative, p->optimization_dim()) = p->get_lower_bound(x.segment(cumulative, p->optimization_dim()));
			cumulative += p->optimization_dim();
		}
		return min;
	}
	Eigen::VectorXd AdjointNLProblem::get_upper_bound(const Eigen::VectorXd &x) const
	{
		Eigen::VectorXd max;
		max.setZero(optimization_dim_);
		int cumulative = 0;
		for (const auto &p : parameters_)
		{
			max.segment(cumulative, p->optimization_dim()) = p->get_upper_bound(x.segment(cumulative, p->optimization_dim()));
			cumulative += p->optimization_dim();
		}
		return max;
	}

	int AdjointNLProblem::n_inequality_constraints()
	{
		int num = 0;
		for (const auto &p : parameters_)
		{
			num += p->n_inequality_constraints();
		}
		return num;
	}

	double AdjointNLProblem::inequality_constraint_val(const Eigen::VectorXd &x, const int index)
	{
		int num = 0;
		int cumulative = 0;
		for (const auto &p : parameters_)
		{
			num += p->n_inequality_constraints();
			if (num > index)
			{
				num -= p->n_inequality_constraints();
				return p->inequality_constraint_val(x.segment(cumulative, p->optimization_dim()), index - num);
			}
			cumulative += p->optimization_dim();
		}
		log_and_throw_error("Exceeding number of inequality constraints!");
		return 0;
	}

	Eigen::VectorXd AdjointNLProblem::inequality_constraint_grad(const Eigen::VectorXd &x, const int index)
	{
		int num = 0;
		int cumulative = 0;
		Eigen::VectorXd grad;
		grad.setZero(optimization_dim_);
		for (const auto &p : parameters_)
		{
			num += p->n_inequality_constraints();
			if (num > index)
			{
				num -= p->n_inequality_constraints();
				grad.segment(cumulative, p->optimization_dim()) = p->inequality_constraint_grad(x.segment(cumulative, p->optimization_dim()), index - num);
				return grad;
			}
			cumulative += p->optimization_dim();
		}
		log_and_throw_error("Exceeding number of inequality constraints!");
		return grad;
	}

	void AdjointNLProblem::solution_changed(const Eigen::VectorXd &newX)
	{
		// if (cur_x.size() > 0 && abs((newX - cur_x).norm()) < 1e-12)
		// 	return;
		// update to new parameter and check if the new parameter is valid to solve

		for (const auto &v : variables_to_simulation_)
			v->update(newX);

		// solve PDE
		solve_pde();

		cur_x = newX;
	}

	void AdjointNLProblem::solve_pde()
	{
		const int cur_log = all_states_[0]->current_log_level;
		all_states_[0]->set_log_level(static_cast<spdlog::level::level_enum>(solve_log_level)); // log level is global, only need to change in one state
		utils::maybe_parallel_for(all_states_.size(), [&](int start, int end, int thread_id) {
			for (int i = start; i < end; i++)
			{
				auto state = all_states_[i];
				if (active_state_mask[i] || state->diff_cached.size() == 0)
				{
					if (state->diff_cached.size() == 1 && better_initial_guess)
						state->pre_sol = state->diff_cached[0].u;
					state->assemble_rhs();
					state->assemble_stiffness_mat();
					Eigen::MatrixXd sol, pressure; // solution is also cached in state
					state->solve_problem(sol, pressure);
				}
			}
		});
		all_states_[0]->set_log_level(static_cast<spdlog::level::level_enum>(cur_log));

		cur_grad.resize(0);
	}

	Eigen::VectorXd AdjointNLProblem::component_values(const Eigen::VectorXd &x)
	{
		Eigen::VectorXd values;
		values.setZero(obj_->n_objs());
		for (int i = 0; i < obj_->n_objs(); i++)
		{
			values(i) = obj_->get_weight(i) * obj_->get_obj(i)->value();
		}
		return values;
	}
	Eigen::MatrixXd AdjointNLProblem::component_gradients(const Eigen::VectorXd &x)
	{
		Eigen::MatrixXd grads;
		grads.setZero(x.size(), obj_->n_objs());

		for (int i = 0; i < obj_->n_objs(); i++)
		{
			auto obj = obj_->get_obj(i);
			std::vector<Eigen::MatrixXd> adjoints;
			adjoints.reserve(all_states_.size());
			for (auto &state_ptr : all_states_)
				adjoints.push_back(state_ptr->solve_adjoint(obj->compute_adjoint_rhs(*state_ptr)));

			int cumulative = 0;
			for (const auto &p : parameters_)
			{
				Eigen::VectorXd gradv_param = obj_->get_weight(i) * obj->gradient(all_states_, adjoints, *p, x.segment(cumulative, p->optimization_dim()));

				grads.block(cumulative, i, p->optimization_dim(), 1) += gradv_param;
				cumulative += p->optimization_dim();
			}
		}
		return grads;
	}

	bool AdjointNLProblem::verify_gradient(const Eigen::VectorXd &x, const Eigen::VectorXd &gradv)
	{
		if (debug_finite_diff)
		{
			Eigen::VectorXd x2 = x + gradv * finite_diff_eps;
			Eigen::VectorXd x1 = x - gradv * finite_diff_eps;

			solution_changed(x2);
			double J2 = value(x2);

			solution_changed(x1);
			double J1 = value(x1);

			solution_changed(x);

			double fd = (J2 - J1) / 2 / finite_diff_eps;
			double analytic = gradv.squaredNorm();

			bool match = abs(fd - analytic) < 1e-8 || abs(fd - analytic) < 1e-1 * abs(analytic);

			if (match)
				logger().info("step size: {}, finite difference: {}, derivative: {}", finite_diff_eps, fd, analytic);
			else
				logger().error("step size: {}, finite difference: {}, derivative: {}", finite_diff_eps, fd, analytic);

			return match;
		}

		return true;
	}

} // namespace polyfem::solver