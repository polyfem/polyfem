#include "AdjointNLProblem.hpp"
#include <polyfem/utils/MaybeParallelFor.hpp>
#include <polyfem/utils/Timer.hpp>

namespace polyfem::solver
{
	double AdjointNLProblem::target_value(const Eigen::VectorXd &x)
	{
		// TODO: user specify selection of functionals to be target
		return obj_->value();
	}

	double AdjointNLProblem::value(const Eigen::VectorXd &x, const bool only_elastic)
	{
		return obj_->value();
	}

	double AdjointNLProblem::value(const Eigen::VectorXd &x)
	{
		return value(x, false);
	}

	void AdjointNLProblem::target_gradient(const Eigen::VectorXd &x, Eigen::VectorXd &gradv)
	{
		gradient(x, gradv, false);
	}

	void AdjointNLProblem::gradient(const Eigen::VectorXd &x, Eigen::VectorXd &gradv)
	{
		gradient(x, gradv, false);
	}

	void AdjointNLProblem::gradient(const Eigen::VectorXd &x, Eigen::VectorXd &gradv, const bool only_elastic)
	{
		if (cur_grad.size() == x.size())
			gradv = cur_grad;
		else
		{
			int cumulative = 0;
			gradv.setZero(optimization_dim_);

			{
				POLYFEM_SCOPED_TIMER("adjoint solve", adjoint_solve_time);
				for (auto &state_ptr : all_states_)
					state_ptr->solve_adjoint(obj_->compute_adjoint_rhs(*state_ptr));
			}

			{
				POLYFEM_SCOPED_TIMER("gradient assembly", grad_assembly_time);
				for (const auto &p : parameters_)
				{
					Eigen::VectorXd gradv_param = obj_->gradient(all_states_, *p);
					
					gradv.segment(cumulative, p->optimization_dim()) += p->map_grad(x.segment(cumulative, p->optimization_dim()), gradv_param);
					cumulative += p->optimization_dim();
				}
			}

			cur_grad = gradv;
		}
	}

	void AdjointNLProblem::smoothing(const Eigen::VectorXd &x, Eigen::VectorXd &new_x)
	{
		int cumulative = 0;
		for (const auto &p : parameters_)
		{
			Eigen::VectorXd tmp = new_x.segment(cumulative, p->optimization_dim());
			p->smoothing(x.segment(cumulative, p->optimization_dim()), tmp);
			assert(tmp.size() == p->optimization_dim());
			new_x.segment(cumulative, p->optimization_dim()) = tmp;
			cumulative += p->optimization_dim();
		}
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
		int cumulative = 0;
		bool is_valid = true;
		for (const auto &p : parameters_)
		{
			is_valid &= p->is_step_valid(x0.segment(cumulative, p->optimization_dim()), x1.segment(cumulative, p->optimization_dim()));
			cumulative += p->optimization_dim();
		}
		return is_valid;
	}

	bool AdjointNLProblem::is_intersection_free(const Eigen::VectorXd &x) const
	{
		int cumulative = 0;
		bool is_valid = true;
		for (const auto &p : parameters_)
		{
			is_valid &= p->is_intersection_free(x.segment(cumulative, p->optimization_dim()));
			cumulative += p->optimization_dim();
		}
		return is_valid;
	}

	bool AdjointNLProblem::is_step_collision_free(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const
	{
		int cumulative = 0;
		bool is_valid = true;
		for (const auto &p : parameters_)
		{
			is_valid &= p->is_step_collision_free(x0.segment(cumulative, p->optimization_dim()), x1.segment(cumulative, p->optimization_dim()));
			cumulative += p->optimization_dim();
		}
		return is_valid;
	}

	double AdjointNLProblem::max_step_size(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const
	{
		// TODO: this was a number multiplied to the descent direction to take either a larger or smaller step, so now it's better to be a vector of numbers?
		return 1;
	}

	void AdjointNLProblem::line_search_begin(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1)
	{
		int cumulative = 0;
		for (const auto &p : parameters_)
		{
			p->line_search_begin(x0.segment(cumulative, p->optimization_dim()), x1.segment(cumulative, p->optimization_dim()));
			cumulative += p->optimization_dim();
		}
	}

	void AdjointNLProblem::line_search_end()
	{

		int cumulative = 0;
		for (const auto &p : parameters_)
		{
			p->line_search_end();
			cumulative += p->optimization_dim();
		}
	}

	void AdjointNLProblem::post_step(const int iter_num, const Eigen::VectorXd &x)
	{
		int cumulative = 0;
		for (const auto &p : parameters_)
		{
			p->post_step(iter_num, x.segment(cumulative, p->optimization_dim()));
			cumulative += p->optimization_dim();
		}
		iter++;
	}

	void AdjointNLProblem::save_to_file(const Eigen::VectorXd &x0)
	{
		logger().info("Iter {}", iter);
		int id = 0;
		for (const auto &state : all_states_)
		{
			std::string vis_mesh_path = state->resolve_output_path(fmt::format("opt_state_{:d}_iter_{:d}.vtu", id, iter));
			logger().debug("Save to file {} ...", vis_mesh_path);
			id++;

			double tend = state->args.value("tend", 1.0);
			double dt = 1;
			if (!state->args["time"].is_null())
				dt = state->args["time"]["dt"];

			state->out_geom.export_data(
				*state,
				state->diff_cached[0].u,
				Eigen::MatrixXd::Zero(state->n_pressure_bases, 1),
				!state->args["time"].is_null(),
				tend, dt,
				io::OutGeometryData::ExportOptions(state->args, state->mesh->is_linear(), state->problem->is_scalar(), state->solve_export_to_file),
				vis_mesh_path,
				"", // nodes_path,
				"", // solution_path,
				"", // stress_path,
				"", // mises_path,
				state->is_contact_enabled(), state->solution_frames);
			
			// TODO: if shape opt, save rest meshes as well
		}
	}

	Eigen::VectorXd AdjointNLProblem::get_lower_bound(const Eigen::VectorXd &x) const
	{
		Eigen::VectorXd min(optimization_dim_);
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
		Eigen::VectorXd max(optimization_dim_);
		int cumulative = 0;
		for (const auto &p : parameters_)
		{
			max.segment(cumulative, p->optimization_dim()) = p->get_upper_bound(x.segment(cumulative, p->optimization_dim()));
			cumulative += p->optimization_dim();
		}
		return max;
	}

	Eigen::VectorXd AdjointNLProblem::force_inequality_constraint(const Eigen::VectorXd &x0, const Eigen::VectorXd &dx)
	{
		Eigen::VectorXd newX;
		newX.setZero(optimization_dim_);
		int cumulative = 0;
		for (const auto &p : parameters_)
		{
			newX.segment(cumulative, p->optimization_dim()) = p->force_inequality_constraint(x0.segment(cumulative, p->optimization_dim()), dx.segment(cumulative, p->optimization_dim()));
			cumulative += p->optimization_dim();
		}
		return newX;
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
		Eigen::VectorXd grad(optimization_dim_);
		grad.setZero();
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
		// if solution was not changed, no action is needed
		if (cur_x.size() == newX.size() && cur_x == newX)
			return;

		// update to new parameter and check if the new parameter is valid to solve
		bool solve = true;
		for (const auto &p : parameters_)
		{
			solve &= p->pre_solve(newX);
		}

		// solve PDE
		if (solve)
		{
			solve_pde();

			// post actions after solving PDE
			for (const auto &p : parameters_)
			{
				p->post_solve(newX);
			}

			cur_x = newX;
		}
	}

	void AdjointNLProblem::solve_pde()
	{
		const int cur_log = all_states_[0]->current_log_level;
		all_states_[0]->set_log_level(static_cast<spdlog::level::level_enum>(solve_log_level)); // log level is global, only need to change in one state
		utils::maybe_parallel_for(all_states_.size(), [&](int start, int end, int thread_id) {
			for (int i = start; i < end; i++)
			{
				const auto &state = all_states_[i];
				state->assemble_rhs();
				state->assemble_stiffness_mat();
				Eigen::MatrixXd sol, pressure; // solution is also cached in state
				state->solve_problem(sol, pressure);
			}
		});
		all_states_[0]->set_log_level(static_cast<spdlog::level::level_enum>(cur_log));

		cur_grad.resize(0);
	}

} // namespace polyfem