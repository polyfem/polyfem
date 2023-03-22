#include "AdjointNLProblem.hpp"

#include <polyfem/utils/MaybeParallelFor.hpp>
#include <polyfem/utils/Timer.hpp>
#include <polyfem/io/OBJWriter.hpp>
#include <igl/boundary_facets.h>
#include <igl/writeOBJ.h>

namespace polyfem::solver
{
	AdjointNLProblem::AdjointNLProblem(std::shared_ptr<CompositeForm> composite_form, const std::vector<std::shared_ptr<VariableToSimulation>> &variables_to_simulation, const std::vector<std::shared_ptr<State>> &all_states, const json &args)
		: FullNLProblem({composite_form}),
		  composite_form_(composite_form),
		  variables_to_simulation_(variables_to_simulation),
		  all_states_(all_states),
		  solve_log_level(args["output"]["solve_log_level"]),
		  save_freq(args["output"]["save_frequency"])
	{
		cur_grad.setZero(0);

		active_state_mask.assign(all_states_.size(), false);
		for (const auto &v2sim : variables_to_simulation_)
			for (int i = 0; i < all_states_.size(); i++)
				if (all_states_[i].get() == &(v2sim->get_state()))
					active_state_mask[i] = true;
	}

	double AdjointNLProblem::value(const Eigen::VectorXd &x)
	{
		return composite_form_->value(x);
	}

	void AdjointNLProblem::gradient(const Eigen::VectorXd &x, Eigen::VectorXd &gradv)
	{
		if (cur_grad.size() == x.size())
			gradv = cur_grad;
		else
		{
			gradv.setZero(x.size());

			{
				POLYFEM_SCOPED_TIMER("adjoint solve");
				for (int i = 0; i < all_states_.size(); i++)
					all_states_[i]->solve_adjoint_cached(composite_form_->compute_adjoint_rhs(x, *all_states_[i])); // caches inside state
			}

			{
				POLYFEM_SCOPED_TIMER("gradient assembly");
				composite_form_->first_derivative(x, gradv);
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
		return composite_form_->is_step_valid(x0, x1);
	}

	bool AdjointNLProblem::is_step_collision_free(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const
	{
		return composite_form_->is_step_collision_free(x0, x1);
	}

	double AdjointNLProblem::max_step_size(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const
	{
		return composite_form_->max_step_size(x0, x1);
	}

	void AdjointNLProblem::line_search_begin(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1)
	{
		composite_form_->line_search_begin(x0, x1);
	}

	void AdjointNLProblem::line_search_end()
	{
		composite_form_->line_search_end();
	}

	void AdjointNLProblem::post_step(const int iter_num, const Eigen::VectorXd &x)
	{
		iter++;
		composite_form_->post_step(iter_num, x);
	}

	void AdjointNLProblem::save_to_file(const Eigen::VectorXd &x0)
	{
		logger().info("Saving iter {}", iter);
		int id = 0;
		if (iter % save_freq != 0)
			return;
		for (const auto &state : all_states_)
		{
			bool save_vtu = true;
			bool save_rest_mesh = true;
			// for (const auto &p : parameters_)
			// 	if (p->contains_state(*state))
			// 	{
			// 		save_vtu = true;
			// 		if (p->name() == "shape")
			// 			save_rest_mesh = true;
			// 	}

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
				sol = state->diff_cached.u(0);
			else
				sol = state->diff_cached.u(state->diff_cached.size() - 1);

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

	void AdjointNLProblem::solution_changed(const Eigen::VectorXd &newX)
	{
		bool need_rebuild_basis = false;

		// update to new parameter and check if the new parameter is valid to solve
		for (const auto &v : variables_to_simulation_)
		{
			v->update(newX);
			if (v->get_parameter_type() == ParameterType::Shape)
				need_rebuild_basis = true;
		}

		if (need_rebuild_basis)
		{
			const int cur_log = all_states_[0]->current_log_level;
			all_states_[0]->set_log_level(static_cast<spdlog::level::level_enum>(solve_log_level)); // log level is global, only need to change in one state
			for (const auto &state : all_states_)
				state->build_basis();
			all_states_[0]->set_log_level(static_cast<spdlog::level::level_enum>(cur_log));
		}

		// solve PDE
		solve_pde();

		composite_form_->solution_changed(newX);
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
		values.setZero(composite_form_->n_objs());
		// for (int i = 0; i < composite_form_->n_objs(); i++)
		// {
		// 	values(i) = composite_form_->get_weight(i) * composite_form_->get_obj(i)->value();
		// }
		return values;
	}
	Eigen::MatrixXd AdjointNLProblem::component_gradients(const Eigen::VectorXd &x)
	{
		Eigen::MatrixXd grads;
		grads.setZero(x.size(), composite_form_->n_objs());

		// for (int i = 0; i < composite_form_->n_objs(); i++)
		// {
		// 	auto obj = composite_form_->get_obj(i);
		// 	std::vector<Eigen::MatrixXd> adjoints;
		// 	adjoints.reserve(all_states_.size());
		// 	for (auto &state_ptr : all_states_)
		// 		adjoints.push_back(state_ptr->solve_adjoint(obj->compute_adjoint_rhs(*state_ptr)));

		// 	int cumulative = 0;
		// 	for (const auto &p : parameters_)
		// 	{
		// 		Eigen::VectorXd gradv_param = composite_form_->get_weight(i) * obj->gradient(all_states_, adjoints, *p, x.segment(cumulative, p->optimization_dim()));

		// 		grads.block(cumulative, i, p->optimization_dim(), 1) += gradv_param;
		// 		cumulative += p->optimization_dim();
		// 	}
		// }
		return grads;
	}

} // namespace polyfem::solver