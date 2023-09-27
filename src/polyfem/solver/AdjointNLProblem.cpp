#include "AdjointNLProblem.hpp"

#include <polyfem/solver/forms/adjoint_forms/CompositeForm.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/MaybeParallelFor.hpp>
#include <polyfem/utils/Timer.hpp>
#include <polyfem/io/OBJWriter.hpp>
#include <polyfem/State.hpp>
#include <igl/boundary_facets.h>
#include <igl/writeOBJ.h>

#include <polyfem/solver/NLProblem.hpp>

namespace polyfem::solver
{
	AdjointNLProblem::AdjointNLProblem(std::shared_ptr<CompositeForm> composite_form, const std::vector<std::shared_ptr<VariableToSimulation>> &variables_to_simulation, const std::vector<std::shared_ptr<State>> &all_states, const json &args)
		: FullNLProblem({composite_form}),
		  composite_form_(composite_form),
		  variables_to_simulation_(variables_to_simulation),
		  all_states_(all_states),
		  solve_log_level(args["output"]["solve_log_level"]),
		  save_freq(args["output"]["save_frequency"]),
		  solve_in_parallel(args["solver"]["advanced"]["solve_in_parallel"]),
		  better_initial_guess(args["solver"]["advanced"]["better_initial_guess"])
	{
		cur_grad.setZero(0);

		solve_in_order.clear();
		if (args["solver"]["advanced"]["solve_in_order"].size() > 0)
		{
			for (int i : args["solver"]["advanced"]["solve_in_order"])
				solve_in_order.push_back(i);
			
			if (solve_in_parallel)
				logger().error("Cannot solve both in order and in parallel, ignoring the order!");
			
			assert(solve_in_order.size() == all_states.size());
		}
		else
		{
			for (int i = 0; i < all_states_.size(); i++)
				solve_in_order.push_back(i);
		}

		active_state_mask.assign(all_states_.size(), false);
		for (int i = 0; i < all_states_.size(); i++)
		{
			for (const auto &v2sim : variables_to_simulation_)
			{
				for (const auto &state : v2sim->get_states())
				{
					if (all_states_[i].get() == state.get())
					{
						active_state_mask[i] = true;
						break;
					}
				}
			}
		}
	}

	AdjointNLProblem::AdjointNLProblem(std::shared_ptr<CompositeForm> composite_form, const std::vector<std::shared_ptr<AdjointForm>> stopping_conditions, const std::vector<std::shared_ptr<VariableToSimulation>> &variables_to_simulation, const std::vector<std::shared_ptr<State>> &all_states, const json &args) : AdjointNLProblem(composite_form, variables_to_simulation, all_states, args)
	{
		stopping_conditions_ = stopping_conditions;
	}

	void AdjointNLProblem::hessian(const Eigen::VectorXd &x, StiffnessMatrix &hessian)
	{
		log_and_throw_error("Hessian not supported!");
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
			state->get_vertices(V);
			state->get_elements(F);
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
			const auto cur_log_level = logger().level();
			all_states_[0]->set_log_level(static_cast<spdlog::level::level_enum>(solve_log_level)); // log level is global, only need to change in one state
			for (const auto &state : all_states_)
				state->build_basis();
			all_states_[0]->set_log_level(cur_log_level);
		}

		// solve PDE
		solve_pde();

		composite_form_->solution_changed(newX);
	}

	void AdjointNLProblem::solve_pde()
	{
		const auto cur_log_level = logger().level();
		all_states_[0]->set_log_level(static_cast<spdlog::level::level_enum>(solve_log_level)); // log level is global, only need to change in one state
		
		if (solve_in_parallel)
		{
			logger().info("Run simulations in parallel...");

			utils::maybe_parallel_for(all_states_.size(), [&](int start, int end, int thread_id) {
				for (int i = start; i < end; i++)
				{
					auto state = all_states_[i];
					if (active_state_mask[i] || state->diff_cached.size() == 0)
					{
						state->assemble_rhs();
						state->assemble_mass_mat();
						Eigen::MatrixXd sol, pressure; // solution is also cached in state
						state->solve_problem(sol, pressure);
					}
				}
			});
		}
		else
		{
			Eigen::MatrixXd sol, pressure; // solution is also cached in state
			for (int i : solve_in_order)
			{
				auto state = all_states_[i];
				if (active_state_mask[i] || state->diff_cached.size() == 0)
				{
					state->assemble_rhs();
					state->assemble_mass_mat();

					state->solve_problem(sol, pressure);
				}
			}
		}

		all_states_[0]->set_log_level(cur_log_level);

		cur_grad.resize(0);
	}

	bool AdjointNLProblem::stop(const TVector &x)
	{
		if (stopping_conditions_.size() == 0)
			return false;
		
		for (auto &obj : stopping_conditions_)
		{
			obj->solution_changed(x);
			if (obj->value(x) > 0)
				return false;
		}
		return true;
	}

} // namespace polyfem::solver