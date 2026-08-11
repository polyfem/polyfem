#include "ALSolver.hpp"

#include <polyfem/utils/Logger.hpp>

#include <algorithm>
#include <cmath>
#include <deque>

namespace polyfem::solver
{
	InexactALOptions InexactALOptions::from_json(const json &params)
	{
		InexactALOptions result;
		result.strategy = params.value("strategy", ALStrategy::Legacy);
		const json raw_opts = params.value("inexact", json::object());
		const json opts = raw_opts.is_object() ? raw_opts : json::object();
		result.inner_max_iterations = opts.value("inner_max_iterations", result.inner_max_iterations);
		result.min_iterations = opts.value("min_iterations", result.min_iterations);
		result.energy_window = opts.value("energy_window", result.energy_window);
		result.min_relative_energy_decrease = opts.value("min_relative_energy_decrease", result.min_relative_energy_decrease);
		result.constraint_reduction_target = opts.value("constraint_reduction_target", result.constraint_reduction_target);
		result.max_outer_iterations = opts.value("max_outer_iterations", result.max_outer_iterations);
		result.max_consecutive_failures = opts.value("max_consecutive_failures", result.max_consecutive_failures);

		if (result.inner_max_iterations <= 0 || result.min_iterations < 0
			|| result.min_iterations > result.inner_max_iterations
			|| result.energy_window < 2 || result.max_outer_iterations <= 0
			|| result.max_consecutive_failures < 0)
			log_and_throw_error("Invalid adaptive_inexact augmented-Lagrangian iteration limits");
		if (result.min_relative_energy_decrease < 0
			|| result.constraint_reduction_target < 0
			|| result.constraint_reduction_target > 1)
			log_and_throw_error("Invalid adaptive_inexact augmented-Lagrangian progress tolerances");
		return result;
	}

	ALSolver::ALSolver(
		const std::vector<std::shared_ptr<AugmentedLagrangianForm>> &alagr_form,
		const double initial_al_weight,
		const double scaling,
		const double max_al_weight,
		const double eta_tol,
		const std::function<void(const Eigen::VectorXd &)> &update_barrier_stiffness,
		const StallRestartOptions &stall_opts,
		const std::function<void(const Eigen::VectorXd &)> &on_stall,
		const InexactALOptions &inexact_opts,
		const std::function<bool(const Eigen::VectorXd &, int)> &contact_restart_requested)
		: alagr_forms(alagr_form),
		  initial_al_weight(initial_al_weight),
		  scaling(scaling),
		  max_al_weight(max_al_weight),
		  eta_tol(eta_tol),
		  update_barrier_stiffness(update_barrier_stiffness),
		  stall_opts(stall_opts),
		  on_stall(on_stall),
		  inexact_opts(inexact_opts),
		  contact_restart_requested(contact_restart_requested)
	{
	}

	json ALSolver::consume_subsolve_diagnostics()
	{
		json result = std::move(subsolve_diagnostics_);
		subsolve_diagnostics_ = json::array();
		return result;
	}

	std::string ALSolver::stop_reason_name(const StopReason reason)
	{
		switch (reason)
		{
		case StopReason::Converged:
			return "converged";
		case StopReason::IterationCap:
			return "iteration_cap";
		case StopReason::EnergyStall:
			return "energy_stall";
		case StopReason::LineSearchStall:
			return "line_search_stall";
		case StopReason::ContactStall:
			return "contact_line_search_stall";
		case StopReason::ContactRefresh:
			return "contact_refresh";
		case StopReason::HardContactStall:
			return "hard_contact_stall";
		default:
			return "unknown";
		}
	}

	void ALSolver::record_subsolve(const SubsolveResult &result, const json &extra)
	{
		json report = {
			{"stop_reason", stop_reason_name(result.reason)},
			{"iterations", result.iterations},
			{"initial_energy", result.initial_energy},
			{"final_energy", result.final_energy},
			{"relative_energy_decrease", result.relative_energy_decrease},
			{"last_alpha", result.last_alpha},
			{"last_step", result.last_step}};
		for (auto it = extra.begin(); it != extra.end(); ++it)
			report[it.key()] = it.value();
		logger().info("AL/Newton subsolve: {}", report.dump());
		subsolve_diagnostics_.push_back(std::move(report));
	}

	double ALSolver::constraint_error(const Eigen::VectorXd &x) const
	{
		double error = 0;
		for (const auto &form : alagr_forms)
			error += form->compute_error(x);
		return error;
	}

	bool ALSolver::projected_state_is_valid(
		NLProblem &nl_problem, const Eigen::VectorXd &full_sol,
		Eigen::VectorXd *projected_reduced)
	{
		nl_problem.use_reduced_size();
		Eigen::VectorXd reduced = nl_problem.full_to_reduced(full_sol);
		nl_problem.line_search_begin(full_sol, reduced);
		const bool valid = std::isfinite(nl_problem.value(reduced))
					   && nl_problem.is_step_valid(full_sol, reduced)
					   && nl_problem.is_step_collision_free(full_sol, reduced);
		nl_problem.line_search_end();
		if (valid && projected_reduced != nullptr)
			*projected_reduced = std::move(reduced);
		return valid;
	}

	ALSolver::SubsolveResult ALSolver::minimize_once(
		NLProblem &nl_problem,
		Eigen::VectorXd &tmp_sol,
		const json &nl_solver_params,
		const json &linear_solver,
		const double characteristic_length,
		const std::shared_ptr<NLSolver> &nl_solverin,
		const bool apply_inexact_limits)
	{
		SubsolveResult result;
		const bool detect_contact_stalls = stall_opts.enabled && on_stall != nullptr;
		int small_alpha_count = 0;
		std::deque<double> energies;

		const auto scale = nl_problem.normalize_forms();
		auto nl_solver = nl_solverin == nullptr
					 ? polysolve::nonlinear::Solver::create(
						   nl_solver_params, linear_solver,
						   characteristic_length * scale, logger())
					 : nl_solverin;

		if (direction_filter)
			nl_solver->set_direction_filter(direction_filter);

		if (detect_contact_stalls || apply_inexact_limits || contact_restart_requested)
		{
			nl_solver->set_iteration_callback([&](const polysolve::nonlinear::Criteria &crit) -> bool {
				// PolySolve invokes this callback after accepting a step and before
				// incrementing Criteria::iterations, so its index is zero based.
				const int completed_iterations = int(crit.iterations) + 1;
				result.iterations = completed_iterations;
				result.last_alpha = crit.alpha;
				result.last_step = crit.step;
				if (std::isfinite(crit.energy))
				{
					if (!std::isfinite(result.initial_energy))
						result.initial_energy = crit.energy;
					result.final_energy = crit.energy;
					energies.push_back(crit.energy);
					const int max_window = apply_inexact_limits ? inexact_opts.energy_window : 2;
					while (int(energies.size()) > max_window)
						energies.pop_front();
				}

				if (completed_iterations < (apply_inexact_limits ? inexact_opts.min_iterations : stall_opts.min_iterations))
					return false;

				if (detect_contact_stalls || apply_inexact_limits)
				{
					if (std::isfinite(crit.alpha) && crit.alpha < stall_opts.alpha_threshold)
						++small_alpha_count;
					else
						small_alpha_count = 0;
					if (small_alpha_count >= stall_opts.patience)
					{
						// In bounded AL solves, a persistent small step is an
						// inexact-Newton return: the outer loop must try exact
						// projection and update lambda/penalty. Only explicit
						// contact events request barrier retuning.
						result.reason = apply_inexact_limits
									? StopReason::LineSearchStall
									: StopReason::ContactStall;
						return true;
					}
				}

				const Eigen::VectorXd full = nl_problem.reduced_to_full(tmp_sol);
				if (contact_restart_requested && contact_restart_requested(full, completed_iterations))
				{
					result.reason = StopReason::ContactRefresh;
					return true;
				}

				if (detect_contact_stalls && stall_opts.soft_iteration_limit > 0
					&& completed_iterations >= stall_opts.soft_iteration_limit)
				{
					result.reason = StopReason::ContactRefresh;
					return true;
				}

				if (apply_inexact_limits && completed_iterations >= inexact_opts.inner_max_iterations)
				{
					result.reason = StopReason::IterationCap;
					return true;
				}

				if (apply_inexact_limits && int(energies.size()) == inexact_opts.energy_window)
				{
					const double denom = std::max({1.0, std::abs(energies.front()), std::abs(energies.back())});
					result.relative_energy_decrease = (energies.front() - energies.back()) / denom;
					if (result.relative_energy_decrease < inexact_opts.min_relative_energy_decrease)
					{
						result.reason = StopReason::EnergyStall;
						return true;
					}
				}
				return false;
			});
		}

		try
		{
			nl_solver->minimize(nl_problem, tmp_sol);
			nl_problem.finish();
		}
		catch (const std::runtime_error &e)
		{
			nl_solver->set_iteration_callback(nullptr);
			nl_solver->set_direction_filter(nullptr);
			if (detect_contact_stalls
				&& std::string(e.what()).find("Line search failed") != std::string::npos)
			{
				result.reason = StopReason::HardContactStall;
				return result;
			}
			throw;
		}
		catch (...)
		{
			nl_solver->set_iteration_callback(nullptr);
			nl_solver->set_direction_filter(nullptr);
			throw;
		}

		nl_solver->set_iteration_callback(nullptr);
		nl_solver->set_direction_filter(nullptr);
		if (std::isfinite(result.initial_energy) && std::isfinite(result.final_energy)
			&& !std::isfinite(result.relative_energy_decrease))
		{
			const double denom = std::max({1.0, std::abs(result.initial_energy), std::abs(result.final_energy)});
			result.relative_energy_decrease = (result.initial_energy - result.final_energy) / denom;
		}
		return result;
	}

	void ALSolver::solve_al(
		NLProblem &nl_problem, Eigen::MatrixXd &sol,
		const json &nl_solver_params, const json &linear_solver,
		const double characteristic_length,
		std::shared_ptr<polysolve::nonlinear::Solver> nl_solverin)
	{
		if (inexact_opts.strategy == ALStrategy::AdaptiveInexact)
			solve_al_inexact(nl_problem, sol, nl_solver_params, linear_solver, characteristic_length, nl_solverin);
		else
			solve_al_legacy(nl_problem, sol, nl_solver_params, linear_solver, characteristic_length, nl_solverin);
	}

	void ALSolver::solve_al_legacy(
		NLProblem &nl_problem, Eigen::MatrixXd &sol,
		const json &nl_solver_params, const json &linear_solver,
		const double characteristic_length,
		const std::shared_ptr<NLSolver> &nl_solverin)
	{
		assert(sol.size() == nl_problem.full_size());
		const Eigen::VectorXd initial_sol = sol;
		double al_weight = initial_al_weight;
		const double initial_error = constraint_error(sol);
		for (auto &form : alagr_forms)
			form->set_initial_weight(al_weight);

		while (!projected_state_is_valid(nl_problem, sol))
		{
			if (!(al_weight > 0) || !std::isfinite(al_weight))
				log_and_throw_error("Augmented Lagrangian requires a finite positive initial weight when projection is not feasible");

			nl_problem.use_full_size();
			nl_problem.init(sol);
			update_barrier_stiffness(sol);
			Eigen::VectorXd tmp_sol = sol;
			const Eigen::VectorXd subsolve_initial = sol;
			int contact_restarts = 0;
			int hard_stalls = 0;
			SubsolveResult result;
			do
			{
				result = minimize_once(
					nl_problem, tmp_sol, nl_solver_params, linear_solver,
					characteristic_length, nl_solverin, false);
				sol = tmp_sol;
				record_subsolve(result, {{"phase", "legacy_al"}, {"weight", al_weight}});
				if (!result.is_contact_stop() || projected_state_is_valid(nl_problem, sol))
					break;
				if (++contact_restarts > stall_opts.max_restarts)
					log_and_throw_error("Contact stall persisted after {} restart(s) in legacy AL", stall_opts.max_restarts);
				if (result.reason == StopReason::HardContactStall && ++hard_stalls > 1)
					sol = subsolve_initial;
				on_stall(sol);
				nl_problem.use_full_size();
				nl_problem.init(sol);
				tmp_sol = sol;
			} while (true);

			if (projected_state_is_valid(nl_problem, sol))
			{
				post_subsolve(al_weight);
				break;
			}

			const double current_error = constraint_error(sol);
			const double eta = initial_error > 0 ? 1 - std::sqrt(current_error / initial_error) : 0;
			if (eta < 0)
				sol = initial_sol;
			if (eta < eta_tol && al_weight < max_al_weight)
				al_weight = std::min(al_weight * scaling, max_al_weight);
			for (auto &form : alagr_forms)
				form->update_lagrangian(sol, al_weight);
			post_subsolve(al_weight);
		}
	}

	void ALSolver::solve_al_inexact(
		NLProblem &nl_problem, Eigen::MatrixXd &sol,
		const json &nl_solver_params, const json &linear_solver,
		const double characteristic_length,
		const std::shared_ptr<NLSolver> &nl_solverin)
	{
		assert(sol.size() == nl_problem.full_size());
		double al_weight = initial_al_weight;
		double previous_error = std::sqrt(std::max(0.0, constraint_error(sol)));
		int outer_iterations = 0;
		int consecutive_failures = 0;
		for (auto &form : alagr_forms)
			form->set_initial_weight(al_weight);

		while (!projected_state_is_valid(nl_problem, sol))
		{
			if (!(al_weight > 0) || !std::isfinite(al_weight))
				log_and_throw_error("adaptive_inexact AL requires a finite positive initial weight when projection is not feasible; use initial_weight=\"hessian_scaled\" or a positive number");
			if (outer_iterations >= inexact_opts.max_outer_iterations)
				log_and_throw_error(
					"adaptive_inexact AL reached its {} outer-iteration limit (constraint norm {:g}, weight {:g})",
					inexact_opts.max_outer_iterations, previous_error, al_weight);

			const Eigen::VectorXd outer_initial = sol;
			nl_problem.use_full_size();
			nl_problem.init(sol);
			update_barrier_stiffness(sol);
			Eigen::VectorXd tmp_sol = sol;
			int contact_restarts = 0;
			int hard_stalls = 0;

			while (true)
			{
				SubsolveResult result = minimize_once(
					nl_problem, tmp_sol, nl_solver_params, linear_solver,
					characteristic_length, nl_solverin, true);
				sol = tmp_sol;
				const bool projection_valid = projected_state_is_valid(nl_problem, sol);

				if (projection_valid)
				{
					record_subsolve(result, {
						{"phase", "adaptive_inexact_al"},
						{"outer_iteration", outer_iterations},
						{"contact_restart", contact_restarts},
						{"projection_valid", true},
						{"weight_before", al_weight},
						{"weight_after", al_weight},
						{"multiplier_updated", false}});
					post_subsolve(al_weight);
					return;
				}

				if (result.is_contact_stop())
				{
					record_subsolve(result, {
						{"phase", "adaptive_inexact_al"},
						{"outer_iteration", outer_iterations},
						{"contact_restart", contact_restarts},
						{"projection_valid", false},
						{"weight_before", al_weight},
						{"weight_after", al_weight},
						{"multiplier_updated", false}});
					if (++contact_restarts > stall_opts.max_restarts)
						log_and_throw_error(
							"Contact stall persisted after {} restart(s) in adaptive_inexact AL",
							stall_opts.max_restarts);
					if (result.reason == StopReason::HardContactStall && ++hard_stalls > 1)
						sol = outer_initial;
					on_stall(sol);
					nl_problem.use_full_size();
					nl_problem.init(sol);
					tmp_sol = sol;
					continue;
				}

				nl_problem.use_full_size();
				nl_problem.line_search_begin(sol, sol);
				const bool accepted = std::isfinite(nl_problem.value(sol))
								  && nl_problem.is_step_valid(sol, sol)
								  && nl_problem.is_step_collision_free(sol, sol);
				nl_problem.line_search_end();
				if (!accepted)
				{
					record_subsolve(result, {
						{"phase", "adaptive_inexact_al"},
						{"outer_iteration", outer_iterations},
						{"contact_restart", contact_restarts},
						{"projection_valid", false},
						{"iterate_accepted", false},
						{"weight_before", al_weight},
						{"weight_after", al_weight},
						{"multiplier_updated", false}});
					sol = outer_initial;
					tmp_sol = sol;
					++outer_iterations;
					if (++consecutive_failures >= inexact_opts.max_consecutive_failures)
						log_and_throw_error(
							"adaptive_inexact AL rejected {} consecutive invalid/non-finite subsolves",
							consecutive_failures);
					post_subsolve(al_weight);
					break;
				}

				consecutive_failures = 0;
				const double new_error = std::sqrt(std::max(0.0, constraint_error(sol)));
				const double reduction = previous_error > 0 ? new_error / previous_error : 0;
				for (auto &form : alagr_forms)
					form->update_lagrangian(sol, al_weight);

				const double next_weight = reduction > inexact_opts.constraint_reduction_target
									   ? std::min(al_weight * scaling, max_al_weight)
									   : al_weight;
				for (auto &form : alagr_forms)
					form->set_initial_weight(next_weight);
				record_subsolve(result, {
					{"phase", "adaptive_inexact_al"},
					{"outer_iteration", outer_iterations},
					{"contact_restart", contact_restarts},
					{"projection_valid", false},
					{"iterate_accepted", true},
					{"constraint_before", previous_error},
					{"constraint_after", new_error},
					{"constraint_ratio", reduction},
					{"weight_before", al_weight},
					{"weight_after", next_weight},
					{"multiplier_updated", true}});

				logger().info(
					"adaptive_inexact AL: stop={} constraint={:g}->{:g} ratio={:g} weight={:g}->{:g}",
					stop_reason_name(result.reason), previous_error, new_error,
					reduction, al_weight, next_weight);
				previous_error = new_error;
				al_weight = next_weight;
				++outer_iterations;
				post_subsolve(al_weight);
				break;
			}
		}
	}

	void ALSolver::solve_reduced(
		NLProblem &nl_problem, Eigen::MatrixXd &sol,
		const json &nl_solver_params, const json &linear_solver,
		const double characteristic_length,
		std::shared_ptr<polysolve::nonlinear::Solver> nl_solverin)
	{
		assert(sol.size() == nl_problem.full_size());
		Eigen::VectorXd tmp_sol;
		if (!projected_state_is_valid(nl_problem, sol, &tmp_sol))
			log_and_throw_error("Failed to apply constraint projection; solve with augmented Lagrangian first");

		logger().debug("Successfully applied constraints; solving in reduced space");
		nl_problem.init(sol);
		update_barrier_stiffness(sol);
		const Eigen::VectorXd reduced_initial = tmp_sol;
		int contact_restarts = 0;
		int hard_stalls = 0;
		while (true)
		{
			SubsolveResult result = minimize_once(
				nl_problem, tmp_sol, nl_solver_params, linear_solver,
				characteristic_length, nl_solverin, false);
			record_subsolve(result, {
				{"phase", "reduced"},
				{"contact_restart", contact_restarts},
				{"projection_valid", true}});
			if (!result.is_contact_stop())
				break;
			if (++contact_restarts > stall_opts.max_restarts)
				log_and_throw_error("Contact stall persisted after {} restart(s) in reduced solve", stall_opts.max_restarts);
			if (result.reason == StopReason::HardContactStall && ++hard_stalls > 1)
				tmp_sol = reduced_initial;
			const Eigen::VectorXd full_sol = nl_problem.reduced_to_full(tmp_sol);
			on_stall(full_sol);
			nl_problem.init(full_sol);
			tmp_sol = nl_problem.full_to_reduced(full_sol);
		}

		sol = nl_problem.reduced_to_full(tmp_sol);
		post_subsolve(0);
	}
} // namespace polyfem::solver
