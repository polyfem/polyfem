#include "OptState.hpp"

#include <polyfem/Common.hpp>

#include <polyfem/utils/StringUtils.hpp>
#include <polyfem/utils/par_for.hpp>
#include <polyfem/utils/GeogramUtils.hpp>
#include <polyfem/utils/Logger.hpp>

#include <polyfem/optimization/Optimizations.hpp>
#include <polyfem/optimization/DiffCache.hpp>
#include <polyfem/optimization/AdjointNLProblem.hpp>
#include <polyfem/optimization/BuildFromJson.hpp>
#include <polyfem/optimization/forms/VariableToSimulation.hpp>

#include <polysolve/nonlinear/Solver.hpp>

#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/ostream_sink.h>

#include <Eigen/Core>

#include <memory>
#include <string>
#include <vector>

namespace spdlog::level
{
	NLOHMANN_JSON_SERIALIZE_ENUM(
		spdlog::level::level_enum,
		{{spdlog::level::level_enum::trace, "trace"},
		 {spdlog::level::level_enum::debug, "debug"},
		 {spdlog::level::level_enum::info, "info"},
		 {spdlog::level::level_enum::warn, "warning"},
		 {spdlog::level::level_enum::err, "error"},
		 {spdlog::level::level_enum::critical, "critical"},
		 {spdlog::level::level_enum::off, "off"},
		 {spdlog::level::level_enum::trace, 0},
		 {spdlog::level::level_enum::debug, 1},
		 {spdlog::level::level_enum::info, 2},
		 {spdlog::level::level_enum::warn, 3},
		 {spdlog::level::level_enum::err, 3},
		 {spdlog::level::level_enum::critical, 4},
		 {spdlog::level::level_enum::off, 5}})
}

namespace polyfem
{
	OptState::~OptState()
	{
	}

	OptState::OptState()
	{
		utils::GeogramUtils::instance().initialize();
	}

	void OptState::init_logger(
		const std::string &log_file,
		const spdlog::level::level_enum log_level,
		const spdlog::level::level_enum file_log_level,
		const bool is_quiet)
	{
		std::vector<spdlog::sink_ptr> sinks;

		if (!is_quiet)
		{
			console_sink_ = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
			sinks.emplace_back(console_sink_);
		}

		if (!log_file.empty())
		{
			file_sink_ = std::make_shared<spdlog::sinks::basic_file_sink_mt>(log_file, /*truncate=*/true);
			// Set the file sink separately from the console so it can save all messages
			file_sink_->set_level(file_log_level);
			sinks.push_back(file_sink_);
		}

		init_logger(sinks, log_level);
		spdlog::flush_every(std::chrono::seconds(3));
	}

	void OptState::init_logger(std::ostream &os, const spdlog::level::level_enum log_level)
	{
		std::vector<spdlog::sink_ptr> sinks;
		sinks.emplace_back(std::make_shared<spdlog::sinks::ostream_sink_mt>(os, false));
		init_logger(sinks, log_level);
	}

	void OptState::init_logger(
		const std::vector<spdlog::sink_ptr> &sinks,
		const spdlog::level::level_enum log_level)
	{
		set_adjoint_logger(std::make_shared<spdlog::logger>("adjoint-polyfem", sinks.begin(), sinks.end()));

		// Set the logger at the lowest level, so all messages are passed to the sinks
		adjoint_logger().set_level(spdlog::level::trace);
		set_log_level(log_level);
	}

	void OptState::set_log_level(const spdlog::level::level_enum log_level)
	{
		adjoint_logger().set_level(log_level);
		if (console_sink_)
			console_sink_->set_level(log_level); // Shared by all loggers
	}

	void OptState::init(const json &p_args_in, const bool strict_validation)
	{
		json args_in = p_args_in; // mutable copy
		args = solver::AdjointOptUtils::apply_opt_json_spec(args_in, strict_validation);

		// Save output directory and resolve output paths dynamically
		const std::string output_dir = utils::resolve_path(this->args["output"]["directory"], root_path(), false);
		if (!output_dir.empty())
		{
			std::filesystem::create_directories(output_dir);
		}
		this->output_dir = output_dir;

		std::string out_path_log = this->args["output"]["log"]["path"];
		if (!out_path_log.empty())
		{
			out_path_log = utils::resolve_path(out_path_log, root_path(), false);
		}

		init_logger(
			out_path_log,
			this->args["output"]["log"]["level"],
			this->args["output"]["log"]["file_level"],
			this->args["output"]["log"]["quiet"]);

		adjoint_logger().info("Saving adjoint output to {}", output_dir);

		const int thread_in = this->args["solver"]["max_threads"];
		utils::NThread::get().set_num_threads(thread_in);
	}

	void OptState::create_states(const int max_threads)
	{
		states = from_json::build_states(
			root_path(),
			args["states"],
			max_threads <= 0 ? std::numeric_limits<unsigned int>::max() : max_threads);

		diff_caches.resize(states.size());
		for (auto &diff_cache : diff_caches)
		{
			diff_cache = std::make_shared<DiffCache>();
		}

		check_unsupported();

		utils::GeogramUtils::instance().set_logger(adjoint_logger());
	}

	void OptState::check_unsupported() const
	{
		for (int i = 0; i < states.size(); ++i)
		{
			const State &state = *states[i];

			// No transient linear support.
			if (state.problem->is_time_dependent() && state.is_problem_linear())
			{
				log_and_throw_adjoint_error(
					"State {}: transient linear problem is not supported in optimization.", i);
			}

			if (state.is_contact_enabled())
			{
				// No non-convergent contact formulation support.
				if (!state.args["contact"]["use_gcp_formulation"].get<bool>()
					&& !state.args["contact"]["use_convergent_formulation"].get<bool>())
				{
					log_and_throw_adjoint_error(
						"State {}: non-convergent contact formulation is not supported in optimization.", i);
				}

				// No non-const barrier stiffness support.
				if (state.args["/solver/contact/barrier_stiffness"_json_pointer].is_string())
				{
					log_and_throw_adjoint_error(
						"State {}: only constant barrier stiffness is supported in optimization.", i);
				}
			}

			// No non-const boundary support.
			if (state.args.contains("boundary_conditions") && state.args["boundary_conditions"].contains("rhs"))
			{
				const json &rhs = state.args["boundary_conditions"]["rhs"];
				if (rhs.is_string() || (rhs.is_array() && rhs.size() > 0 && rhs[0].is_string()))
				{
					log_and_throw_adjoint_error(
						"State {}: only constant rhs over space is supported in optimization.", i);
				}
			}

			// No high order geometric basis support.
			for (const auto &element_bases : state.geom_bases())
			{
				for (const auto &basis : element_bases.bases)
				{
					if (basis.order() > 1)
					{
						log_and_throw_adjoint_error(
							"State {}: high-order geometry basis is not supported in optimization.", i);
					}
				}
			}
		}
	}

	void OptState::init_variables()
	{
		/* DOFS */
		ndof = 0;
		for (const auto &arg : args["parameters"])
		{
			int size = solver::AdjointOptUtils::compute_variable_size(arg, states);
			ndof += size;
			variable_sizes.push_back(size);
		}

		/* variable to simulations */
		variable_to_simulations = from_json::build_variable_to_simulation_group(
			args["variable_to_simulation"], states, diff_caches, variable_sizes);
	}

	void OptState::create_problem()
	{
		/* forms */
		std::shared_ptr<solver::AdjointForm> obj = from_json::build_form(
			args["functionals"], variable_to_simulations, states, diff_caches);

		/* stopping conditions */
		std::vector<std::shared_ptr<solver::AdjointForm>> stopping_conditions;
		for (const auto &arg : args["stopping_conditions"])
			stopping_conditions.push_back(
				from_json::build_form(arg, variable_to_simulations, states, diff_caches));

		nl_problem = std::make_unique<solver::AdjointNLProblem>(
			obj, stopping_conditions, variable_to_simulations, states, diff_caches, args);
	}

	void OptState::initial_guess(Eigen::VectorXd &x)
	{
		x = solver::AdjointOptUtils::inverse_evaluation(args["parameters"], ndof, variable_sizes, variable_to_simulations);

		variable_to_simulations.update(x);
	}

	double OptState::eval(Eigen::VectorXd &x) const
	{
		nl_problem->solution_changed(x);
		return nl_problem->value(x);
	}

	void OptState::solve(Eigen::VectorXd &x)
	{
		auto nl_solver = solver::AdjointOptUtils::make_nl_solver(
			args["solver"]["nonlinear"],
			args["solver"]["linear"],
			args["solver"]["advanced"]["characteristic_length"]);
		nl_problem->normalize_forms();
		nl_solver->minimize(*nl_problem, x);
	}
} // namespace polyfem
