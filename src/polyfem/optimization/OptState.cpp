#include "OptState.hpp"

#include <polyfem/Common.hpp>

#include <polyfem/utils/StringUtils.hpp>
#include <polyfem/utils/par_for.hpp>
#include <polyfem/utils/GeogramUtils.hpp>
#include <polyfem/utils/Logger.hpp>

#include <polyfem/optimization/Optimizations.hpp>
#include <polyfem/optimization/AdjointTools.hpp>
#include <polyfem/optimization/DiffCache.hpp>
#include <polyfem/optimization/AdjointNLProblem.hpp>
#include <polyfem/optimization/BuildFromJson.hpp>
#include <polyfem/optimization/var2sims/VariableToSimulationGroup.hpp>
#include <polyfem/optimization/var2sims/ShapeVariableToSimulation.hpp>

#include <polyfem/io/MshWriter.hpp>
#include <polyfem/mesh/remesh/MMGRemesh.hpp>

#include <polysolve/nonlinear/Solver.hpp>

#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/ostream_sink.h>

#include <Eigen/Core>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
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
	namespace
	{
		bool is_failed_status(const polysolve::nonlinear::Status status)
		{
			using Status = polysolve::nonlinear::Status;
			return status == Status::NanEncountered
				   || status == Status::NotDescentDirection
				   || status == Status::LineSearchFailed
				   || status == Status::UpdateDirectionFailed
				   || status == Status::NotStarted
				   || status == Status::Continue;
		}

		std::string parse_remeshing_trigger(const json &args)
		{
			const json &remeshing = args["remeshing"];
			if (remeshing.is_null())
				return "none";

			const json &trigger = remeshing["trigger"];
			if (trigger.is_null())
				return "none";

			const bool periodic = trigger.contains("periodic") && trigger["periodic"].is_object();
			const bool scaled_jacobian = trigger.contains("scaled_jacobian") && trigger["scaled_jacobian"].is_object();
			if (periodic && scaled_jacobian)
				return "multiple";
			if (periodic)
				return "periodic";
			if (scaled_jacobian)
				return "scaled_jacobian";
			return "none";
		}

		/// @brief Read forward sim state json config from file.
		/// @param root_path
		/// @param args Json array of state containing "path" field.
		std::vector<json> load_state_jsons(const std::string &root_path, const json &args)
		{
			std::vector<json> result;
			for (int i = 0; i < args.size(); ++i)
			{
				json state_args;
				const std::string state_path = utils::resolve_path(args[i]["path"], root_path, false);
				std::ifstream file(state_path);
				if (!file.is_open())
					log_and_throw_adjoint_error("Can't find json for varform::DifferentiableVarForm {}", i);
				file >> state_args;
				state_args["root_path"] = state_path;
				result.push_back(std::move(state_args));
			}
			return result;
		}

		void validate_remeshing_json(const json &args)
		{
			const std::string trigger = parse_remeshing_trigger(args);
			if (trigger == "multiple")
				log_and_throw_error("Optimization remeshing requires exactly one trigger: periodic or scaled_jacobian.");
			const json &remeshing = args["remeshing"];
			const int state = remeshing["state"];
			if (state >= args["states"].size())
				log_and_throw_error("Optimization remeshing state index {} is out of range.", state);

			for (const auto &variable : args["variable_to_simulation"])
			{
				if (variable["type"] != "shape")
					continue;

				const json &variable_state = variable["state"];
				const bool targets_state = variable_state.is_array()
											   ? std::find(variable_state.begin(), variable_state.end(), state) != variable_state.end()
											   : variable_state.get<int>() == state;
				if (!targets_state)
					continue;

				const json &selection = variable["active_geometry_nodes"];
				const bool selects_all_nodes = selection.is_array() && selection.empty();
				if (selects_all_nodes || selection.is_object())
					continue;

				log_and_throw_error(
					"Optimization remeshing changes mesh vertex numbering. "
					"Shape variable {} targeting state {} must use a geometry-based "
					"active_geometry_nodes selector (interior, boundary, or "
					"boundary_excluding_surface), or an empty selector for all nodes.",
					variable.value("name", "shape"), state);
			}
		}

		void validate_remeshing_geometry(const json &state_args, const int geometry)
		{
			if (geometry < 0 || geometry >= state_args["geometry"].size())
				log_and_throw_error("Optimization remeshing geometry index {} is out of range.", geometry);
			if (state_args["geometry"][geometry].value("is_obstacle", false))
				log_and_throw_error("Optimization remeshing geometry {} is an obstacle.", geometry);
		}

		/// @brief Compute minimal scaled jacobian determinant.
		double minimum_scaled_jacobian(
			const Eigen::MatrixXd &vertices,
			const Eigen::MatrixXi &elements)
		{
			Eigen::VectorXd quality;
			solver::AdjointTools::scaled_jacobian(vertices, elements, quality);
			return quality.minCoeff();
		}

		/// @brief Compute minimal scaled jacobian determinant.
		double minimum_scaled_jacobian(const varform::DifferentiableVarForm &varform)
		{
			Eigen::MatrixXd vertices;
			Eigen::MatrixXi elements;
			varform.get_vertices(vertices);
			varform.get_elements(elements);
			return minimum_scaled_jacobian(vertices, elements);
		}

		/// @brief Remesh and write result to file.
		/// @param opt_state Optimization state.
		/// @param state Forward simulation state.
		/// @param body_id Body to remesh.
		/// @param path Output path.
		/// @param quality_threshold Optional scaled jacobian mesh quality check.
		bool remesh_and_write(
			const OptState &opt_state,
			const int state,
			const int body_id,
			const std::filesystem::path &path,
			const std::optional<double> quality_threshold)
		{
			mesh::MmgOptions options;
			options.optim = true;

			Eigen::MatrixXd vertices;
			Eigen::MatrixXi elements;
			opt_state.varforms[state]->get_vertices(vertices);
			opt_state.varforms[state]->get_elements(elements);

			Eigen::MatrixXd remeshed_vertices;
			Eigen::MatrixXi remeshed_boundary;
			Eigen::MatrixXi remeshed_elements;
			bool has_shape_variable = false;
			for (const auto &var2sim : opt_state.variable_to_simulations.data)
			{
				const auto shape_v2s = std::dynamic_pointer_cast<solver::ShapeVariableToSimulation>(var2sim);
				if (!shape_v2s || !shape_v2s->affects_varform(*opt_state.varforms[state]))
					continue;
				has_shape_variable = true;

				for (int other_state = 0; other_state < opt_state.varforms.size(); ++other_state)
				{
					if (other_state != state
						&& shape_v2s->affects_varform(*opt_state.varforms[other_state]))
					{
						log_and_throw_error(
							"Optimization remeshing cannot remesh state {} while its shape variable also affects state {}. This is not supported currently.",
							state, other_state);
					}
				}
			}
			if (!has_shape_variable)
			{
				logger().error("Can not remesh state {} because it's not affected by shape variables. No reason to remesh if you are not doing shape optimization.", state);
				return false;
			}

			const mesh::Mesh &mesh = opt_state.varforms[state]->get_mesh();
			bool success = false;
			if (mesh.dimension() == 2)
			{
				success = mesh::remesh_2d(
					vertices, elements, remeshed_vertices, remeshed_elements,
					options);
			}
			else
			{
				success = mesh::remesh_3d(
					vertices, elements, remeshed_vertices, remeshed_boundary,
					remeshed_elements, options);
			}
			if (!success)
			{
				return false;
			}

			const double input_quality = minimum_scaled_jacobian(vertices, elements);
			const double output_quality = minimum_scaled_jacobian(remeshed_vertices, remeshed_elements);
			logger().info(
				"Optimization remeshing minimum scaled Jacobian: {} -> {}.",
				input_quality, output_quality);
			// If trigger mode is scaled jacobian, check quality immediately after remeshing.
			if (quality_threshold.has_value() && output_quality <= *quality_threshold)
			{
				log_and_throw_error(
					"MMG remeshing failed to satisfy minimum scaled Jacobian threshold: "
					"input={}, output={}, required > {}. This is because MMG use a different convergence criteria, please switch remeshing trigger to periodic.",
					input_quality, output_quality, *quality_threshold);
			}

			io::MshWriter::write(
				path.string(), remeshed_vertices, remeshed_elements,
				std::vector<int>(remeshed_elements.rows(), body_id),
				opt_state.varforms[state]->get_mesh().is_volume(), false);
			return true;
		}
	} // namespace

	OptState::~OptState()
	{
	}

	OptState::OptState()
	{
		utils::GeogramUtils::instance().initialize();
	}

	int OptState::run(json input_args, const bool strict_validation)
	{
		input_args = solver::AdjointOptUtils::apply_opt_json_spec(input_args, strict_validation);

		std::string mode = parse_remeshing_trigger(input_args);
		bool remeshing_enabled = mode != "none";
		if (remeshing_enabled)
			validate_remeshing_json(input_args);

		std::filesystem::path output_root = utils::resolve_path(
			input_args["output"]["directory"], input_args["root_path"], false);
		int max_restarts = remeshing_enabled
							   ? input_args["remeshing"]["max_restarts"].get<int>()
							   : 0;
		state_args = load_state_jsons(input_args["root_path"], input_args["states"]);

		int state = 0;
		int geometry = 0;
		int body_id = 0;
		if (remeshing_enabled)
		{
			state = input_args["remeshing"]["state"].get<int>();
			geometry = input_args["remeshing"]["geometry"].get<int>();
			validate_remeshing_geometry(state_args[state], geometry);
			body_id = state_args[state]["geometry"][geometry].value("volume_selection", 0);
		}

		int remesh_count = 0;
		while (true)
		{
			remeshing_requested_ = false;

			// Each remeshing round use a unique output directory.
			json round_args = input_args;
			if (remeshing_enabled)
			{
				const std::filesystem::path round_dir = output_root / fmt::format("remesh_round_{:d}", remesh_count);
				round_args["output"]["directory"] = round_dir.string();
				state_args[state]["output"]["directory"] = round_dir.string();
			}

			init(round_args, strict_validation);
			create_varforms(args["solver"]["max_threads"].get<int>());
			init_variables();
			create_problem();

			Eigen::VectorXd x;
			initial_guess(x);
			// Dry run mode. Compute objective and exit immediately.
			if (args["compute_objective"].get<bool>())
			{
				logger().info("Objective is {}", eval(x));
				return EXIT_SUCCESS;
			}

			const polysolve::nonlinear::Status status = solve(x);
			if (is_failed_status(status))
			{
				logger().error("Optimization failed: {}.", polysolve::nonlinear::status_message(status));
				return EXIT_FAILURE;
			}
			if (!remeshing_requested_)
			{
				return EXIT_SUCCESS;
			}
			if (remesh_count >= max_restarts)
			{
				logger().info("Reached the optimization remeshing limit of {} restart(s).", max_restarts);
				return EXIT_SUCCESS;
			}

			const std::filesystem::path round_dir = output_root / fmt::format("remesh_round_{:d}", remesh_count);
			const std::filesystem::path remeshed_path = std::filesystem::absolute(round_dir / "remeshed.msh");
			const std::optional<double> quality_threshold = mode == "scaled_jacobian"
																? std::optional<double>(args["remeshing"]["trigger"]["scaled_jacobian"]["quality_threshold"].get<double>())
																: std::nullopt;
			if (!remesh_and_write(
					*this, state, body_id,
					remeshed_path, quality_threshold))
			{
				logger().error("MMG failed to produce a valid optimization restart mesh.");
				return EXIT_FAILURE;
			}

			state_args[state]["geometry"][geometry]["mesh"] = remeshed_path.string();
			logger().info("Restarting optimization from remeshed mesh {}.", remeshed_path.string());
			++remesh_count;
		}
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
		strict_validation_ = strict_validation;
		json args_in = p_args_in; // mutable copy
		args = solver::AdjointOptUtils::apply_opt_json_spec(args_in, strict_validation);

		// Save output directory and resolve output paths dynamically
		const std::string output_dir = utils::resolve_path(args["output"]["directory"], root_path(), false);
		if (!output_dir.empty())
		{
			std::filesystem::create_directories(output_dir);
		}
		this->output_dir = output_dir;

		std::string out_path_log = args["output"]["log"]["path"];
		if (!out_path_log.empty())
		{
			out_path_log = utils::resolve_path(out_path_log, root_path(), false);
		}

		init_logger(
			out_path_log,
			args["output"]["log"]["level"],
			args["output"]["log"]["file_level"],
			args["output"]["log"]["quiet"]);

		adjoint_logger().info("Saving adjoint output to {}", output_dir);

		const int thread_in = args["solver"]["max_threads"];
		utils::NThread::get().set_num_threads(thread_in);
	}

	void OptState::create_varforms(const int max_threads)
	{
		if (state_args.empty())
			state_args = load_state_jsons(root_path(), args["states"]);

		size_t threads = max_threads <= 0
							 ? std::numeric_limits<unsigned int>::max()
							 : max_threads;
		varforms.clear();
		for (int i = 0; i < state_args.size(); ++i)
		{
			json cur_args = state_args[i];
			if (!args["output"]["log"].empty())
				cur_args["output"]["log"].merge_patch(args["output"]["log"]);
			varforms.push_back(from_json::build_differentiable_varform(cur_args, threads));
		}

		diff_caches.resize(varforms.size());
		for (auto &diff_cache : diff_caches)
		{
			diff_cache = std::make_shared<DiffCache>();
		}

		check_unsupported();

		utils::GeogramUtils::instance().set_logger(adjoint_logger());
	}

	void OptState::check_unsupported() const
	{
		for (int i = 0; i < varforms.size(); ++i)
		{
			const varform::DifferentiableVarForm &varform = *varforms[i];
			if (!varform.solve_data())
			{
				log_and_throw_adjoint_error(
					"varform::DifferentiableVarForm {} ({}) does not expose solve data required by optimization.",
					i, varform.name());
			}

			// No transient linear support.
			if (varform.get_problem().is_time_dependent() && varform.is_problem_linear())
			{
				log_and_throw_adjoint_error(
					"varform::DifferentiableVarForm {}: transient linear problem is not supported in optimization.", i);
			}

			if (varform.is_contact_enabled())
			{
				// No non-convergent contact formulation support.
				if (!varform.get_args()["contact"]["use_gcp_formulation"].get<bool>()
					&& !varform.get_args()["contact"]["use_convergent_formulation"].get<bool>())
				{
					log_and_throw_adjoint_error(
						"varform::DifferentiableVarForm {}: non-convergent contact formulation is not supported in optimization.", i);
				}

				// No non-const barrier stiffness support.
				if (varform.get_args()["/solver/contact/barrier_stiffness"_json_pointer].is_string())
				{
					log_and_throw_adjoint_error(
						"varform::DifferentiableVarForm {}: only constant barrier stiffness is supported in optimization.", i);
				}
			}

			// No non-const boundary support.
			if (varform.get_args().contains("boundary_conditions") && varform.get_args()["boundary_conditions"].contains("rhs"))
			{
				const json &rhs = varform.get_args()["boundary_conditions"]["rhs"];
				if (rhs.is_string() || (rhs.is_array() && rhs.size() > 0 && rhs[0].is_string()))
				{
					log_and_throw_adjoint_error(
						"varform::DifferentiableVarForm {}: only constant rhs over space is supported in optimization.", i);
				}
			}

			// No high order geometric basis support.
			for (const auto &element_bases : varform.primary_space().geometry_basis_list())
			{
				for (const auto &basis : element_bases.bases)
				{
					if (basis.order() > 1)
					{
						log_and_throw_adjoint_error(
							"varform::DifferentiableVarForm {}: high-order geometry basis is not supported in optimization.", i);
					}
				}
			}
		}
	}

	void OptState::init_variables()
	{
		const json &parameters = args["parameters"];
		bool is_auto = parameters.is_string() && parameters.get<std::string>() == "auto";

		// Auto mode.
		// In auto mode optimization parameters dof is inferred. No need to parse json.
		if (is_auto)
		{
			if (args["variable_to_simulation"].size() != 1)
			{
				log_and_throw_adjoint_error(
					"Auto parameters are only supported with a single variable to simulation.");
			}

			for (auto &composition : utils::json_as_array(args["variable_to_simulation"][0]["composition"]))
			{
				if (composition["type"].get<std::string>() == "slice")
				{
					log_and_throw_adjoint_error("Auto parameters do not support slice maps in composition.");
				}
			}

			variable_to_simulations = from_json::build_variable_to_simulation_group(args["variable_to_simulation"], varforms, diff_caches, {});

			ndof = variable_to_simulations.data[0]->inverse_dof();
			variable_sizes = {ndof};

			return;
		}

		// Manual mode.
		// We need to parse optimization parameter blocks to load dof first.
		variable_sizes.clear();
		ndof = 0;
		for (const auto &arg : args["parameters"])
		{
			int size = solver::AdjointOptUtils::compute_variable_size(arg, varforms);
			ndof += size;
			variable_sizes.push_back(size);
		}

		/* variable to simulations */
		variable_to_simulations = from_json::build_variable_to_simulation_group(
			args["variable_to_simulation"], varforms, diff_caches, variable_sizes);

		// Verify varaible dof.
		for (int i = 0; i < variable_to_simulations.data.size(); ++i)
		{
			auto &var2sim = variable_to_simulations.data[i];
			int inv_dof = var2sim->inverse_dof();
			if (inv_dof != ndof)
			{
				log_and_throw_adjoint_error(
					"VariableToSimulation {} (type {}) expects {} DOF, but parameters define {} DOF.",
					i, var2sim->name(), inv_dof, ndof);
			}
		}
	}

	void OptState::create_problem()
	{
		/* forms */
		std::shared_ptr<solver::AdjointForm> obj = from_json::build_form(
			args["functionals"], variable_to_simulations, varforms, diff_caches);

		/* stopping conditions */
		std::vector<std::shared_ptr<solver::AdjointForm>> stopping_conditions;
		for (const auto &arg : args["stopping_conditions"])
			stopping_conditions.push_back(
				from_json::build_form(arg, variable_to_simulations, varforms, diff_caches));

		std::function<bool()> remeshing_trigger;
		const std::string mode = parse_remeshing_trigger(args);
		if (mode == "periodic")
		{
			int period = args["remeshing"]["trigger"]["periodic"]["period"];
			// this capture is for remeshing_requested_ specifically.
			remeshing_trigger = [this, period, iter = 0]() mutable {
				remeshing_requested_ = (iter % period) == 0;
				++iter;
				if (remeshing_requested_)
					adjoint_logger().debug(
						"Periodic optimization remeshing triggered after {} accepted iteration(s).", iter);
				return remeshing_requested_;
			};
		}
		else if (mode == "scaled_jacobian")
		{
			int state = args["remeshing"]["state"];
			double threshold = args["remeshing"]["trigger"]["scaled_jacobian"]["quality_threshold"];
			auto varform = varforms[state];
			// this capture is for remeshing_requested_ specifically.
			remeshing_trigger = [this, varform, threshold]() {
				double quality = minimum_scaled_jacobian(*varform);
				adjoint_logger().debug(
					"Minimum scaled Jacobian: {} (remesh threshold: {}).",
					quality, threshold);
				remeshing_requested_ = quality <= threshold;
				return remeshing_requested_;
			};
		}

		nl_problem = std::make_unique<solver::AdjointNLProblem>(
			obj, stopping_conditions, variable_to_simulations, varforms, diff_caches,
			args, std::move(remeshing_trigger));
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

	polysolve::nonlinear::Status OptState::solve(Eigen::VectorXd &x)
	{
		auto nl_solver = solver::AdjointOptUtils::make_nl_solver(
			args["solver"]["nonlinear"],
			args["solver"]["linear"],
			args["solver"]["advanced"]["characteristic_length"],
			strict_validation_);
		nl_problem->normalize_forms();
		nl_solver->minimize(*nl_problem, x);
		return nl_solver->status();
	}
} // namespace polyfem
