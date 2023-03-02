#include <polyfem/State.hpp>

#include <polyfem/problem/ProblemFactory.hpp>
#include <polyfem/assembler/GenericProblem.hpp>

#include <polyfem/autogen/auto_p_bases.hpp>
#include <polyfem/autogen/auto_q_bases.hpp>

#include <polyfem/utils/Logger.hpp>
#include <polyfem/problem/KernelProblem.hpp>
#include <polyfem/utils/par_for.hpp>

#include <polysolve/LinearSolver.hpp>

#include <polyfem/utils/JSONUtils.hpp>

#include <jse/jse.h>

#include <geogram/basic/logger.h>
#include <geogram/basic/command_line.h>
#include <geogram/basic/command_line_args.h>

#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/ostream_sink.h>

#include <ipc/utils/logger.hpp>

#include <sstream>

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
	using namespace problem;
	using namespace utils;

	namespace
	{
		class GeoLoggerForward : public GEO::LoggerClient
		{
			std::shared_ptr<spdlog::logger> logger_;

		public:
			template <typename T>
			GeoLoggerForward(T logger) : logger_(logger) {}

		private:
			std::string truncate(const std::string &msg)
			{
				static size_t prefix_len = GEO::CmdLine::ui_feature(" ", false).size();
				return msg.substr(prefix_len, msg.size() - 1 - prefix_len);
			}

		protected:
			void div(const std::string &title) override
			{
				logger_->trace(title.substr(0, title.size() - 1));
			}

			void out(const std::string &str) override
			{
				logger_->info(truncate(str));
			}

			void warn(const std::string &str) override
			{
				logger_->warn(truncate(str));
			}

			void err(const std::string &str) override
			{
				logger_->error(truncate(str));
			}

			void status(const std::string &str) override
			{
				// Errors and warnings are also dispatched as status by geogram, but without
				// the "feature" header. We thus forward them as trace, to avoid duplicated
				// logger info...
				logger_->trace(str.substr(0, str.size() - 1));
			}
		};
	} // namespace

	State::State()
	{
		using namespace polysolve;
#ifndef WIN32
		setenv("GEO_NO_SIGNAL_HANDLER", "1", 1);
#endif

		GEO::initialize();

		// Import standard command line arguments, and custom ones
		GEO::CmdLine::import_arg_group("standard");
		GEO::CmdLine::import_arg_group("pre");
		GEO::CmdLine::import_arg_group("algo");

		problem = ProblemFactory::factory().get_problem("Linear");
	}

	void State::init_logger(const std::string &log_file, const spdlog::level::level_enum log_level, const bool is_quiet)
	{
		std::vector<spdlog::sink_ptr> sinks;

		if (!is_quiet)
			sinks.emplace_back(std::make_shared<spdlog::sinks::stdout_color_sink_mt>());

		if (!log_file.empty())
			sinks.emplace_back(std::make_shared<spdlog::sinks::basic_file_sink_mt>(log_file, /*truncate=*/true));

		init_logger(sinks, log_level);
		spdlog::flush_every(std::chrono::seconds(3));
	}

	void State::init_logger(std::ostream &os, const spdlog::level::level_enum log_level)
	{
		std::vector<spdlog::sink_ptr> sinks;
		sinks.emplace_back(std::make_shared<spdlog::sinks::ostream_sink_mt>(os, false));
		init_logger(sinks, log_level);
	}

	void State::init_logger(const std::vector<spdlog::sink_ptr> &sinks, const spdlog::level::level_enum log_level)
	{
		spdlog::set_level(log_level);

		set_logger(std::make_shared<spdlog::logger>("polyfem", sinks.begin(), sinks.end()));
		logger().set_level(log_level);

		GEO::Logger *geo_logger = GEO::Logger::instance();
		geo_logger->unregister_all_clients();
		geo_logger->register_client(new GeoLoggerForward(logger().clone("geogram")));
		geo_logger->set_pretty(false);

		ipc::set_logger(std::make_shared<spdlog::logger>("ipctk", sinks.begin(), sinks.end()));
		ipc::logger().set_level(log_level);
	}

	void State::init(const json &p_args_in, const bool strict_validation)
	{
		json args_in = p_args_in; // mutable copy

		apply_common_params(args_in);

		// CHECK validity json
		json rules;
		jse::JSE jse;
		{
			jse.strict = strict_validation;
			const std::string polyfem_input_spec = POLYFEM_INPUT_SPEC;
			std::ifstream file(polyfem_input_spec);

			if (file.is_open())
				file >> rules;
			else
			{
				logger().error("unable to open {} rules", polyfem_input_spec);
				throw std::runtime_error("Invald spec file");
			}
		}

		// Set valid options for enabled linear solvers
		for (int i = 0; i < rules.size(); i++)
		{
			if (rules[i]["pointer"] == "/solver/linear/solver")
			{
				rules[i]["default"] = polysolve::LinearSolver::defaultSolver();
				rules[i]["options"] = polysolve::LinearSolver::availableSolvers();
			}
			else if (rules[i]["pointer"] == "/solver/linear/precond")
			{
				rules[i]["default"] = polysolve::LinearSolver::defaultPrecond();
				rules[i]["options"] = polysolve::LinearSolver::availablePrecond();
			}
		}

		const bool valid_input = jse.verify_json(args_in, rules);

		if (!valid_input)
		{
			logger().error("invalid input json:\n{}", jse.log2str());
			throw std::runtime_error("Invald input json file");
		}
		// end of check

		this->args = jse.inject_defaults(args_in, rules);

		const bool fallback_solver = this->args["solver"]["linear"]["enable_overwrite_solver"];
		// Fallback to default linear solver if the specified solver is invalid
		if (fallback_solver)
		{
			const std::string s_json = this->args["solver"]["linear"]["solver"];
			const auto ss = polysolve::LinearSolver::availableSolvers();
			const auto solver_found = std::find(ss.begin(), ss.end(), s_json);
			if (solver_found == ss.end())
			{
				logger().warn("Solver {} is invalid, falling back to {}", s_json, polysolve::LinearSolver::defaultSolver());
				this->args["solver"]["linear"]["solver"] = polysolve::LinearSolver::defaultSolver();
			}
		}

		// Save output directory and resolve output paths dynamically
		const std::string output_dir = resolve_input_path(this->args["output"]["directory"]);
		if (!output_dir.empty())
		{
			std::filesystem::create_directories(output_dir);
		}
		this->output_dir = output_dir;

		std::string out_path_log = this->args["output"]["log"]["path"];
		if (!out_path_log.empty())
		{
			out_path_log = resolve_output_path(out_path_log);
		}

		spdlog::level::level_enum log_level = this->args["output"]["log"]["level"];
		init_logger(out_path_log, log_level, this->args["output"]["log"]["quiet"]);

		logger().info("Saving output to {}", output_dir);

		const unsigned int thread_in = this->args["solver"]["max_threads"];
		set_max_threads(thread_in <= 0 ? std::numeric_limits<unsigned int>::max() : thread_in);

		has_dhat = args_in["contact"].contains("dhat");

		init_time();

		if (is_contact_enabled())
		{
			if (args["solver"]["contact"]["friction_iterations"] == 0)
			{
				logger().info("specified friction_iterations is 0; disabling friction");
				args["contact"]["friction_coefficient"] = 0.0;
			}
			else if (args["solver"]["contact"]["friction_iterations"] < 0)
			{
				args["solver"]["contact"]["friction_iterations"] = std::numeric_limits<int>::max();
			}
			if (args["contact"]["friction_coefficient"] == 0.0)
			{
				args["solver"]["contact"]["friction_iterations"] = 0;
			}
		}
		else
		{
			args["solver"]["contact"]["friction_iterations"] = 0;
			args["contact"]["friction_coefficient"] = 0;
		}

		if (!args.contains("preset_problem"))
		{
			if (assembler.is_scalar(formulation()))
				problem = std::make_shared<assembler::GenericScalarProblem>("GenericScalar");
			else
				problem = std::make_shared<assembler::GenericTensorProblem>("GenericTensor");

			problem->clear();
			if (!args["time"].is_null())
			{
				const auto tmp = R"({"is_time_dependent": true})"_json;
				problem->set_parameters(tmp);
			}
			// important for the BC

			auto bc = args["boundary_conditions"];
			bc["root_path"] = root_path();
			problem->set_parameters(bc);
			problem->set_parameters(args["initial_conditions"]);

			problem->set_parameters(args["output"]);
		}
		else
		{
			if (args["preset_problem"]["type"] == "Kernel")
			{
				problem = std::make_shared<KernelProblem>("Kernel", assembler);
				problem->clear();
				KernelProblem &kprob = *dynamic_cast<KernelProblem *>(problem.get());
			}
			else
			{
				problem = ProblemFactory::factory().get_problem(args["preset_problem"]["type"]);
				problem->clear();
			}
			// important for the BC
			problem->set_parameters(args["preset_problem"]);
		}
	}

	void State::set_max_threads(const unsigned int max_threads)
	{
		const unsigned int num_threads = std::max(1u, std::min(max_threads, std::thread::hardware_concurrency()));
		NThread::get().num_threads = num_threads;
#ifdef POLYFEM_WITH_TBB
		thread_limiter = std::make_shared<tbb::global_control>(tbb::global_control::max_allowed_parallelism, num_threads);
#endif
		Eigen::setNbThreads(num_threads);
	}

	void State::init_time()
	{
		if (!is_param_valid(args, "time"))
			return;

		const double t0 = args["time"]["t0"];
		double tend, dt;
		int time_steps;

		// from "tend", "dt", "time_steps" only two can be used at a time
		const int num_valid = is_param_valid(args["time"], "tend")
							  + is_param_valid(args["time"], "dt")
							  + is_param_valid(args["time"], "time_steps");
		if (num_valid < 2)
		{
			log_and_throw_error("Exactly two of (tend, dt, time_steps) must be specified");
		}
		else if (num_valid == 2)
		{
			if (is_param_valid(args["time"], "tend"))
			{
				tend = args["time"]["tend"];
				assert(tend > t0);
				if (is_param_valid(args["time"], "dt"))
				{
					dt = args["time"]["dt"];
					assert(dt > 0);
					time_steps = int(ceil((tend - t0) / dt));
					assert(time_steps > 0);
				}
				else if (is_param_valid(args["time"], "time_steps"))
				{
					time_steps = args["time"]["time_steps"];
					assert(time_steps > 0);
					dt = (tend - t0) / time_steps;
					assert(dt > 0);
				}
				else
				{
					assert(false);
				}
			}
			else if (is_param_valid(args["time"], "dt"))
			{
				// tend is already confirmed to be invalid, so time_steps must be valid
				assert(is_param_valid(args["time"], "time_steps"));

				dt = args["time"]["dt"];
				assert(dt > 0);

				time_steps = args["time"]["time_steps"];
				assert(time_steps > 0);

				tend = t0 + time_steps * dt;
			}
			else
			{
				// tend and dt are already confirmed to be invalid
				assert(false);
			}
		}
		else if (num_valid == 3)
		{
			tend = args["time"]["tend"];
			dt = args["time"]["dt"];
			time_steps = args["time"]["time_steps"];

			// Check that all parameters agree
			if (abs(t0 + dt * time_steps - tend) > 1e-12)
			{
				logger().error("Exactly two of (tend, dt, time_steps) must be specified");
				throw std::runtime_error("Exactly two of (tend, dt, time_steps) must be specified");
			}
		}

		// Store these for use later
		args["time"]["tend"] = tend;
		args["time"]["dt"] = dt;
		args["time"]["time_steps"] = time_steps;

		logger().info("t0={}, dt={}, tend={}", t0, dt, tend);
	}

} // namespace polyfem
