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

	State::State(const unsigned int max_threads)
	{
		using namespace polysolve;
#ifndef WIN32
		setenv("GEO_NO_SIGNAL_HANDLER", "1", 1);
#endif

		GEO::initialize();
		const unsigned int num_threads = std::max(1u, std::min(max_threads, std::thread::hardware_concurrency()));
		NThread::get().num_threads = num_threads;
#ifdef POLYFEM_WITH_TBB
		thread_limiter = std::make_shared<tbb::global_control>(tbb::global_control::max_allowed_parallelism, num_threads);
#endif

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

#ifdef IPC_TOOLKIT_WITH_LOGGER
		ipc::set_logger(std::make_shared<spdlog::logger>("ipctk", sinks.begin(), sinks.end()));
		ipc::logger().set_level(log_level);
#endif
	}

	void State::init(const json &p_args_in, const std::string &output_dir)
	{
		json args_in = p_args_in; // mutable copy

		if (args_in.contains("common"))
			apply_default_params(args_in);

		// CHECK validity json
		json rules;
		jse::JSE jse;
		{
			jse.strict = true;
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

		const bool valid_input = jse.verify_json(args_in, rules);

		if (!valid_input)
		{
			logger().error("invalid input json, error {}", jse.log2str());
			throw std::runtime_error("Invald input json file");
		}
		//end of check

		this->args = jse.inject_defaults(args_in, rules);
		// std::cout << this->args.dump() << std::endl;

		if (args_in.contains("solver") && args_in["solver"].contains("linear"))
		{
			if (!args_in["solver"]["linear"].contains("solver"))
				this->args["solver"]["linear"]["solver"] = polysolve::LinearSolver::defaultSolver();
			if (!args_in["solver"]["linear"].contains("precond"))
				this->args["solver"]["linear"]["precond"] = polysolve::LinearSolver::defaultPrecond();
		}
		else
		{
			this->args["solver"]["linear"]["solver"] = polysolve::LinearSolver::defaultSolver();
			this->args["solver"]["linear"]["precond"] = polysolve::LinearSolver::defaultPrecond();
		}

		//this cannot be done in the spec as it is system dependent
		{
			const auto ss = polysolve::LinearSolver::availableSolvers();
			const std::string s_json = this->args["solver"]["linear"]["solver"];
			const auto solver_found = std::find(ss.begin(), ss.end(), s_json);
			if (solver_found == ss.end())
			{
				std::stringstream sss;
				for (const auto &s : ss)
					sss << ", " << s;
				log_and_throw_error(fmt::format("Solver {} is invalid, should be one of {}", s_json, sss.str()));
			}

			const auto pp = polysolve::LinearSolver::availablePrecond();
			const std::string p_json = this->args["solver"]["linear"]["precond"];
			const auto precond_found = std::find(pp.begin(), pp.end(), p_json);
			if (precond_found == pp.end())
			{
				std::stringstream sss;
				for (const auto &s : pp)
					sss << ", " << s;
				log_and_throw_error(fmt::format("Precond {} is invalid, should be one of {}", s_json, sss.str()));
			}
		}

		// std::cout << this->args.dump() << std::endl;

		has_dhat = args_in["contact"].contains("dhat");

		init_time();

		if (this->args["contact"]["enabled"])
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
			problem->set_parameters(args["boundary_conditions"]);
			problem->set_parameters(args["initial_conditions"]);

			problem->set_parameters(args["output"]);
		}
		else
		{
			problem = ProblemFactory::factory().get_problem(args["preset_problem"]["type"]);

			problem->clear();
			if (args["preset_problem"]["type"] == "Kernel")
			{
				KernelProblem &kprob = *dynamic_cast<KernelProblem *>(problem.get());
				kprob.state = this;
			}
			// important for the BC
			problem->set_parameters(args["preset_problem"]);
		}

		// TODO:
		// if (args["use_spline"] && args["n_refs"] == 0)
		// {
		// 	logger().warn("n_refs > 0 with spline");
		// }

		// Save output directory and resolve output paths dynamically
		this->output_dir = output_dir;
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
