#include <polyfem/State.hpp>

#include <polyfem/autogen/auto_p_bases.hpp>
#include <polyfem/autogen/auto_q_bases.hpp>

#include <polyfem/utils/Logger.hpp>
#include <polyfem/problem/KernelProblem.hpp>
#include <polyfem/utils/par_for.hpp>

#include <polysolve/LinearSolver.hpp>

#include <polyfem/utils/JSONUtils.hpp>

#include <geogram/basic/logger.h>
#include <geogram/basic/command_line.h>
#include <geogram/basic/command_line_args.h>

#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/ostream_sink.h>
#include <ipc/utils/logger.hpp>

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

		this->args = R"({
						"defaults": "",
						"root_path": "",

						"geometry": null,

					    "space": {
        					"discr_order": 1,
        					"pressure_discr_order": 1,

        					"use_p_ref": false,

        					"advanced": {
            					"discr_order_max": 4,

								"serendipity": false,
								"isoparametric": false,
								"use_spline": false,

								"bc_method": "lsq",

								"n_boundary_samples": -1,
								"quadrature_order": -1,

								"poly_bases": "MFSHarmonic",
								"integral_constraints": 2,
								"n_harmonic_samples": 10,
								"force_no_ref_for_harmonic": false,

								"B": 3,
								"h1_formula": false,

								"count_flipped_els": true
        					}
    					},

    					"time": null,

						"contact": {
        					"enabled": false,
        					"dhat": 1e-3,
        					"dhat_percentage": 0.8,
        					"epsv": 1e-3,

							"friction_coefficient": 0
    					},

						"solver": {
							"linear": {
								"solver": "",
								"precond": ""
							},

							"nonlinear": {
								"solver" : "newton",
								"f_delta" : 1e-10,
								"grad_norm" : 1e-8,
								"max_iterations" : 1000,
								"use_grad_norm" : true,
								"relative_gradient" : false,

								"line_search": {
									"method" : "backtracking",
									"use_grad_norm_tol" : 1e-4
								}
							},

							"augmented_lagrangian" : {
								"initial_weight" : 1e6,
								"max_weight" : 1e11,

								"force" : false
							},

							"contact": {
								"CCD" : {
									"broad_phase" : "hash_grid",
									"tolerance" : 1e-6,
									"max_iterations" : 1e6
								},
								"friction_iterations" : 1,
								"friction_convergence_tol": 1e-2,
								"barrier_stiffness": "adaptive",
								"lagged_damping_weight": 0
							},

							"ignore_inertia" : false,

							"advanced": {
								"cache_size" : 900000,
								"lump_mass_matrix" : false
							}
						},

						"materials" : null,

						"output": {
							"json" : "",

							"paraview" : {
								"file_name" : "",
								"vismesh_rel_area" : 0.00001,

								"skip_frame" : 1,

								"high_order_mesh" : true,

								"volume" : true,
								"surface" : false,
								"wireframe" : false,

								"options" : {
									"material" : false,
									"body_ids" : false,
									"contact_forces" : false,
									"friction_forces" : false,
									"velocity" : false,
									"acceleration" : false
								}
							},

							"data" : {
								"solution" : "",
								"full_mat" : "",
								"stiffness_mat" : "",
								"solution_mat" : "",
								"stress_mat" : "",
								"u_path" : "",
								"v_path" : "",
								"a_path" : "",
								"mises" : "",
								"nodes" : ""
							},

							"advanced": {
								"timestep_prefix" : "step_",
								"sol_on_grid" : -1,

								"compute_error" : true,

								"sol_at_node" : -1,

								"vis_boundary_only" : false,

								"curved_mesh_size" : false,
								"save_solve_sequence_debug" : false,
								"save_time_sequence" : true,
								"save_nl_solve_sequence" : false,

								"spectrum" : false
							}
						},

						"input": {
							"data" : {
								"u_path" : "",
								"v_path" : "",
								"a_path" : ""
							}
						}
					})"_json;

		this->args["solver"]["linear"]["solver"] = LinearSolver::defaultSolver();
		this->args["solver"]["linear"]["precond"] = LinearSolver::defaultPrecond();
	}

	void State::init_logger(const std::string &log_file, int log_level, const bool is_quiet)
	{
		std::vector<spdlog::sink_ptr> sinks;
		if (!is_quiet)
		{
			sinks.emplace_back(std::make_shared<spdlog::sinks::stdout_color_sink_mt>());
		}
		if (!log_file.empty())
		{
			sinks.emplace_back(std::make_shared<spdlog::sinks::basic_file_sink_mt>(log_file, /*truncate=*/true));
		}
		init_logger(sinks, log_level);
		spdlog::flush_every(std::chrono::seconds(3));
	}

	void State::init_logger(std::ostream &os, int log_level)
	{
		std::vector<spdlog::sink_ptr> sinks;
		sinks.emplace_back(std::make_shared<spdlog::sinks::ostream_sink_mt>(os, false));
		init_logger(sinks, log_level);
	}

	void State::init_logger(std::vector<spdlog::sink_ptr> &sinks, int log_level)
	{
		spdlog::level::level_enum level =
			static_cast<spdlog::level::level_enum>(std::max(0, std::min(6, log_level)));
		spdlog::set_level(level);

		GEO::Logger *geo_logger = GEO::Logger::instance();
		geo_logger->unregister_all_clients();
		geo_logger->register_client(new GeoLoggerForward(logger().clone("geogram")));
		geo_logger->set_pretty(false);

		IPC_LOG(set_level(level));
	}

	void State::init(const json &p_args_in, const std::string &output_dir)
	{
		json args_in = p_args_in; // mutable copy

		if (args_in.contains("default_params"))
			apply_default_params(args_in);

		check_for_unknown_args(args, args_in);

		this->args.merge_patch(args_in);
		has_dhat = args_in["contact"].contains("dhat");

		if (!args["time"].is_null())
		{
			const auto time_default = R"({
				"t0": 0,
				"tend": -1,
				"dt": -1,
				"time_steps": 10,

				"integrator": "ImplicitEuler",
				"newmark": {
					"gamma": 0.5,
					"beta": 0.25
				},
				"BDF": {
					"steps": 1
				}
			})"_json;

			const auto tmp = args["time"];
			args["time"] = time_default;
			args["time"].merge_patch(tmp);
		}

		// if (this->args["contact"]["enabled"])
		// {
		// 	if (!args_in.contains("line_search"))
		// 	{
		// 		args["solver"]["nonlinear"]["line_search"]["method"] = "backtracking";
		// 		logger().warn("Changing default linesearch to backtracking");
		// 	}

		// 	if (args["solver"]["contact"]["friction_iterations"] == 0)
		// 	{
		// 		logger().info("specified friction_iterations is 0; disabling friction");
		// 		args["mu"] = 0.0;
		// 	}
		// 	else if (args["solver"]["contact"]["friction_iterations"] < 0)
		// 	{
		// 		args["solver"]["contact"]["friction_iterations"] = std::numeric_limits<int>::max();
		// 	}

		// 	if (args["mu"] == 0.0)
		// 	{
		// 		args["solver"]["contact"]["friction_iterations"] = 0;
		// 	}
		// }
		// else
		// {
		// 	args["solver"]["contact"]["friction_iterations"] = 0;
		// 	args["mu"] = 0;
		// }

		if (!args.contains("preset_problem"))
		{
			if (assembler.is_scalar(formulation()))
				problem = ProblemFactory::factory().get_problem("GenericScalar");
			else
				problem = ProblemFactory::factory().get_problem("GenericTensor");

			problem->clear();
			if (!args["time"].is_null())
			{
				const auto tmp = R"({"is_time_dependent": true})"_json;
				problem->set_parameters(tmp);
			}
			// important for the BC
			problem->set_parameters(args["boundary_conditions"]);
			if (args.contains("initial_conditions"))
				problem->set_parameters(args["initial_conditions"]);

			if (args["output"].contains("reference"))
				problem->set_parameters(args["output"]["reference"]);
		}
		else
		{
			problem = ProblemFactory::factory().get_problem(args["preset_problem"]["name"]);

			problem->clear();
			if (args["preset_problem"]["name"] == "Kernel")
			{
				KernelProblem &kprob = *dynamic_cast<KernelProblem *>(problem.get());
				kprob.state = this;
			}
			// important for the BC
			problem->set_parameters(args["preset_problem"]);
		}

		//TODO
		// if (args["use_spline"] && args["n_refs"] == 0)
		// {
		// 	logger().warn("n_refs > 0 with spline");
		// }

		// Save output directory and resolve output paths dynamically
		this->output_dir = output_dir;
	}

} // namespace polyfem
