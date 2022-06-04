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

		this->args = {
			{"root_path", ""},
			{"mesh", ""},
			{"meshes", nullptr},
			{"obstacles", nullptr},
			{"force_linear_geometry", false},
			{"bc_tag", ""},
			{"boundary_id_threshold", -1.0},

			{"n_refs", 0},
			// {"n_refs_path", ""},
			// {"bodies_n_refs", {}},

			{"vismesh_rel_area", 0.00001},
			{"refinenemt_location", 0.5},
			{"bc_method", "lsq"},
			{"n_boundary_samples", 6},
			{"problem", "Franke"},
			{"normalize_mesh", true},
			{"compute_error", true},

			{"curved_mesh_size", false},

			{"count_flipped_els", false},
			{"project_to_psd", false},
			{"use_al", false},
			{"min_component", -1},

			{"has_collision", false},
			{"dhat", 1e-3},
			{"dhat_percentage", 0.8},
			{"barrier_stiffness", "adaptive"},
			{"force_al", false},
			{"ignore_inertia", false},
			{"epsv", 1e-3},
			{"mu", 0.0},
			{"friction_iterations", 1},
			{"friction_convergence_tol", 1e-2},
			{"lagged_damping_weight", 0},

			{"t0", 0},
			{"tend", -1}, // Compute this automatically
			{"dt", -1},   // Compute this automatically
			{"time_steps", 10},
			{"skip_frame", 1},
			{"time_integrator", "ImplicitEuler"},
			{"time_integrator_params",
			 {{"gamma", 0.5},
			  {"beta", 0.25},
			  {"num_steps", 1}}},

			{"scalar_formulation", "Laplacian"},
			{"tensor_formulation", "LinearElasticity"},

			{"B", 3},
			{"h1_formula", false},

			{"quadrature_order", -1},

			{"discr_order", 1},
			{"discr_orders_path", ""},
			{"bodies_discr_order", {}},

			{"poly_bases", "MFSHarmonic"},
			{"serendipity", false},
			{"discr_order_max", autogen::MAX_P_BASES},
			{"pressure_discr_order", 1},
			{"use_p_ref", false},
			{"has_neumann", false},
			{"use_spline", false},
			{"iso_parametric", false},
			{"integral_constraints", 2},
			{"cache_size", 900000},
			{"al_weight", 1e6},
			{"max_al_weight", 1e11},

			// {"fit_nodes", false},

			{"n_harmonic_samples", 10},

			{"solver_type", LinearSolver::defaultSolver()},
			{"precond_type", LinearSolver::defaultPrecond()},
			{"solver_params",
			 {{"broad_phase_method", "hash_grid"},
			  {"ccd_tolerance", 1e-6},
			  {"ccd_max_iterations", 1e6},
			  {"fDelta", 1e-10},
			  {"gradNorm", 1e-8},
			  {"nl_iterations", 1000},
			  {"useGradNorm", true},
			  {"relativeGradient", false},
			  {"use_grad_norm_tol", 1e-4}}},

			{"rhs_solver_type", LinearSolver::defaultSolver()},
			{"rhs_precond_type", LinearSolver::defaultPrecond()},
			{"rhs_solver_params", json({})},

			{"line_search", "armijo"},
			{"nl_solver", "newton"},
			// {"nl_solver_rhs_steps", 1},

			{"force_no_ref_for_harmonic", false},
			{"lump_mass_matrix", false},

			{"rhs_path", ""},

			{"params",
			 {{"lambda", 0.32967032967032966},
			  {"mu", 0.3846153846153846},
			  {"k", 1.0},
			  {"elasticity_tensor", json({})},
			  {"E", 1e5},
			  {"nu", 0.3},
			  {"density", 1},
			  {"alphas", {2.13185026692482, -0.600299816209491}},
			  {"mus", {0.00407251192475097, 0.000167202574129608}},
			  {"Ds", {9.4979, 1000000}}}},

			{"problem_params",
			 {{"is_time_dependent", false},
			  {"rhs", nullptr},
			  {"exact", nullptr},
			  {"exact_grad", nullptr},
			  {"dirichlet_boundary", nullptr},
			  {"neumann_boundary", nullptr},
			  {"pressure_boundary", nullptr},
			  {"initial_solution", nullptr},
			  {"initial_velocity", nullptr},
			  {"initial_acceleration", nullptr}}},

			{"body_params", nullptr},
			{"boundary_sidesets", nullptr},

			{"output", ""},
			// {"solution", ""},
			// {"stiffness_mat_save_path", ""},

			{"import",
			 {{"u_path", ""},
			  {"v_path", ""},
			  {"a_path", ""}}},

			// {"save_solve_sequence", false},
			{"save_solve_sequence_debug", false},
			{"save_time_sequence", true},
			{"save_nl_solve_sequence", false},

			{"export",
			 {{"sol_at_node", -1},
			  {"high_order_mesh", true},
			  {"volume", true},
			  {"surface", false},
			  {"wireframe", false},
			  {"vis_mesh", ""},
			  {"sol_on_grid", -1},
			  {"paraview", ""},
			  {"vis_boundary_only", false},
			  {"material_params", false},
			  {"body_ids", false},
			  {"contact_forces", false},
			  {"friction_forces", false},
			  {"velocity", false},
			  {"acceleration", false},
			  {"nodes", ""},
			  {"wire_mesh", ""},
			  {"spectrum", false},
			  {"solution", ""},
			  {"full_mat", ""},
			  {"stiffness_mat", ""},
			  {"solution_mat", ""},
			  {"stress_mat", ""},
			  {"u_path", ""},
			  {"v_path", ""},
			  {"a_path", ""},
			  {"mises", ""},
			  {"time_sequence", "sim.pvd"}}}};
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
		has_dhat = args_in.contains("dhat");

		use_avg_pressure = !args["has_neumann"];

		if (args_in.contains("BDF_order"))
		{
			logger().warn("use time_integrator_params: { num_steps: <value> } instead of BDF_order");
			this->args["time_integrator_params"]["num_steps"] = args_in["BDF_order"];
		}

		if (args_in.contains("stiffness_mat_save_path") && !args_in["stiffness_mat_save_path"].empty())
		{
			logger().warn("use export: { stiffness_mat: 'path' } instead of stiffness_mat_save_path");
			this->args["export"]["stiffness_mat"] = args_in["stiffness_mat_save_path"];
		}

		if (args_in.contains("solution") && !args_in["solution"].empty())
		{
			logger().warn("use export: { solution: 'path' } instead of solution");
			this->args["export"]["solution"] = args_in["solution"];
		}

		if (this->args["has_collision"])
		{
			if (!args_in.contains("line_search"))
			{
				args["line_search"] = "backtracking";
				logger().warn("Changing default linesearch to backtracking");
			}

			if (args["friction_iterations"] == 0)
			{
				logger().info("specified friction_iterations is 0; disabling friction");
				args["mu"] = 0.0;
			}
			else if (args["friction_iterations"] < 0)
			{
				args["friction_iterations"] = std::numeric_limits<int>::max();
			}

			if (args["mu"] == 0.0)
			{
				args["friction_iterations"] = 0;
			}
		}
		else
		{
			args["friction_iterations"] = 0;
			args["mu"] = 0;
		}

		problem = ProblemFactory::factory().get_problem(args["problem"]);
		problem->clear();
		if (args["problem"] == "Kernel")
		{
			KernelProblem &kprob = *dynamic_cast<KernelProblem *>(problem.get());
			kprob.state = this;
		}
		// important for the BC
		problem->set_parameters(args["problem_params"]);

		if (args["use_spline"] && args["n_refs"] == 0)
		{
			logger().warn("n_refs > 0 with spline");
		}

		// Save output directory and resolve output paths dynamically
		this->output_dir = output_dir;
	}

} // namespace polyfem
