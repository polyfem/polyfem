#include <polyfem/State.hpp>

#include <polyfem/auto_p_bases.hpp>
#include <polyfem/auto_q_bases.hpp>

#include <polyfem/RefElementSampler.hpp>
#include <polyfem/Logger.hpp>

#include <polysolve/LinearSolver.hpp>

#include <geogram/basic/logger.h>
#include <geogram/basic/command_line.h>
#include <geogram/basic/command_line_args.h>

namespace polyfem
{
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

#ifdef USE_TBB
        const size_t MB = 1024 * 1024;
        const size_t stack_size = 64 * MB;
        unsigned int num_threads = std::max(1u, std::thread::hardware_concurrency());
        tbb::task_scheduler_init scheduler(num_threads, stack_size);
#endif

        // Import standard command line arguments, and custom ones
        GEO::CmdLine::import_arg_group("standard");
        GEO::CmdLine::import_arg_group("pre");
        GEO::CmdLine::import_arg_group("algo");

        problem = ProblemFactory::factory().get_problem("Linear");

        use_avg_pressure = true;

        this->args = {
            {"mesh", ""},
            {"force_linear_geometry", false},
            {"bc_tag", ""},
            {"boundary_id_threshold", -1.0},
            {"n_refs", 0},
            {"vismesh_rel_area", 0.00001},
            {"refinenemt_location", 0.5},
            {"n_boundary_samples", 6},
            {"problem", "Franke"},
            {"normalize_mesh", true},

            {"curved_mesh_size", false},

            {"count_flipped_els", false},
            {"project_to_psd", false},

            {"has_collision", false},
            {"dhat", 0.03},

            {"tend", 1},
            {"time_steps", 10},

            {"scalar_formulation", "Laplacian"},
            {"tensor_formulation", "LinearElasticity"},

            {"B", 3},
            {"h1_formula", false},

            {"BDF_order", 1},
            {"quadrature_order", 4},
            {"discr_order", 1},
            {"poly_bases", "MFSHarmonic"},
            {"serendipity", false},
            {"discr_order_max", autogen::MAX_P_BASES},
            {"pressure_discr_order", 1},
            {"use_p_ref", false},
            {"use_spline", false},
            {"iso_parametric", false},
            {"integral_constraints", 2},

            {"fit_nodes", false},

            {"n_harmonic_samples", 10},

            {"solver_type", LinearSolver::defaultSolver()},
            {"precond_type", LinearSolver::defaultPrecond()},
            {"solver_params", json({})},

            {"rhs_solver_type", LinearSolver::defaultSolver()},
            {"rhs_precond_type", LinearSolver::defaultPrecond()},
            {"rhs_solver_params", json({})},

            {"line_search", "armijo"},
            {"nl_solver", "newton"},
            {"nl_solver_rhs_steps", 1},
            {"save_solve_sequence", false},
            {"save_solve_sequence_debug", false},
            {"save_time_sequence", true},

            {"force_no_ref_for_harmonic", false},

            {"rhs_path", ""},

            {"params", {{"lambda", 0.32967032967032966}, {"mu", 0.3846153846153846}, {"k", 1.0}, {"elasticity_tensor", json({})},
                        // {"young", 1.0},
                        // {"nu", 0.3},
                        {"density", 1},
                        {"alphas", {2.13185026692482, -0.600299816209491}},
                        {"mus", {0.00407251192475097, 0.000167202574129608}},
                        {"Ds", {9.4979, 1000000}}}},

            {"problem_params", json({})},

            {"output", ""},
            // {"solution", ""},
            // {"stiffness_mat_save_path", ""},

            {"export", {{"sol_at_node", -1}, {"vis_mesh", ""}, {"paraview", ""}, {"vis_boundary_only", false}, {"material_params", false}, {"body_ids", false}, {"nodes", ""}, {"wire_mesh", ""}, {"iso_mesh", ""}, {"spectrum", false}, {"solution", ""}, {"full_mat", ""}, {"stiffness_mat", ""}, {"solution_mat", ""}, {"stress_mat", ""}, {"mises", ""}}}};
    }

    void State::init_logger(const std::string &log_file, int log_level, const bool is_quiet)
    {
        Logger::init(!is_quiet, log_file);
        log_level = std::max(0, std::min(6, log_level));
        spdlog::set_level(static_cast<spdlog::level::level_enum>(log_level));
        spdlog::flush_every(std::chrono::seconds(3));

        GEO::Logger *geo_logger = GEO::Logger::instance();
        geo_logger->unregister_all_clients();
        geo_logger->register_client(new GeoLoggerForward(logger().clone("geogram")));
        geo_logger->set_pretty(false);
    }

    void State::init_logger(std::ostream &os, int log_level)
    {
        Logger::init(os);
        log_level = std::max(0, std::min(6, log_level));
        spdlog::set_level(static_cast<spdlog::level::level_enum>(log_level));
        spdlog::flush_every(std::chrono::seconds(3));

        GEO::Logger *geo_logger = GEO::Logger::instance();
        geo_logger->unregister_all_clients();
        geo_logger->register_client(new GeoLoggerForward(logger().clone("geogram")));
        geo_logger->set_pretty(false);
    }

    void State::init_logger(std::vector<spdlog::sink_ptr> &sinks, int log_level)
    {
        Logger::init(sinks);
        log_level = std::max(0, std::min(6, log_level));
        spdlog::set_level(static_cast<spdlog::level::level_enum>(log_level));

        GEO::Logger *geo_logger = GEO::Logger::instance();
        geo_logger->unregister_all_clients();
        geo_logger->register_client(new GeoLoggerForward(logger().clone("geogram")));
        geo_logger->set_pretty(false);
    }

    void State::init(const json &args_in)
    {
        this->args.merge_patch(args_in);

        if (args_in.find("stiffness_mat_save_path") != args_in.end() && !args_in["stiffness_mat_save_path"].empty())
        {
            logger().warn("use export: { stiffness_mat: 'path' } instead of stiffness_mat_save_path");
            this->args["export"]["stiffness_mat"] = args_in["stiffness_mat_save_path"];
        }

        if (args_in.find("solution") != args_in.end() && !args_in["solution"].empty())
        {
            logger().warn("use export: { solution: 'path' } instead of solution");
            this->args["export"]["solution"] = args_in["solution"];
        }

        if (this->args["has_collision"])
        {
            std::string psd = "";
            std::string gradNorm = "";
            std::string useGradNorm = "";
            std::string linesearch = "";

            if (args_in.find("project_to_psd") == args_in.end())
            {
                args["project_to_psd"] = true;
                psd = "Chaning default project to psd to true ";
            }

            if (args_in.find("line_search") == args_in.end())
            {
                args["line_search"] = "bisection";
                linesearch = "Chaning default linesearch to bisection ";
            }

            if (args_in.find("solver_params") == args_in.end() || args_in["solver_params"].find("gradNorm") == args_in["solver_params"].end())
            {
                args["solver_params"]["gradNorm"] = 1e-5;
                gradNorm = "Chaning default convergence to 1e-5 ";
            }

            if (args_in.find("solver_params") == args_in.end() || args_in["solver_params"].find("useGradNorm") == args_in["solver_params"].end())
            {
                args["solver_params"]["useGradNorm"] = false;
                useGradNorm = "Chaning convergence check to Newton direction ";
            }

            std::string message = psd + linesearch + gradNorm + useGradNorm;

            if (message.length() > 0)
                logger().warn(message);
        }

        problem = ProblemFactory::factory().get_problem(args["problem"]);
        problem->clear();
        //important for the BC
        problem->set_parameters(args["problem_params"]);

        if (args["use_spline"] && args["n_refs"] == 0)
        {
            logger().warn("n_refs > 0 with spline");
        }
    }

} // namespace polyfem