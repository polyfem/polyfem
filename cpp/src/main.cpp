#include <polyfem/UIState.hpp>

#include <polyfem/State.hpp>

#include <polyfem/LinearSolver.hpp>
#include <polyfem/StringUtils.hpp>
#include <polyfem/Logger.hpp>

#include <polyfem/Problem.hpp>
#include <polyfem/AssemblerUtils.hpp>
#include <polyfem/LinearSolver.hpp>

#include <geogram/basic/command_line.h>
#include <geogram/basic/command_line_args.h>

#include <CLI11.hpp>

#ifdef USE_TBB
#include <tbb/task_scheduler_init.h>
#include <thread>
#endif


using namespace polyfem;
using namespace Eigen;


#include <geogram/basic/logger.h>

class GeoLoggerForward: public GEO::LoggerClient
{
	void div(const std::string& title) override
	{
		logger().info("[GEO] " + title.substr(0, title.size()-1));
	}

	void out(const std::string& str) override
	{
		logger().info("[GEO] " + str.substr(0, str.size()-1));
	}

	void warn(const std::string& str) override
	{
		logger().warn("[GEO] " + str.substr(0, str.size()-1));
	}

	void err(const std::string& str) override
	{
		logger().error("[GEO] " + str.substr(0, str.size()-1));
	}

	void status(const std::string& str) override
	{
		logger().critical("[GEO] " + str.substr(0, str.size()-1));
	}
};


int main(int argc, char **argv)
{
#ifndef WIN32
	setenv("GEO_NO_SIGNAL_HANDLER", "1", 1);
#endif

	GEO::initialize();

#ifdef USE_TBB
	const size_t MB = 1024*1024;
	const size_t stack_size = 64 * MB;
	unsigned int num_threads = std::max(1u, std::thread::hardware_concurrency());
	tbb::task_scheduler_init scheduler(num_threads, stack_size);
#endif

    // Import standard command line arguments, and custom ones
	GEO::CmdLine::import_arg_group("standard");
	GEO::CmdLine::import_arg_group("pre");
	GEO::CmdLine::import_arg_group("algo");


	CLI::App command_line{"polyfem"};


	std::string path = "";
	std::string output = "";
	std::string vtu = "";
	std::string screenshot = "";
	std::string problem_name = "Franke";
	std::string json_file = "";
	
	int n_refs = 0;

	std::string scalar_formulation = "Laplacian";
	std::string tensor_formulation = "LinearElasticity"; //"SaintVenant";
	std::string mixed_formulation = "Stokes"; //"SaintVenant";
	std::string solver = "";

	int discr_order = 1;

	bool use_splines = false;
	bool normalize_mesh = true;
	bool no_ui = false;
	bool p_ref = false;
	bool force_linear = false;


	std::string log_file = "";
	bool use_cout = true;
	int log_level = 1;

	command_line.add_option("-j,--json", json_file, "Simulation json file")->check(CLI::ExistingFile);
	command_line.add_option("-m,--mesh", path, "Mesh path")->check(CLI::ExistingFile);


	//for debugging
	command_line.add_option("--n_refs", n_refs, "Number of refinements");
	command_line.add_flag("--not_norm", normalize_mesh, "Skips mesh normalization");


	const ProblemFactory &p_factory = ProblemFactory::factory();
	command_line.add_set("--problem", problem_name, std::set<std::string>(p_factory.get_problem_names().begin(), p_factory.get_problem_names().end()), "Problem name");

	const AssemblerUtils &assembler = AssemblerUtils::instance();
	command_line.add_set("--sform", scalar_formulation, std::set<std::string>(assembler.scalar_assemblers().begin(), assembler.scalar_assemblers().end()), "Scalar formulation");
	command_line.add_set("--tform", tensor_formulation, std::set<std::string>(assembler.tensor_assemblers().begin(), assembler.tensor_assemblers().end()), "Tensor formulation");
	command_line.add_set("--mform", mixed_formulation, std::set<std::string>(assembler.mixed_assemblers().begin(), assembler.mixed_assemblers().end()),  "Mixed formulation");

	const std::vector<std::string> solvers = LinearSolver::availableSolvers();
	command_line.add_set("--solver", solver, std::set<std::string>(solvers.begin(), solvers.end()), "Solver to use");

	command_line.add_option("-q,-p", discr_order, "Discretization order");
	command_line.add_flag("--p_ref", p_ref, "Use p refimenet");
	command_line.add_flag("--spline", use_splines, "Use spline for quad/hex meshes");
	command_line.add_flag("--lin_geom", force_linear, "Force use linear geometric mapping");

	//disable out
	command_line.add_flag("--cmd", no_ui, "Runs in command line mode, no ui");

	//IO
	command_line.add_option("--output", output, "Output json file");
	command_line.add_option("--vtu", vtu, "Vtu output file");
	command_line.add_option("--screenshot", screenshot, "screenshot (disabled)");


	command_line.add_option("--no_cout_log", use_cout, "Disable cout for logging");
	command_line.add_option("--log_file", log_file, "Log to a file");
	command_line.add_option("--log_level", log_level, "Log level 1 debug 2 info");


    try {
        command_line.parse(argc, argv);
    } catch (const CLI::ParseError &e) {
        return command_line.exit(e);
    }

	if (!screenshot.empty()) { no_ui = false; }

	Logger::init(use_cout, log_file);
	log_level = std::max(0, std::min(6, log_level));
    spdlog::set_level(static_cast<spdlog::level::level_enum>(log_level));
    spdlog::flush_every(std::chrono::seconds(3));

	GEO::Logger *geo_logger = GEO::Logger::instance();
	geo_logger->unregister_all_clients();
	geo_logger->register_client(new GeoLoggerForward());



	json in_args = json({});

	if(!json_file.empty())
	{
		std::ifstream file(json_file);

		if (file.is_open())
			file >> in_args;
		else
			logger().error("unable to open {} file", json_file);
		file.close();
	}
	else
	{
		in_args["mesh"] = path;
		in_args["force_linear_geometry"] = force_linear;
		in_args["n_refs"] = n_refs;
		in_args["problem"] = problem_name;
		in_args["normalize_mesh"] = normalize_mesh;


		in_args["scalar_formulation"] = scalar_formulation;
		in_args["tensor_formulation"] = tensor_formulation;
		in_args["mixed_formulation"] = mixed_formulation;

		in_args["discr_order"] = discr_order;
		in_args["use_spline"] = use_splines;
		in_args["output"] = output;
		in_args["use_p_ref"] = p_ref;

		if (!vtu.empty()) {
			in_args["export"]["vis_mesh"] = vtu;
			in_args["export"]["wire_mesh"] = StringUtils::replace_ext(vtu, "obj");
		}
		if (!solver.empty())
			in_args["solver_type"] = solver;
	}


	if(no_ui)
	{
		State &state = State::state();
		state.init(in_args);


		state.load_mesh();
		state.compute_mesh_stats();

		state.build_basis();
		state.build_polygonal_basis();


		state.assemble_rhs();
		state.assemble_stiffness_mat();

		state.solve_problem();

		state.compute_errors();

		state.save_json();
		state.export_data();
	}
	else
	{
		UIState::ui_state().launch(in_args);
	}


	return EXIT_SUCCESS;
}

