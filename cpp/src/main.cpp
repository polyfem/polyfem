#include <polyfem/UIState.hpp>

#include <polyfem/State.hpp>

#include <polyfem/LinearSolver.hpp>
#include <polyfem/CommandLine.hpp>
#include <polyfem/StringUtils.hpp>
#include <polyfem/Logger.hpp>

#include <geogram/basic/command_line.h>
#include <geogram/basic/command_line_args.h>

#ifdef USE_TBB
#include <tbb/task_scheduler_init.h>
#include <thread>
#endif


using namespace polyfem;
using namespace Eigen;


int main(int argc, const char **argv)
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

	CommandLine command_line;

	std::string path = "";
	std::string output = "";
	std::string vtu = "";
	std::string screenshot = "";
	std::string problem_name = "Franke";
	std::string json_file = "";
	
	int n_refs = 0;

	std::string scalar_formulation = "Laplacian";
	std::string tensor_formulation = "LinearElasticity"; //"SaintVenant";
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

	command_line.add_option("-json", json_file);

	command_line.add_option("-mesh", path);


	//for debugging
	command_line.add_option("-n_refs", n_refs);
	command_line.add_option("-problem", problem_name);
	command_line.add_option("-normalize", "-not_norm", normalize_mesh);

	command_line.add_option("-sform", scalar_formulation);
	command_line.add_option("-tform", tensor_formulation);

	command_line.add_option("-solver", solver);

	command_line.add_option("-q", discr_order);
	command_line.add_option("-p_ref", "-no_p_ref", p_ref);
	command_line.add_option("-spline", "-fem", use_splines);
	command_line.add_option("-lin_geom", "-nl_geom", force_linear);

	//disable out
	command_line.add_option("-cmd", "-ui", no_ui);

	//IO
	command_line.add_option("-output", output);
	command_line.add_option("-vtu", vtu);
	command_line.add_option("-screenshot", screenshot);


	command_line.add_option("-cout_log", "-no_cout_log", use_cout);
	command_line.add_option("-log_file", log_file);
	command_line.add_option("-log_level", log_level);



	command_line.parse(argc, argv);
	if (!screenshot.empty()) { no_ui = false; }

	Logger::init(use_cout, log_file);
	log_level = std::max(0, std::min(6, log_level));
    spdlog::set_level(static_cast<spdlog::level::level_enum>(log_level));
    spdlog::flush_every(std::chrono::seconds(3));


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

















    // const int n_samples = 1000;
    // const int n_poly_samples = n_samples/3;
    // Eigen::MatrixXd boundary_samples(n_samples, 2);
    // Eigen::MatrixXd poly_samples(n_poly_samples, 2);

    // for(int i = 0; i < n_samples; ++i)
    // {
    //     boundary_samples(i,0) = cos((2.*i)*M_PI/n_samples);
    //     boundary_samples(i,1) = sin((2.*i)*M_PI/n_samples);
    // }

    // for(int i = 0; i < n_poly_samples; ++i)
    // {
    //     poly_samples(i,0) = 1.01*cos((2.*i)*M_PI/n_samples);
    //     poly_samples(i,1) = 1.01*sin((2.*i)*M_PI/n_samples);
    // }

    // Eigen::MatrixXd rhs(3*n_samples, 1);
    // rhs.setZero();

    // for(int i = 0; i < n_samples; ++i)
    // {
    //     rhs(n_samples + 2*i)   = -10*sin((2.*i)*M_PI/n_samples);
    //     rhs(n_samples + 2*i+1) = -10*sin((2.*i)*M_PI/n_samples);
    // }


    // Biharmonic biharmonic(poly_samples, boundary_samples, rhs);

    // exit(0);

