#include "State.hpp"
#include "UIState.hpp"

#include "LinearSolver.hpp"
#include "CommandLine.hpp"


#include <geogram/basic/command_line.h>
#include <geogram/basic/command_line_args.h>

using namespace poly_fem;
using namespace Eigen;


/**
*
* args:
*   -mesh <path to the mesh>
*   -n_refs <refinements>
*   -problem <problem name>
*   -spline <use spline basis>
*   -fem <use standard fem with quad/hex meshes>
*   -cmd <runs without ui>
*   -ui <runs with ui>
**/
int main(int argc, const char **argv)
{
#ifndef WIN32
	setenv("GEO_NO_SIGNAL_HANDLER", "1", 1);
#endif

	GEO::initialize();

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

	int discr_order = 1;

	bool use_splines = false;
	bool normalize_mesh = true;
	bool no_ui = false;

	command_line.add_option("-json", json_file);

	command_line.add_option("-mesh", path);

	//for debugging
	command_line.add_option("-n_refs", n_refs);
	command_line.add_option("-problem", problem_name);
	command_line.add_option("-normalize", "-not_norm", normalize_mesh);

	command_line.add_option("-sform", scalar_formulation);
	command_line.add_option("-tform", tensor_formulation);

	command_line.add_option("-q", discr_order);
	command_line.add_option("-spline", "-fem", use_splines);

	//disable out
	command_line.add_option("-cmd", "-ui", no_ui);

	//IO
	command_line.add_option("-output", output);
	command_line.add_option("-vtu", vtu);
	command_line.add_option("-screenshot", screenshot);




	command_line.parse(argc, argv);
	if (!screenshot.empty()) { no_ui = false; }

	json j_args = {
		{"mesh", ""},
		{"n_refs", 0},
		{"refinenemt_location", 0.5},
		{"n_boundary_samples", 10},
		{"problem", "Franke"},
		{"normalize_mesh", true},

		{"scalar_formulation", "Laplacian"},
		{"tensor_formulation", "LinearElasticity"},

		{"quadrature_order", 4},
		{"discr_order", 1},
		{"boundary_samples", 10},
		{"use_spline", false},
		{"iso_parametric", true},
		{"integral_constraints", 2},

		{"fit_nodes", false},

		{"n_harmonic_samples", 10},

		{"solver_type", LinearSolver::defaultSolver()},
		{"precond_type", LinearSolver::defaultPrecond()},

		{"solver_params", {}},

		{"params", {
			{"lambda", 0.75},
			{"mu", 0.375},
			{"k", 1.0},
			{"elasticity_tensor", {}},
			{"young", 1.0},
			{"nu", 0.0},
			{"alphas", {2.13185026692482, -0.600299816209491}},
			{"mus", {0.00407251192475097, 0.000167202574129608}},
			{"Ds", {9.4979, 1000000}}
		}},

		{"problem_params", {}},

		{"output", {}}
	};

	json in_args;

	if(!json_file.empty())
	{
		std::ifstream file(json_file);

		if (file.is_open())
			file >> in_args;
		else
			std::cerr<<"unable to open "<<json_file<<" file"<<std::endl;
		file.close();
	}
	else
	{
		in_args["mesh"] = path;
		in_args["n_refs"] = n_refs;
		in_args["problem"] = problem_name;
		in_args["normalize_mesh"] = normalize_mesh;

		in_args["scalar_formulation"] = scalar_formulation;
		in_args["tensor_formulation"] = tensor_formulation;

		in_args["discr_order"] = discr_order;
		in_args["use_spline"] = use_splines;
		in_args["output"] = output;
	}

	j_args.merge_patch(in_args);

	// std::cout<<j_args.dump(4)<<std::endl;


	if(no_ui)
	{
		State &state = State::state();
		state.init(j_args);


		state.load_mesh();
		state.compute_mesh_stats();

		state.build_basis();
		state.build_polygonal_basis();


		state.assemble_rhs();
		state.assemble_stiffness_mat();

		state.solve_problem();

		state.compute_errors();

		if(j_args.count("output")){
			const std::string out_path = j_args["output"];
			std::ofstream out(out_path);
			state.save_json(out);
		}

		if(!vtu.empty())
			state.save_vtu(vtu);
	}
	else
	{
		UIState::ui_state().launch(j_args);
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

