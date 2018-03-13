#include "State.hpp"
#include "UIState.hpp"

#include "CommandLine.hpp"


#include <geogram/basic/command_line.h>
#include <geogram/basic/command_line_args.h>

using namespace poly_fem;
using namespace Eigen;


/**
* no ui:
* <exec> -mesh <path> -problem <0,1,2,3> -cmd
*
* ui:
* <exec> -mesh <path> -problem <0,1,2,3>
*
* args:
*   -mesh <path to the mesh>
*   -n_refs <refinements>
*   -problem <problem name>
*   -quad <quadrature order>
*   -b_samples <number of boundary samples>
*   -spline <use spline basis>
*   -fem <use standard fem with quad/hex meshes>
*   -lambda <first lame parameter>
*   -mu <second lame parameter>
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
	int n_refs = 0;

	std::string scalar_formulation = "Laplacian";
	std::string tensor_formulation = "LinearElasticity"; //"SaintVenant";

	int discr_order = 1;

	bool use_splines = false;
	bool normalize_mesh = true;
	bool no_ui = false;

	command_line.add_option("-mesh", path);
	command_line.add_option("-n_refs", n_refs);
	command_line.add_option("-problem", problem_name);
	command_line.add_option("-normalize", "-not_norm", normalize_mesh);

	command_line.add_option("-sform", scalar_formulation);
	command_line.add_option("-tform", tensor_formulation);

	command_line.add_option("-q", discr_order);
	command_line.add_option("-spline", "-fem", use_splines);

	command_line.add_option("-cmd", "-ui", no_ui);

	command_line.add_option("-output", output);
	command_line.add_option("-vtu", vtu);
	command_line.add_option("-screenshot", screenshot);

	command_line.parse(argc, argv);
	if (!screenshot.empty()) { no_ui = false; }


	json j_args = {
		{"mesh", ""},
		{"n_refs", 0},
		{"ref_t", 0.5},
		{"problem", "Franke"},
		{"normalize_mesh", true},

		{"scalar_form", "Laplacian"},
		{"tensor_form", "LinearElasticity"},

		{"n_quadrature_points", 4},
		{"disc_order", 1},
		{"boundary_samples", 10},
		{"use_spline", false},
		{"iso_parametric", true},
		{"integral_constraints", 2},

		{"params", {
			{"lambda", 1},
			{"mu", 1},
			{"k", 1},
		}}
	};

	json in_args;
	in_args["mesh"] = path;
	in_args["n_refs"] = n_refs;
	in_args["problem"] = problem_name;
	in_args["normalize"] = normalize_mesh;

	in_args["scalar_form"] = scalar_formulation;
	in_args["tensor_form"] = tensor_formulation;

	in_args["disc_order"] = discr_order;
	in_args["use_spline"] = use_splines;

	j_args.merge_patch(in_args);


	std::cout << j_args.dump(4) << std::endl;

	exit(0);

	// if(no_ui)
	// {
	// 	State &state = State::state();

	// 	state.quadrature_order = quadrature_order;
	// 	state.use_splines = use_splines;
	// 	state.lambda = lambda;
	// 	state.mu = mu;
	// 	state.discr_order = discr_order;
	// 	state.n_boundary_samples = n_boundary_samples;
	// 	state.refinenemt_location = refinenemt_location;
	// 	state.iso_parametric = iso_parametric;
	// 	state.integral_constraints = integral_constraints;
	// 	state.normalize_mesh = normalize_mesh;
	// 	state.scalar_formulation = scalar_formulation;
	// 	state.tensor_formulation = tensor_formulation;

	// 	state.init(path, n_refs, problem_name);
 //        // std::cout<<path<<std::endl;
 //        // for(int i = 0; i < 6; ++i)
	// 	{
	// 		state.load_mesh();
	// 		state.compute_mesh_stats();
	// 		state.build_basis();

 //            // if(state.n_flipped == 0)
 //                // break;
	// 	}
	// 	state.build_polygonal_basis();

	// 	// if(!hack.empty()){
	// 	// 	state.compute_poly_basis_error(hack);
	// 	// 	return EXIT_SUCCESS;
	// 	// }

	// 	state.assemble_rhs();
	// 	state.assemble_stiffness_mat();
	// 	state.solve_problem();
 //        // state.solve_problem_old();
	// 	state.compute_errors();

	// 	if(!output.empty()){
	// 		std::ofstream out(output);
	// 		state.save_json(out);
	// 	}

	// 	if(!vtu.empty())
	// 		state.save_vtu(vtu);
	// }
	// else
	// {
	// 	UIState::ui_state().state.quadrature_order = quadrature_order;
	// 	UIState::ui_state().state.discr_order = discr_order;
	// 	UIState::ui_state().state.use_splines = use_splines;
	// 	UIState::ui_state().state.lambda = lambda;
	// 	UIState::ui_state().state.mu = mu;
	// 	UIState::ui_state().state.n_boundary_samples = n_boundary_samples;
	// 	UIState::ui_state().state.refinenemt_location = refinenemt_location;
	// 	UIState::ui_state().state.iso_parametric = iso_parametric;
	// 	UIState::ui_state().state.integral_constraints = integral_constraints;
	// 	UIState::ui_state().screenshot = screenshot;
	// 	UIState::ui_state().state.normalize_mesh = normalize_mesh;
	// 	UIState::ui_state().state.scalar_formulation = scalar_formulation;
	// 	UIState::ui_state().state.tensor_formulation = tensor_formulation;

	// 	UIState::ui_state().launch(path, n_refs, problem_name);
	// }


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

