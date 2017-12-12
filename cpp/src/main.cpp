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
*   -problem <0: linear, 1: quadratic, 2: franke, 3: linear elasiticity>
*   -quad <quadrature order>
*   -b_samples <number of boundary samples>
*   -spline <use spline basis>
*   -fem <use standard fem with quad/hex meshes>
*   -elasticity <use linear elasitcity>
*   -poisson <use poisson problem>
*   -cmd <runs without ui>
*   -ui <runs with ui>
**/
int main(int argc, const char **argv)
{
#ifndef WIN32
    setenv("GEO_NO_SIGNAL_HANDLERS", "1", 1);
#endif

    GEO::initialize();

    // Import standard command line arguments, and custom ones
    GEO::CmdLine::import_arg_group("standard");
    GEO::CmdLine::import_arg_group("pre");

    // GEO::Logger::set_quiet(true);



    CommandLine command_line;

    std::string path = "";
    int n_refs = 0;
    int problem_num = 0;

    int quadrature_order = 4;
    int n_boundary_samples = 10;


    bool use_splines = false;
    bool linear_elasticity = false;

    bool no_ui = false;

    command_line.add_option("-mesh", path);
    command_line.add_option("-n_refs", n_refs);
    command_line.add_option("-problem", problem_num);


    command_line.add_option("-quad", quadrature_order);
    command_line.add_option("-b_samples", n_boundary_samples);
    command_line.add_option("-spline", "-fem", use_splines);
    command_line.add_option("-elasticity", "-poisson", linear_elasticity);

    command_line.add_option("-cmd", "-ui", no_ui);

    command_line.parse(argc, argv);

    if(no_ui)
    {
        State &state = State::state();

        state.quadrature_order = quadrature_order;
        state.use_splines = use_splines;
        state.linear_elasticity = linear_elasticity;
        state.n_boundary_samples = n_boundary_samples;

        state.init(path, n_refs, problem_num);

        state.load_mesh();
        state.build_basis();
        state.compute_assembly_vals();
        state.assemble_stiffness_mat();
        state.assemble_rhs();
        state.solve_problem();
        state.compute_errors();
    }
    else
    {
        UIState::ui_state().state.quadrature_order = quadrature_order;
        UIState::ui_state().state.use_splines = use_splines;
        UIState::ui_state().state.linear_elasticity = linear_elasticity;
        UIState::ui_state().state.n_boundary_samples = n_boundary_samples;

        UIState::ui_state().init(path, n_refs, problem_num);
    }


    return EXIT_SUCCESS;
}
