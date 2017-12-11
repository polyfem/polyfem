#include "State.hpp"
#include "UIState.hpp"

#include "Mesh.hpp"

#include "QuadraticBSpline.hpp"
#include "QuadraticTPBSpline.hpp"

#include <geogram/basic/command_line.h>
#include <geogram/basic/command_line_args.h>

#include <fstream>
#include <iostream>


using namespace poly_fem;
using namespace Eigen;






int main(int argc, char *argv[])
{
#ifndef WIN32
    setenv("GEO_NO_SIGNAL_HANDLERS", "1", 1);
#endif

    GEO::initialize();

    // Import standard command line arguments, and custom ones
    GEO::CmdLine::import_arg_group("standard");
    GEO::CmdLine::import_arg_group("pre");

    // GEO::Logger::set_quiet(true);


    // QuadraticBSpline spline({0, 0, 1, 1});
    // std::cout<<spline.derivative(0)<<std::endl;
    // std::cout<<spline.derivative(0.5)<<std::endl;
    // std::cout<<spline.derivative(1)<<std::endl;
    // exit(0);

    // QuadraticTensorProductBSpline spline({0, 1, 1, 1}, {0, 1, 1, 1});
    // MatrixXd tmp, ts(3,2);
    // ts.row(0)=Vector2d(0, 0); ts.row(1)=Vector2d(0.5, 0 ); ts.row(2)=Vector2d(1, 0.5);

    // spline.interpolate(ts, tmp);
    // std::cout<<tmp<<std::endl;

    // spline.derivative(ts, tmp);
    // std::cout<<tmp<<std::endl;

    // exit(0);

    std::string path = "";
    if(argc>=2)
        path = argv[1];

    int n_refs = 0;

    if(argc>=3)
        n_refs=atoi(argv[2]);

    int problem_num = 0;
    if(argc>=4)
        problem_num = atoi(argv[3]);

    UIState::ui_state().init(path, n_refs, problem_num);



//     Eigen::MatrixXd u_exact;
//     problem->exact(mesh.pts, u_exact);

//     Eigen::MatrixXd err = (u - u_exact).cwiseAbs();
//     // Eigen::MatrixXd err = u_exact;

//     Matrix<double,Dynamic, 3> col;
//     igl::jet(err, true, col);
//     auto vis_pts = mesh.pts;

//     if(vis_pts.cols() == 2)
//     {
//         vis_pts = MatrixXd(vis_pts.rows(), 3);
//         vis_pts.col(0) = mesh.pts.col(0);
//         vis_pts.col(1) = mesh.pts.col(1);
//         vis_pts.col(2) = err;
//     }


//     viewer.data.set_mesh(vis_pts, vis_faces);
//     viewer.data.set_colors(col);


    return EXIT_SUCCESS;
}
