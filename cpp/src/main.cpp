#include "State.hpp"

#include "Mesh.hpp"

#include "QuadraticBSpline.hpp"
#include "QuadraticTPBSpline.hpp"

#include <geogram/basic/file_system.h>
#include <geogram/mesh/mesh_io.h>
#include <geogram/mesh/mesh.h>

#include <fstream>
#include <iostream>


using namespace poly_fem;
using namespace Eigen;


void save_obj(const std::string &path, const Mesh &mesh)
{
    std::ofstream file;
    file.open(path.c_str());

    IOFormat format(10, DontAlignCols, " ");

    if(file.good())
    {

        for(long i = 0; i < mesh.pts.rows(); ++i)
        {
            file << "v " << mesh.pts.row(i).format(format) << "\n";
        }

        file << "\n\n";

        for(long i = 0; i < mesh.els.rows(); ++i)
        {
            file << "f " << (mesh.els.row(i).array()+1).format(format) << "\n";
        }

        file.close();
    }
}








int main(int argc, char *argv[])
{
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

    std::string filename;
    GEO::Mesh mesh_;
    if(!GEO::FileSystem::is_file(filename)) {
        std::cerr << "is not a file" << std::endl;
    }

    mesh_.clear(false,false);

    GEO::MeshIOFlags flags;
    if(!mesh_load(filename, mesh_, flags)) {
        std::cerr << "unable to load mesh" << std::endl;
    }

    bool use_hex = false;
    if(argc>=2 && std::string(argv[1])=="hex")
        use_hex = true;

    int n_x_el=2;
    int n_y_el=2;
    int n_z_el=2;

    if(argc>=3)
        n_x_el=atoi(argv[2]);
    if(argc>=4)
        n_y_el=atoi(argv[3]);
    if(argc>=5)
        n_z_el=atoi(argv[4]);

    int problem_num = 0;
    if(argc>=6)
        problem_num = atoi(argv[5]);

    State::state().init(n_x_el, n_y_el, n_z_el, use_hex, problem_num);



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
