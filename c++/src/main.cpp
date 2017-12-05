#include "State.hpp"

#include "Mesh.hpp"

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
    // const int quadrature_order = 2;
    // const int n_boundary_samples = 10;

    // Problem *problem = NULL;
    // Mesh mesh;

    // int n_bases;

    // std::vector< std::vector<Basis *> >  bases;
    // std::vector< ElementAssemblyValues > values;
    // std::vector< int >                   bounday_nodes;

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
