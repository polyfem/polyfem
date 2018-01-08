#include "Mesh2D.hpp"
#include "Mesh3D.hpp"

#include "CLI11.hpp"

#include <geogram/basic/command_line.h>
#include <geogram/basic/command_line_args.h>

using namespace poly_fem;
using namespace Eigen;

int main(int argc, char * argv[]) {
	GEO::initialize();

    // Import standard command line arguments, and custom ones
    GEO::CmdLine::import_arg_group("standard");
    GEO::CmdLine::import_arg_group("pre");


    struct {
        std::string mesh_path = "";
        float height = 1.;
        int layers = 1;
    } args;


    CLI::App app{"mesh2d to mesh3d"};
    app.add_option("mesh_path,-m,--mesh_path", args.mesh_path, "Mesh without extension.")->required();
    app.add_option("--height", args.height, "Height.");
    app.add_option("-l,--layers", args.layers, "Number of layers.");
    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError &e) {
        return app.exit(e);
    }

    Mesh2D mesh;
    mesh.load(args.mesh_path+".obj");

    Mesh3DStorage tmp;
    tmp.type = MeshType::HSur;
    Mesh3D mesh_3d;
    Mesh3D::geomesh_2_mesh_storage(mesh.geo_mesh(), tmp);
    MeshProcessing3D::straight_sweeping(tmp, 2, args.height, args.layers, mesh_3d.mesh_storge());

    mesh_3d.save(args.mesh_path+".HYBRID");

    return EXIT_SUCCESS;
}