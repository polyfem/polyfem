#include <CLI/CLI.hpp>
#include <polyfem/Mesh2D.hpp>
#include <polyfem/Mesh3D.hpp>


#include <geogram/basic/command_line.h>
#include <geogram/basic/command_line_args.h>
#include <geogram/mesh/mesh_io.h>

using namespace polyfem;
using namespace Eigen;

int main(int argc, char * argv[]) {
#ifndef WIN32
    setenv("GEO_NO_SIGNAL_HANDLER", "1", 1);
#endif
	GEO::initialize();

    // Import standard command line arguments, and custom ones
    GEO::CmdLine::import_arg_group("standard");
    GEO::CmdLine::import_arg_group("pre");


    struct {
        std::string mesh_path = "";
        float height = 1.;
        int layers = 1;
    } args;


    CLI::App app{"extrude 2D mesh"};
    app.add_option("mesh_path,-m,--mesh_path", args.mesh_path, "Mesh without extension.")->required();
    app.add_option("--height", args.height, "Height.");
    app.add_option("-l,--layers", args.layers, "Number of layers.");
    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError &e) {
        return app.exit(e);
    }

    GEO::Mesh mesh;
    GEO::mesh_load(args.mesh_path+".obj", mesh);

    Mesh3DStorage tmp;
    tmp.type = MeshType::HSur;
    Mesh3D mesh_3d;
    Mesh3D::geomesh_2_mesh_storage(mesh, tmp);
    MeshProcessing3D::straight_sweeping(tmp, 2, args.height, args.layers, mesh_3d.mesh_storge());

    mesh_3d.save(args.mesh_path+".HYBRID");

    return EXIT_SUCCESS;
}
