#include <CLI/CLI.hpp>
#include <polyfem/Mesh2D.hpp>
#include <polyfem/Mesh3D.hpp>

#include <geogram/basic/command_line.h>
#include <geogram/basic/command_line_args.h>
#include <geogram/mesh/mesh_io.h>

using namespace polyfem;
using namespace Eigen;

int main(int argc, char *argv[])
{
#ifndef WIN32
    setenv("GEO_NO_SIGNAL_HANDLER", "1", 1);
#endif
    GEO::initialize();

    // Import standard command line arguments, and custom ones
    GEO::CmdLine::import_arg_group("standard");
    GEO::CmdLine::import_arg_group("pre");

    struct
    {
        std::string mesh_path = "";
        std::string out_path = "";
    } args;

    CLI::App app{"convert 3d mesh"};
    app.add_option("mesh_path,-m,--mesh_path", args.mesh_path, "Input mesh")->check(CLI::ExistingFile);
    app.add_option("output,-o,--opout", args.out_path, "Output mesh")->required();

    try
    {
        app.parse(argc, argv);
    }
    catch (const CLI::ParseError &e)
    {
        return app.exit(e);
    }

    const auto mesh = Mesh::create(args.mesh_path);
    const auto &mesh3d = *dynamic_cast<Mesh3D *>(mesh.get());
    mesh3d.save(args.out_path);

    return EXIT_SUCCESS;
}
