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
        std::string out_mesh = "";
        float height = 100.;
        float side = 20.;
        int vertices = 200;
    } args;


    CLI::App app{"extrude 2D mesh"};
    app.add_option("out_mesh,-o,--out_mesh", args.out_mesh, "out mesh.")->required();
    app.add_option("--height", args.height, "Height.");
    app.add_option("--side", args.side, "Side.");
    app.add_option("-v,--vertices", args.vertices, "Number of vertices.");
    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError &e) {
        return app.exit(e);
    }

    const double v = args.vertices;
    const double s = (pow(0.512e3 + (double) (2700 * v) + 0.60e2 * sqrt(0.3e1) * sqrt((double) (675 * v * v + 256 * v)), 0.2e1 / 0.3e1) + 0.8e1 * pow(0.512e3 + (double) (2700 * v) + 0.60e2 * sqrt(0.3e1) * sqrt((double) (675 * v * v + 256 * v)), 0.1e1 / 0.3e1) + 0.64e2) * pow(0.512e3 + (double) (2700 * v) + 0.60e2 * sqrt(0.3e1) * sqrt((double) (675 * v * v + 256 * v)), -0.1e1 / 0.3e1) / 0.30e2;
    const int x = round(s);
    const int n = x - 1;
    const int b = x*x;
    const int h = round(v/b);



    GEO::Mesh mesh;
    //build grid of x times x vertices, of side args.side

    Mesh3DStorage m;
    m.type = MeshType::HSur;
    m.vertices.clear(); m.edges.clear(); m.faces.clear();
    m.vertices.resize(b);
    m.points.resize(3, b);
    m.faces.resize(n*n);

    int index = 0;

    for (int i = 0; i < x; i++) {
        for (int j = 0; j < x; j++) {
            Vertex &vv = m.vertices[index];
            vv.id = index;

            double xx = args.side/n * j;
            double yy = args.side/n * i;

            vv.v.push_back(xx);
            vv.v.push_back(yy);
            vv.v.push_back(0);

            m.points(0, index) = xx;
            m.points(1, index) = yy;
            m.points(2, index) = 0;

            ++index;
        }
    }
    assert(m.vertices.size() == index);

    index = 0;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            Face &f = m.faces[index];
            f.id = index;
            f.vs.resize(4);

            f.vs[0] = j +       i*x;
            f.vs[1] = j + 1 + i * x;
            f.vs[2] = j + 1 + (i+1) * x;
            f.vs[3] = j + (i+1) * x;

            ++index;
        }
    }
    assert(m.faces.size() == index);
    MeshProcessing3D::build_connectivity(m);

    Mesh3D mesh_3d;
    MeshProcessing3D::straight_sweeping(m, 2, args.height, h, mesh_3d.mesh_storge());

    std::cout<<mesh_3d.n_vertices()<<" vs "<<v<<std::endl;

    mesh_3d.save(args.out_mesh);

    return EXIT_SUCCESS;
}
