#include <CLI/CLI.hpp>
#include <polyfem/Mesh3D.hpp>
#include <polyfem/Common.hpp>


#include <geogram/basic/command_line.h>
#include <geogram/basic/command_line_args.h>

#include <igl/writeOBJ.h>

#include <iostream>
#include <fstream>
#include <vector>


using namespace polyfem;
using namespace Eigen;


void load_and_save(const std::string &path, double dx, double dy, double dz, const std::string &out_path)
{
	auto tmp = Mesh::create(path);
	Mesh3D &mesh = *dynamic_cast<Mesh3D *>(tmp.get());

	std::cout<<"N vertices "<<mesh.n_vertices()<<std::endl;

	std::ofstream file;
	file.open(out_path);

	if(!file.good())
	{
		std::cerr<<"Unable to open "<<out_path<<std::endl;
		return;
	}

	for(int i = 0; i < mesh.n_vertices(); ++i){
		const auto tmp = mesh.point(i);
		file << "v " << tmp(0)+dx << " "<<tmp(1)+dy << " " << tmp(2)+dz << "\n";
	}
	file <<"\n";

	for(int f = 0; f < mesh.n_faces(); ++f)
	{
		if(!mesh.is_boundary_face(f))
			continue;

		const int n_f_v = mesh.n_face_vertices(f);
		file <<"f";
		for(int j = 0; j < n_f_v; ++j)
			file <<" " << mesh.face_vertex(f, j)+1;

		file <<"\n";
	}

	file.close();
}

void load_and_triangulate(const std::string &path, double dx, double dy, double dz, MatrixXd &V, MatrixXi &F)
{
	auto tmp = Mesh::create(path);
	Mesh3D &mesh = *dynamic_cast<Mesh3D *>(tmp.get());

	std::cout<<"N vertices "<<mesh.n_vertices()<<std::endl;

	V.resize(mesh.n_vertices() + mesh.n_faces(), 3);
	for(int i = 0; i < mesh.n_vertices(); ++i)
		V.row(i) = mesh.point(i) + Eigen::RowVector3d(dx, dy, dz);

	int v_index = mesh.n_vertices();

	F.resize(mesh.n_faces()*4, 3);

	int index = 0;
	for(int f = 0; f < mesh.n_faces(); ++f)
	{
		if(!mesh.is_boundary_face(f))
			continue;

		const int n_f_v = mesh.n_face_vertices(f);
		if(n_f_v == 3)
		{
			F.row(index) << mesh.face_vertex(f, 2), mesh.face_vertex(f, 1), mesh.face_vertex(f, 0);
			++index;
		}
		else
		{
			auto bary = mesh.face_barycenter(f);
			for(int j = 0; j < n_f_v; ++j)
			{
				F.row(index) << mesh.face_vertex(f, j), mesh.face_vertex(f, (j+1)%n_f_v), v_index;
				++index;
			}

			V.row(v_index) = bary;
			++v_index;
		}
	}

	F.conservativeResize(index, 3);
	V.conservativeResize(v_index, 3);
}



int main(int argc, char **argv)
{
#ifndef WIN32
	setenv("GEO_NO_SIGNAL_HANDLER", "1", 1);
#endif

	GEO::initialize();

    // Import standard command line arguments, and custom ones
	GEO::CmdLine::import_arg_group("standard");
	GEO::CmdLine::import_arg_group("pre");
	GEO::CmdLine::import_arg_group("algo");


	CLI::App command_line{"b_extractor"};
	std::string path = "";
	bool not_triangulate = false;
	double dx = 0, dy = 0, dz = 0;
	command_line.add_option("--mesh,-m", path, "Path to the input mesh");
	command_line.add_flag("--not_tri", not_triangulate, "Skips mesh surface triangulation");
	command_line.add_option("--dx", dx, "x displacement");
	command_line.add_option("--dy", dy, "y displacement");
	command_line.add_option("--dz", dz, "z displacement");

	try {
        command_line.parse(argc, argv);
    } catch (const CLI::ParseError &e) {
        return command_line.exit(e);
    }

	MatrixXd V;
	MatrixXi F;


	if(!path.empty())
	{
		if(not_triangulate)
		{
			load_and_save(path, dx, dy, dz, path + ".obj");
		}
		else
		{
			load_and_triangulate(path, dx, dy, dz, V, F);
			igl::writeOBJ(path + ".obj", V, F);
		}
	}
}
