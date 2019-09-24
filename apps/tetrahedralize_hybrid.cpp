#include <CLI/CLI.hpp>
#include <polyfem/Mesh3D.hpp>
#include <polyfem/MeshUtils.hpp>


#include <geogram/basic/command_line.h>
#include <geogram/basic/command_line_args.h>
#include <geogram/basic/logger.h>
#include <geogram/basic/progress.h>
#include <geogram/basic/stopwatch.h>
#include <igl/read_triangle_mesh.h>
#include <igl/writeMESH.h>
#include <igl/writeOBJ.h>
#include <igl/copyleft/tetgen/tetrahedralize.h>

using namespace polyfem;


////////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv) {
	#ifndef WIN32
	setenv("GEO_NO_SIGNAL_HANDLER", "1", 1);
	#endif

	// Initialize the Geogram library
	GEO::initialize();

	// Import standard command line arguments, and custom ones
	GEO::CmdLine::import_arg_group("standard");
	GEO::CmdLine::import_arg_group("pre");

	struct {
		std::string mesh_path = "";
	} args;

	CLI::App app{"tetrahedralize mesh"};
	app.add_option("mesh_path", args.mesh_path, "volume mesh")->required();
	try {
		app.parse(argc, argv);
	} catch (const CLI::ParseError &e) {
		return app.exit(e);
	}

	std::string mesh_path = args.mesh_path;
	std::string out_mesh_path = mesh_path + ".mesh";


	// Display input and output filenames
	GEO::Logger::div("Command line");
	GEO::Logger::out("I/O") << "Input file: " << mesh_path << std::endl;
	GEO::Logger::out("I/O") << "Output file: " << out_mesh_path << std::endl;

	std::unique_ptr<Mesh> tmp;
	GEO::Logger::div("Loading");
	{
		GEO::Stopwatch W("Load");
		tmp = Mesh::create(mesh_path);
		if (!tmp) {
			return 1;
		}
	}
	Mesh3D &mesh = *dynamic_cast<Mesh3D *>(tmp.get());
	mesh.normalize();

	// Load the mesh and call tetgen
	Eigen::MatrixXd IV, OV;
	Eigen::MatrixXi IF, OF, OT;

	IV.resize(2*mesh.n_vertices(), 3);
	IF.resize(mesh.n_faces()*4, 3);

	Eigen::MatrixXi remap(mesh.n_vertices(), 1);
	remap.setConstant(-1);

	int n_vertices_copied = 0;
	int n_faces = 0;

	for(int c = 0; c < mesh.n_cells(); ++c)
	{
		for(int lf = 0; lf < mesh.n_cell_faces(c); ++lf)
		{
			auto index = mesh.get_index_from_element(c, lf, 0);
			if(!mesh.is_boundary_face(index.face))
				continue;

			std::vector<int> vids(mesh.n_face_vertices(index.face));

			for(int lv = 0; lv < mesh.n_face_vertices(index.face); ++lv)
			{
				const int v_id = index.vertex;
				int new_vid;

				if(remap(v_id) < 0)
				{
					remap(v_id) = n_vertices_copied;
					new_vid = n_vertices_copied;

					IV.row(n_vertices_copied) = mesh.point(v_id);
					++n_vertices_copied;
				}
				else
				{
					new_vid = remap(v_id);
				}

				vids[lv] = new_vid;

				index = mesh.next_around_face(index);
			}

			if(vids.size() == 3)
			{
				IF(n_faces, 0) = vids[0];
				IF(n_faces, 2) = vids[1];
				IF(n_faces, 1) = vids[2];

				n_faces++;
			}
			else if (vids.size() == 4)
			{
				IF(n_faces, 0) = vids[0];
				IF(n_faces, 2) = vids[1];
				IF(n_faces, 1) = vids[2];

				n_faces++;

				IF(n_faces, 0) = vids[0];
				IF(n_faces, 2) = vids[2];
				IF(n_faces, 1) = vids[3];

				n_faces++;
			}
			else
			{
				IV.row(n_vertices_copied) = mesh.face_barycenter(index.face);
				const int center = n_vertices_copied;

				for(size_t v = 0; v < vids.size(); ++v)
				{
					IF(n_faces, 0) = vids[v];
					IF(n_faces, 2) = vids[(v+1)%vids.size()];
					IF(n_faces, 1) = center;

					n_faces++;
				}

				++n_vertices_copied;
			}
		}
	}

	IV = IV.block(0, 0, n_vertices_copied, 3).eval();
	IF = IF.block(0, 0, n_faces, 3).eval();

	igl::writeOBJ(mesh_path + ".obj", IV, IF);

	double start_vol = 0.01;
	double end_vol = 1;

	const int target_v = mesh.n_vertices();
	std::cout<<"target v "<<target_v<<std::endl;

	while(true)
	{
		std::string flags = "QpYa" + std::to_string(start_vol);
		int res = igl::copyleft::tetgen::tetrahedralize(IV, IF, flags, OV, OT, OF);

		if(abs(OV.rows() - target_v) < target_v * 0.051)
		{
			igl::writeMESH(out_mesh_path, OV, OT, OF);
			std::cout<<"created mesh with "<<OV.rows()<<" vertices"<<std::endl;
			return 0;
		}

		if(OV.rows() > target_v)
			break;

		start_vol /= 1.5;
		std::cout<<"vol "<<start_vol<<"  "<<OV.rows()<<std::endl;
	}

	while(true)
	{
		std::string flags = "QpYa" + std::to_string(end_vol);
		int res = igl::copyleft::tetgen::tetrahedralize(IV, IF, flags, OV, OT, OF);

		if(abs(OV.rows() - target_v) < target_v * 0.051)
		{
			igl::writeMESH(out_mesh_path, OV, OT, OF);
			std::cout<<"created mesh with "<<OV.rows()<<" vertices"<<std::endl;
			return 0;
		}

		if(OV.rows() < target_v)
			break;

		end_vol *= 1.5;
		std::cout<<"vol "<<end_vol<<"  "<<OV.rows()<<std::endl;
	}


	while((end_vol - start_vol) > 1e-10)
	{
		const double vol = (end_vol+start_vol)/2.0;
		std::string flags = "QpYa" + std::to_string(vol);
		int res = igl::copyleft::tetgen::tetrahedralize(IV, IF, flags, OV, OT, OF);

		if(abs(OV.rows() - target_v) < target_v * 0.051)
		{
			igl::writeMESH(out_mesh_path, OV, OT, OF);
			std::cout<<"created mesh with "<<OV.rows()<<" vertices"<<std::endl;
			return 0;
		}

		if(OV.rows() > target_v)
			start_vol = vol;
		else
			end_vol = vol;

		std::cout<<"vol "<<start_vol<<" - "<<end_vol<<" "<<OV.rows()<<std::endl;
	}

	igl::writeMESH(out_mesh_path, OV, OT, OF);
	std::cout<<"created mesh with "<<OV.rows()<<" vertices"<<std::endl;

	return 0;
}
