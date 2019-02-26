#include <polyfem/Mesh3D.hpp>
#include <polyfem/MeshUtils.hpp>
#include <geogram/basic/command_line.h>
#include <geogram/basic/command_line_args.h>
#include <geogram/basic/logger.h>
#include <geogram/basic/progress.h>
#include <geogram/basic/stopwatch.h>
#include <geogram/mesh/mesh.h>
#include <geogram/mesh/mesh_geometry.h>
#include <geogram/mesh/mesh_repair.h>
#include <geogram/mesh/mesh_io.h>

////////////////////////////////////////////////////////////////////////////////

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
	GEO::CmdLine::declare_arg("refinement", 0, "Number of refinement to perform");
	GEO::CmdLine::declare_arg("triangulated", false, "Triangulate each facet by inserting a vertex at its barycenter");

	// Parse command line options and filenames
	std::vector<std::string> filenames;
	if (!GEO::CmdLine::parse(argc, argv, filenames, "input_mesh <output_folder>")) {
		return 1;
	}

	int num_refinement = GEO::CmdLine::get_arg_int("refinement");
	bool triangulated = GEO::CmdLine::get_arg_bool("triangulated");

	// Default output filename is "output" if unspecified
	if (filenames.size() == 1) {
		filenames.push_back("./");
	}

	// Display input and output filenames
	GEO::Logger::div("Command line");
	GEO::Logger::out("Extract") << "Input file: " << filenames[0] << std::endl;
	GEO::Logger::out("Extract") << "Output folder: " << filenames[1] << std::endl;

	// Load the mesh and display timings
	std::unique_ptr<Mesh> tmp;
	GEO::Logger::div("Loading");
	{
		GEO::Stopwatch W("Load");
		tmp = Mesh::create(filenames[0]);
		if (!tmp) {
			return 1;
		}
	}

	Mesh3D &mesh = *dynamic_cast<Mesh3D *>(tmp.get());
	std::vector<int> parent_elements;
	mesh.refine(num_refinement, 0.5, parent_elements);


	// Extracting polyhedra
	std::vector<std::unique_ptr<GEO::Mesh>> polys;
	GEO::Logger::div("Extraction");
	{
		GEO::Stopwatch W("Extract");
		extract_polyhedra(mesh, polys, triangulated);
	}

	// Save mesh
	GEO::Logger::div("Saving");
	{
		GEO::Stopwatch W("Save");
		int counter = 0;
		for (const auto &poly : polys) {
			std::stringstream ss;
			ss << std::setw(6) << std::setfill('0') << counter++;
			std::string s = ss.str();
			GEO::mesh_save(*poly, filenames[1] + "_" + s + ".obj");
		}
	}

	return 0;
}
