#include <geogram/basic/command_line.h>
#include <geogram/basic/command_line_args.h>
#include <geogram/basic/logger.h>
#include <geogram/basic/progress.h>
#include <geogram/basic/stopwatch.h>
#include <igl/read_triangle_mesh.h>
#include <igl/writeMESH.h>
#include <igl/copyleft/tetgen/tetrahedralize.h>

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
	GEO::CmdLine::declare_arg("volume", 1.0, "Target volume");

	// Parse command line options and filenames
	std::vector<std::string> filenames;
	if (!GEO::CmdLine::parse(argc, argv, filenames, "input_mesh <output_mesh>")) {
		return 1;
	}

	double volume = GEO::CmdLine::get_arg_double("volume");

	// Default output filename is "output" if unspecified
	if (filenames.size() == 1) {
		filenames.push_back("output.mesh");
	}

	// Display input and output filenames
	GEO::Logger::div("Command line");
	GEO::Logger::out("I/O") << "Input file: " << filenames[0] << std::endl;
	GEO::Logger::out("I/O") << "Output file: " << filenames[1] << std::endl;

	// Load the mesh and call tetgen
	Eigen::MatrixXd IV, OV;
	Eigen::MatrixXi IF, OF, OT;

	igl::read_triangle_mesh(filenames[0], IV, IF);
	std::string flags = "Vpq1.41a" + std::to_string(volume);
	std::cout << "calling tetgen" << std::endl;
	int res = igl::copyleft::tetgen::tetrahedralize(IV, IF, flags, OV, OT, OF);
	igl::writeMESH(filenames[1], OV, OT, OF);
	std::cout<<"created mesh with "<<OV.rows()<<" vertices"<<std::endl;

	return 0;
}
