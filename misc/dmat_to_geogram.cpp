////////////////////////////////////////////////////////////////////////////////
#include <CLI/CLI.hpp>
#include <igl/readDMAT.h>
#include <geogram/basic/command_line.h>
#include <geogram/basic/command_line_args.h>
#include <geogram/mesh/mesh_io.h>
#include <iostream>
#include <fstream>
////////////////////////////////////////////////////////////////////////////////

std::string remove_ext(const std::string &filename) {
	size_t lastdot = filename.find_last_of(".");
	if (lastdot == std::string::npos) {
		return filename;
	}
	return filename.substr(0, lastdot);
}

// -----------------------------------------------------------------------------

template<typename T>
void read_dmat(const std::string &filename, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &M) {
	std::ifstream in(filename);
	int r, c;
	in >> r >> c;
	M.resize(r, c);
	for (int i = 0; i < r; ++i) {
		for (int j = 0; j < c; ++j) {
			in >> M(i, j);
		}
	}
}

////////////////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[]) {
#ifndef WIN32
	setenv("GEO_NO_SIGNAL_HANDLER", "1", 1);
#endif
	GEO::initialize();

	// Import standard command line arguments, and custom ones
	GEO::CmdLine::import_arg_group("standard");
	GEO::CmdLine::import_arg_group("pre");

	struct {
		std::string input = "";
		std::string output = "output.geogram";
		std::vector<std::string> vertex_attributes;
	} args;

	CLI::App app{"Assemble matrices into a single geogram mesh"};
	app.add_option("input,-i,--input", args.input, "Input mesh.")->required()->check(CLI::ExistingFile);
	app.add_option("output,-o,--output", args.output, "Output mesh.");
	app.add_option("-a,--attributes", args.vertex_attributes, "List of matrices for each vertex attributes.");
	try {
		app.parse(argc, argv);
	} catch (const CLI::ParseError &e) {
		return app.exit(e);
	}

	// Read mesh + attrs
	GEO::Mesh mesh;
	GEO::mesh_load(args.input, mesh);
	std::vector<Eigen::MatrixXd> attrs;
	for (auto f : args.vertex_attributes) {
		attrs.emplace_back();
		read_dmat(f, attrs.back());
		assert(attrs.back().rows() == mesh.vertices.nb());
	}

	for (size_t k = 0; k < attrs.size(); ++k) {
		std::string name = remove_ext(args.vertex_attributes[k]);
		const Eigen::MatrixXd &A = attrs[k];
		if (A.cols() == 1) {
			GEO::Attribute<double> val(mesh.vertices.attributes(), name);
			for (int v = 0; v < mesh.vertices.nb(); ++v) {
				val[v] = A(v, 0);
			}
		} else if (A.cols() == 2) {
			GEO::Attribute<GEO::vec2> val(mesh.vertices.attributes(), name);
			for (int v = 0; v < mesh.vertices.nb(); ++v) {
				for (int d = 0; d < A.cols(); ++d) {
					val[v][d] = A(v, d);
				}
			}
		} else if (A.cols() == 3) {
			GEO::Attribute<GEO::vec3> val(mesh.vertices.attributes(), name);
			for (int v = 0; v < mesh.vertices.nb(); ++v) {
				for (int d = 0; d < A.cols(); ++d) {
					val[v][d] = A(v, d);
				}
			}
		} else {
			std::cout << "Warning: Atttribute size not supported: " << A.cols() << std::endl;
		}
	}

	// Save geogram mesh
	GEO::mesh_save(mesh, args.output);

	return EXIT_SUCCESS;
}
