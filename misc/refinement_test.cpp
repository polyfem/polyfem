////////////////////////////////////////////////////////////////////////////////
#include <CLI/CLI.hpp>
#include <polyfem/Refinement.hpp>
#include <polyfem/StringUtils.hpp>
#include <vector>
////////////////////////////////////////////////////////////////////////////////

using namespace polyfem;

////////////////////////////////////////////////////////////////////////////////

template<int NUM_SIDES>
bool loadObj(const std::string &filename, Eigen::MatrixXd &V, Eigen::MatrixXi &F) {
	std::string line;
	std::ifstream in(filename);
	if (!in.is_open()) {
		throw std::runtime_error("failed to open file " + filename);
	}

	std::vector<Eigen::RowVector3d> VV;
	std::vector<Eigen::Matrix<int, 1, NUM_SIDES>> FF;
	while (std::getline(in, line)) {
		if (StringUtils::startswith(line, "# Vertices:")) {
			int n;
			std::sscanf(line.c_str(), "# Vertices: %d", &n);
			VV.reserve(n);
			continue;
		}
		if (StringUtils::startswith(line, "# Faces:")) {
			int n;
			std::sscanf(line.c_str(), "# Faces: %d", &n);
			FF.reserve(n);
			continue;
		}
		std::istringstream iss(line);
		std::string key;
		if (iss >> key) {
			if (StringUtils::startswith(key, "#")) {
				continue;
			} else if (key == "v") {
				double x, y, z;
				iss >> x >> y >> z;
				VV.emplace_back();
				VV.back() << x, y, z;
			} else if (key == "f" || key == "l") {
				auto tokens = StringUtils::split(line.substr(1));
				if (tokens.size() != NUM_SIDES) {
					std::cerr << "Facet has incorrect size: " << line << std::endl;
					return false;
				} else {
					FF.emplace_back();
					for (int lv = 0; lv < NUM_SIDES; ++lv) {
						std::string str = tokens[lv];
						int v;
						if (str.find('/') != std::string::npos) {
							v = std::stoi(StringUtils::split(str, "/").front());
						} else {
							v = std::stoi(str);
						}
						FF.back()[lv] = v - 1; // Shift indices by 1 (start from 0)
					}
				}
			}
		}
	}

	V.resize(VV.size(), 3);
	F.resize(FF.size(), NUM_SIDES);
	for (size_t v = 0; v < VV.size(); ++v) {
		V.row(v) = VV[v];
	}
	for (size_t f = 0; f < FF.size(); ++f) {
		F.row(f) = FF[f];
	}

	// std::cout << "Read a mesh with " << VV.size() << " vertices, " << FF.size() << " elements." << std::endl;

	return true;
}

////////////////////////////////////////////////////////////////////////////////

namespace QuadMesh {

bool loadObj(const std::string &filename, Eigen::MatrixXd &V, Eigen::MatrixXi &F) {
	return ::loadObj<4>(filename, V, F);
}

void saveObj(const std::string &filename, const Eigen::MatrixXd &V, const Eigen::MatrixXi &F) {
	using namespace Eigen;
	std::ofstream out(filename);
	if (!out.is_open()) {
		throw std::runtime_error("failed to open file " + filename);
	}
	out << "# Vertices: " << V.rows() << "\n# Faces: " << F.rows() << "\n"
		<< V.cast<float>().format(IOFormat(FullPrecision,DontAlignCols," ","\n","v ","","","\n"))
		<< (F.array()+1).format(IOFormat(FullPrecision,DontAlignCols," ","\n","f ","","","\n"));
}

} // namespace QuadMesh

////////////////////////////////////////////////////////////////////////////////

int main(int argc, char * argv[]) {
	// Default arguments
	struct {
		std::string input = "input.obj";
		std::string output = "output.obj";
	} args;

	// Parse arguments
	CLI::App app{"quadgen"};
	app.add_option("input,-i,--input", args.input, "Input mesh.")->required();
	app.add_option("output,-o,--output", args.output, "Output mesh.");
	try {
		app.parse(argc, argv);
	} catch (const CLI::ParseError &e) {
		return app.exit(e);
	}


	Eigen::MatrixXd V_in, V_out;
	Eigen::MatrixXi F_in, F_out;
	QuadMesh::loadObj(args.input, V_in, F_in);
	polyfem::refine_quad_mesh(V_in, F_in, V_out, F_out);
	QuadMesh::saveObj(args.output, V_out, F_out);

	return 0;
}
