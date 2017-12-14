////////////////////////////////////////////////////////////////////////////////
#include "Refinement.hpp"
#include "CLI11.hpp"
#include <vector>
////////////////////////////////////////////////////////////////////////////////

namespace Utils {

// Split a string into tokens
std::vector<std::string> split(const std::string &str, const std::string &delimiters = " ") {
	// Skip delimiters at beginning.
	std::string::size_type lastPos = str.find_first_not_of(delimiters, 0);
	// Find first "non-delimiter".
	std::string::size_type pos     = str.find_first_of(delimiters, lastPos);

	std::vector<std::string> tokens;
	while (std::string::npos != pos || std::string::npos != lastPos) {
		// Found a token, add it to the vector.
		tokens.push_back(str.substr(lastPos, pos - lastPos));
		// Skip delimiters.  Note the "not_of"
		lastPos = str.find_first_not_of(delimiters, pos);
		// Find next "non-delimiter"
		pos = str.find_first_of(delimiters, lastPos);
	}

	return tokens;
}

// Skip comments in a stream
std::istream &skip(std::istream &in, char x = '#') {
	std::string dummy;
	while ((in >> std::ws).peek() ==
		std::char_traits<char>::to_int_type(x))
	{
		std::getline(in, dummy);
	}
	return in;
}

// Tests whether a string starts with a given prefix
bool startswith(const std::string &str, const std::string &prefix) {
	return (str.compare(0, prefix.size(), prefix) == 0);
}

// Tests whether a string ends with a given suffix
bool endswidth(const std::string &str, const std::string &suffix) {
	if (str.length() >= suffix.length()) {
		return (0 == str.compare(str.length() - suffix.length(), suffix.length(), suffix));
	} else {
		return false;
	}
}

} // namespace Utils

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
		if (Utils::startswith(line, "# Vertices:")) {
			int n;
			std::sscanf(line.c_str(), "# Vertices: %d", &n);
			VV.reserve(n);
			continue;
		}
		if (Utils::startswith(line, "# Faces:")) {
			int n;
			std::sscanf(line.c_str(), "# Faces: %d", &n);
			FF.reserve(n);
			continue;
		}
		std::istringstream iss(line);
		std::string key;
		if (iss >> key) {
			if (Utils::startswith(key, "#")) {
				continue;
			} else if (key == "v") {
				double x, y, z;
				iss >> x >> y >> z;
				VV.emplace_back();
				VV.back() << x, y, z;
			} else if (key == "f" || key == "l") {
				auto tokens = Utils::split(line.substr(1));
				if (tokens.size() != NUM_SIDES) {
					std::cerr << "Facet has incorrect size: " << line << std::endl;
					return false;
				} else {
					FF.emplace_back();
					for (int lv = 0; lv < NUM_SIDES; ++lv) {
						std::string str = tokens[lv];
						int v;
						if (str.find('/') != std::string::npos) {
							v = std::stoi(Utils::split(str, "/").front());
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
	poly_fem::refine_quad_mesh(V_in, F_in, V_out, F_out);
	QuadMesh::saveObj(args.output, V_out, F_out);

	return 0;
}
