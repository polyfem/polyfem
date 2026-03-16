#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <cstdlib>

namespace fs = std::filesystem;

struct Options
{
    std::string mesh_path;
    std::string output_dir = "out";
    std::string problem = "laplacian";
    double rhs = 10.0;
    std::string polyfem_bin = "./PolyFEM_bin";
    bool json_only = false;
};

void print_usage()
{
    std::cout << R"(Usage:
  easy_polyfem --mesh <mesh_file> [--problem laplacian] [--output <dir>]
               [--rhs <value>] [--polyfem-bin <path>] [--json-only]

Example:
  easy_polyfem --mesh mesh.obj --problem laplacian --output out
)";
}

bool parse_args(int argc, char **argv, Options &opt)
{
    if (argc < 3)
    {
        print_usage();
        return false;
    }

    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];

        auto need_value = [&](const std::string &name) -> bool {
            if (i + 1 >= argc)
            {
                std::cerr << "Missing value for " << name << "\n";
                return false;
            }
            return true;
        };

        if (arg == "--mesh")
        {
            if (!need_value(arg)) return false;
            opt.mesh_path = argv[++i];
        }
        else if (arg == "--output")
        {
            if (!need_value(arg)) return false;
            opt.output_dir = argv[++i];
        }
        else if (arg == "--problem")
        {
            if (!need_value(arg)) return false;
            opt.problem = argv[++i];
        }
        else if (arg == "--rhs")
        {
            if (!need_value(arg)) return false;
            opt.rhs = std::stod(argv[++i]);
        }
        else if (arg == "--polyfem-bin")
        {
            if (!need_value(arg)) return false;
            opt.polyfem_bin = argv[++i];
        }
        else if (arg == "--json-only")
        {
            opt.json_only = true;
        }
        else if (arg == "--help" || arg == "-h")
        {
            print_usage();
            return false;
        }
        else
        {
            std::cerr << "Unknown argument: " << arg << "\n";
            return false;
        }
    }

    if (opt.mesh_path.empty())
    {
        std::cerr << "Error: --mesh is required.\n";
        return false;
    }

    if (opt.problem != "laplacian")
    {
        std::cerr << "Error: only 'laplacian' is supported in this first version.\n";
        return false;
    }

    return true;
}

std::string escape_json(const std::string &s)
{
    std::ostringstream out;
    for (char c : s)
    {
        switch (c)
        {
        case '\"': out << "\\\""; break;
        case '\\': out << "\\\\"; break;
        case '\n': out << "\\n"; break;
        case '\r': out << "\\r"; break;
        case '\t': out << "\\t"; break;
        default: out << c; break;
        }
    }
    return out.str();
}

std::string make_laplacian_json(const Options &opt)
{
    fs::path out_dir = fs::absolute(opt.output_dir);
    fs::path mesh_abs = fs::absolute(opt.mesh_path);

    std::ostringstream json;
    json << "{\n";
    json << "  \"geometry\": {\n";
    json << "    \"advanced\": {\n";
    json << "      \"normalize_mesh\": true\n";
    json << "    },\n";
    json << "    \"mesh\": \"" << escape_json(mesh_abs.string()) << "\",\n";
    json << "    \"surface_selection\": {\n";
    json << "      \"threshold\": 1e-08\n";
    json << "    }\n";
    json << "  },\n";
    json << "  \"materials\": {\n";
    json << "    \"type\": \"Laplacian\"\n";
    json << "  },\n";
    json << "  \"boundary_conditions\": {\n";
    json << "    \"dirichlet_boundary\": [\n";
    json << "      { \"id\": 1, \"value\": 0.0 },\n";
    json << "      { \"id\": 4, \"value\": 1.0 }\n";
    json << "    ],\n";
    json << "    \"rhs\": " << opt.rhs << "\n";
    json << "  },\n";
    json << "  \"output\": {\n";
    json << "    \"directory\": \"" << escape_json(out_dir.string()) << "\",\n";
    json << "    \"json\": \"stats.json\",\n";
    json << "    \"paraview\": {\n";
    json << "      \"file_name\": \"result.vtu\"\n";
    json << "    }\n";
    json << "  }\n";
    json << "}\n";

    return json.str();
}

bool write_text_file(const fs::path &path, const std::string &content)
{
    std::ofstream out(path);
    if (!out.is_open())
        return false;

    out << content;
    return true;
}

int run_polyfem(const Options &opt, const fs::path &json_path)
{
    std::ostringstream cmd;
    cmd << "\"" << opt.polyfem_bin << "\""
        << " --json "
        << "\"" << json_path.string() << "\"";

    std::cout << "Running: " << cmd.str() << "\n";
    return std::system(cmd.str().c_str());
}

int main(int argc, char **argv)
{
    Options opt;
    if (!parse_args(argc, argv, opt))
        return EXIT_FAILURE;

    try
    {
        fs::create_directories(opt.output_dir);

        fs::path json_path = fs::path(opt.output_dir) / "generated_input.json";

        std::string json = make_laplacian_json(opt);

        if (!write_text_file(json_path, json))
        {
            std::cerr << "Failed to write JSON file: " << json_path << "\n";
            return EXIT_FAILURE;
        }

        std::cout << "Generated PolyFEM JSON: " << fs::absolute(json_path) << "\n";

        if (opt.json_only)
        {
            std::cout << "JSON-only mode enabled. Not running PolyFEM.\n";
            return EXIT_SUCCESS;
        }

        int code = run_polyfem(opt, fs::absolute(json_path));
        if (code != 0)
        {
            std::cerr << "PolyFEM execution failed with code: " << code << "\n";
            return EXIT_FAILURE;
        }

        std::cout << "Done. Output directory: " << fs::absolute(opt.output_dir) << "\n";
        return EXIT_SUCCESS;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
}