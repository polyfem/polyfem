#include "easy_polyfem/cli.hpp"
#include "easy_polyfem/json_writer.hpp"
#include "easy_polyfem/runner.hpp"
#include "easy_polyfem/validator.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

static bool write_text_file(const fs::path &path, const std::string &content)
{
    std::ofstream out(path);
    if (!out.is_open())
        return false;

    out << content;
    return true;
}

int main(int argc, char **argv)
{
    easy_polyfem::Options opt;
    std::string error_message;

    if (!easy_polyfem::parse_args(argc, argv, opt, error_message))
    {
        if (!error_message.empty())
            std::cerr << "Error: " << error_message << "\n";
        return error_message.empty() ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    if (!easy_polyfem::validate_options(opt, error_message))
    {
        std::cerr << "Validation error: " << error_message << "\n";
        return EXIT_FAILURE;
    }

    try
    {
        fs::create_directories(opt.output_dir);

        const fs::path json_path = fs::path(opt.output_dir) / opt.json_name;
        const std::string json = easy_polyfem::build_json(opt);

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

        const int code = easy_polyfem::run_polyfem(opt, fs::absolute(json_path));
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
        std::cerr << "Exception: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
}