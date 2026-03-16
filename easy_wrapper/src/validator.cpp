#include "easy_polyfem/validator.hpp"

#include <filesystem>

namespace fs = std::filesystem;

namespace easy_polyfem
{
    bool validate_options(const Options &opt, std::string &error_message)
    {
        if (opt.mesh_path.empty())
        {
            error_message = "--mesh is required.";
            return false;
        }

        if (!fs::exists(opt.mesh_path))
        {
            error_message = "Mesh file does not exist: " + opt.mesh_path;
            return false;
        }

        if (opt.output_dir.empty())
        {
            error_message = "--output must not be empty.";
            return false;
        }

        if (opt.dirichlet_boundaries.empty())
        {
            error_message = "At least one --dirichlet boundary condition is required.";
            return false;
        }

        if (!opt.json_only && !fs::exists(opt.polyfem_bin))
        {
            error_message = "PolyFEM binary does not exist: " + opt.polyfem_bin;
            return false;
        }

        return true;
    }
}