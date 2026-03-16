#include "easy_polyfem/cli.hpp"

#include <iostream>
#include <sstream>
#include <stdexcept>

namespace easy_polyfem
{
    std::string problem_type_to_string(ProblemType problem)
    {
        switch (problem)
        {
        case ProblemType::Laplacian:
            return "Laplacian";
        case ProblemType::Helmholtz:
            return "Helmholtz";
        case ProblemType::LinearElasticity:
            return "LinearElasticity";
        }
        return "Laplacian";
    }

    bool parse_problem_type(const std::string &text, ProblemType &problem)
    {
        if (text == "laplacian")
        {
            problem = ProblemType::Laplacian;
            return true;
        }
        if (text == "helmholtz")
        {
            problem = ProblemType::Helmholtz;
            return true;
        }
        if (text == "linear_elasticity")
        {
            problem = ProblemType::LinearElasticity;
            return true;
        }
        return false;
    }

    static bool parse_dirichlet(const std::string &text, DirichletBC &bc)
    {
        const std::size_t colon_pos = text.find(':');
        if (colon_pos == std::string::npos)
            return false;

        try
        {
            bc.id = std::stoi(text.substr(0, colon_pos));
            bc.value = std::stod(text.substr(colon_pos + 1));
            return true;
        }
        catch (...)
        {
            return false;
        }
    }

    void print_usage()
    {
        std::cout << R"(Usage:
  easy_polyfem --mesh <mesh_file>
               [--output <dir>]
               [--problem <laplacian|helmholtz|linear_elasticity>]
               [--rhs <value>]
               [--dirichlet <id:value>]
               [--polyfem-bin <path>]
               [--json-name <file>]
               [--stats-name <file>]
               [--vtu-name <file>]
               [--json-only]

Examples:
  easy_polyfem --mesh mesh.obj --output out --json-only

  easy_polyfem --mesh mesh.obj --output out --problem laplacian \
               --rhs 10 --dirichlet 1:0 --dirichlet 4:1 \
               --polyfem-bin ./build/PolyFEM_bin
)";
    }

    bool parse_args(int argc, char **argv, Options &opt, std::string &error_message)
    {
        if (argc < 2)
        {
            print_usage();
            error_message = "No arguments provided.";
            return false;
        }

        for (int i = 1; i < argc; ++i)
        {
            const std::string arg = argv[i];

            auto require_value = [&](const std::string &name) -> bool {
                if (i + 1 >= argc)
                {
                    error_message = "Missing value for " + name;
                    return false;
                }
                return true;
            };

            if (arg == "--mesh")
            {
                if (!require_value(arg)) return false;
                opt.mesh_path = argv[++i];
            }
            else if (arg == "--output")
            {
                if (!require_value(arg)) return false;
                opt.output_dir = argv[++i];
            }
            else if (arg == "--problem")
            {
                if (!require_value(arg)) return false;
                ProblemType problem;
                const std::string value = argv[++i];
                if (!parse_problem_type(value, problem))
                {
                    error_message = "Unsupported problem type: " + value;
                    return false;
                }
                opt.problem = problem;
            }
            else if (arg == "--rhs")
            {
                if (!require_value(arg)) return false;
                try
                {
                    opt.rhs = std::stod(argv[++i]);
                }
                catch (...)
                {
                    error_message = "Invalid rhs value.";
                    return false;
                }
            }
            else if (arg == "--dirichlet")
            {
                if (!require_value(arg)) return false;
                DirichletBC bc;
                if (!parse_dirichlet(argv[++i], bc))
                {
                    error_message = "Invalid --dirichlet format. Expected id:value";
                    return false;
                }
                opt.dirichlet_boundaries.push_back(bc);
            }
            else if (arg == "--polyfem-bin")
            {
                if (!require_value(arg)) return false;
                opt.polyfem_bin = argv[++i];
            }
            else if (arg == "--json-name")
            {
                if (!require_value(arg)) return false;
                opt.json_name = argv[++i];
            }
            else if (arg == "--stats-name")
            {
                if (!require_value(arg)) return false;
                opt.stats_name = argv[++i];
            }
            else if (arg == "--vtu-name")
            {
                if (!require_value(arg)) return false;
                opt.vtu_name = argv[++i];
            }
            else if (arg == "--json-only")
            {
                opt.json_only = true;
            }
            else if (arg == "--help" || arg == "-h")
            {
                print_usage();
                error_message.clear();
                return false;
            }
            else
            {
                error_message = "Unknown argument: " + arg;
                return false;
            }
        }

        return true;
    }
}