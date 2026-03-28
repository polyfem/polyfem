#pragma once

#include <optional>
#include <string>
#include <vector>

namespace easy_polyfem
{
    enum class ProblemType
    {
        Laplacian,
        Helmholtz,
        LinearElasticity
    };

    struct DirichletBC
    {
        int id = 0;
        double value = 0.0;
    };

    struct Options
    {
        std::string mesh_path;
        std::string output_dir = "out";
        ProblemType problem = ProblemType::Laplacian;
        std::vector<DirichletBC> dirichlet_boundaries;
        std::optional<double> rhs;
        std::string polyfem_bin = "./PolyFEM_bin";
        bool json_only = false;
        bool normalize_mesh = true;
        std::string json_name = "generated_input.json";
        std::string stats_name = "stats.json";
        std::string vtu_name = "result.vtu";

        // Automatic boundary selection options
        bool auto_boundaries = false;
        std::optional<double> bc_left;
        std::optional<double> bc_right;
        std::optional<double> bc_bottom;
        std::optional<double> bc_top;
        double boundary_eps = 1e-8;
    };

    std::string problem_type_to_string(ProblemType problem);
    bool parse_problem_type(const std::string &text, ProblemType &problem);
}