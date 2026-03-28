#include "easy_polyfem/json_writer.hpp"

#include <filesystem>
#include <sstream>

namespace fs = std::filesystem;

namespace easy_polyfem
{
    static std::string escape_json(const std::string &s)
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

    static void append_manual_dirichlet(std::ostringstream &json, const Options &opt)
    {
        for (std::size_t i = 0; i < opt.dirichlet_boundaries.size(); ++i)
        {
            const auto &bc = opt.dirichlet_boundaries[i];
            json << "      { \"id\": " << bc.id << ", \"value\": " << bc.value << " }";
            if (i + 1 < opt.dirichlet_boundaries.size())
                json << ",";
            json << "\n";
        }
    }

    static void append_auto_dirichlet(std::ostringstream &json, const Options &opt)
    {
        bool first = true;

        auto emit = [&](int id, double value)
        {
            if (!first)
                json << ",\n";
            json << "      { \"id\": " << id << ", \"value\": " << value << " }";
            first = false;
        };

        // PolyFEM box_side ids:
        // 1 = left
        // 2 = bottom
        // 3 = right
        // 4 = top
        // 5 = front
        // 6 = back

        if (opt.bc_left.has_value())   emit(1, *opt.bc_left);
        if (opt.bc_bottom.has_value()) emit(2, *opt.bc_bottom);
        if (opt.bc_right.has_value())  emit(3, *opt.bc_right);
        if (opt.bc_top.has_value())    emit(4, *opt.bc_top);

        json << "\n";
    }

    std::string build_json(const Options &opt)
    {
        const fs::path out_dir = fs::absolute(opt.output_dir);
        const fs::path mesh_abs = fs::absolute(opt.mesh_path);

        std::ostringstream json;
        json << "{\n";

        // IMPORTANT: geometry must be a list
        json << "  \"geometry\": [\n";
        json << "    {\n";
        json << "      \"advanced\": {\n";
        json << "        \"normalize_mesh\": " << (opt.normalize_mesh ? "true" : "false") << "\n";
        json << "      },\n";
        json << "      \"mesh\": \"" << escape_json(mesh_abs.string()) << "\"";

        if (opt.auto_boundaries)
        {
            json << ",\n";
            json << "      \"surface_selection\": [\n";
            json << "        {\n";
            json << "          \"threshold\": " << opt.boundary_eps << "\n";
            json << "        }\n";
            json << "      ]\n";
        }
        else
        {
            json << ",\n";
            json << "      \"surface_selection\": {\n";
            json << "        \"threshold\": 1e-08\n";
            json << "      }\n";
        }

        json << "    }\n";
        json << "  ],\n";

        json << "  \"materials\": {\n";
        json << "    \"type\": \"" << problem_type_to_string(opt.problem) << "\"\n";
        json << "  },\n";

        json << "  \"boundary_conditions\": {\n";
        json << "    \"dirichlet_boundary\": [\n";

        if (opt.auto_boundaries)
            append_auto_dirichlet(json, opt);
        else
            append_manual_dirichlet(json, opt);

        json << "    ]";

        if (opt.rhs.has_value())
        {
            json << ",\n";
            json << "    \"rhs\": " << *opt.rhs << "\n";
        }
        else
        {
            json << "\n";
        }

        json << "  },\n";

        json << "  \"output\": {\n";
        json << "    \"directory\": \"" << escape_json(out_dir.string()) << "\",\n";
        json << "    \"json\": \"" << escape_json(opt.stats_name) << "\",\n";
        json << "    \"paraview\": {\n";
        json << "      \"file_name\": \"" << escape_json(opt.vtu_name) << "\"\n";
        json << "    }\n";
        json << "  }\n";

        json << "}\n";

        return json.str();
    }
}