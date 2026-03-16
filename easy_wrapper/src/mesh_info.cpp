#include "easy_polyfem/mesh_info.hpp"

#include <fstream>
#include <limits>
#include <sstream>
#include <string>

namespace easy_polyfem
{
    static bool ends_with(const std::string &text, const std::string &suffix)
    {
        return text.size() >= suffix.size() &&
               text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
    }

    static bool compute_obj_bounding_box(const std::string &mesh_path,
                                         BoundingBox &bbox,
                                         std::string &error_message)
    {
        std::ifstream in(mesh_path);
        if (!in.is_open())
        {
            error_message = "Failed to open mesh file: " + mesh_path;
            return false;
        }

        double xmin = std::numeric_limits<double>::max();
        double xmax = std::numeric_limits<double>::lowest();
        double ymin = std::numeric_limits<double>::max();
        double ymax = std::numeric_limits<double>::lowest();
        double zmin = std::numeric_limits<double>::max();
        double zmax = std::numeric_limits<double>::lowest();

        bool found_vertex = false;

        std::string line;
        while (std::getline(in, line))
        {
            if (line.size() > 1 && line[0] == 'v' && std::isspace(static_cast<unsigned char>(line[1])))
            {
                std::istringstream iss(line);
                char vchar;
                double x, y, z;
                if (!(iss >> vchar >> x >> y >> z))
                    continue;

                xmin = std::min(xmin, x);
                xmax = std::max(xmax, x);
                ymin = std::min(ymin, y);
                ymax = std::max(ymax, y);
                zmin = std::min(zmin, z);
                zmax = std::max(zmax, z);
                found_vertex = true;
            }
        }

        if (!found_vertex)
        {
            error_message = "No OBJ vertices found in mesh: " + mesh_path;
            return false;
        }

        bbox.xmin = xmin;
        bbox.xmax = xmax;
        bbox.ymin = ymin;
        bbox.ymax = ymax;
        bbox.zmin = zmin;
        bbox.zmax = zmax;

        const double dz = zmax - zmin;
        bbox.dimension = (dz > 1e-12) ? 3 : 2;
        return true;
    }

    bool compute_mesh_bounding_box(const std::string &mesh_path,
                                   BoundingBox &bbox,
                                   std::string &error_message)
    {
        if (ends_with(mesh_path, ".obj"))
            return compute_obj_bounding_box(mesh_path, bbox, error_message);

        error_message = "Automatic boundary detection currently supports only .obj meshes.";
        return false;
    }
}