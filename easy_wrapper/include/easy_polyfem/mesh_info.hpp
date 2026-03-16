#pragma once

#include <string>
#include <vector>

namespace easy_polyfem
{
    struct BoundingBox
    {
        double xmin = 0.0;
        double xmax = 0.0;
        double ymin = 0.0;
        double ymax = 0.0;
        double zmin = 0.0;
        double zmax = 0.0;
        int dimension = 2;
    };

    bool compute_mesh_bounding_box(const std::string &mesh_path,
                                   BoundingBox &bbox,
                                   std::string &error_message);
}