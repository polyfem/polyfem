#include "IsosurfaceInflator.hpp"
#include "Logger.hpp"

// #include <isosurface_inflator/IsosurfaceInflator.hh>
// #include <isosurface_inflator/MeshingOptions.hh>
// #include <isosurface_inflator/IsosurfaceInflatorConfig.hh>

namespace polyfem::utils
{
    void inflate(std::vector<double> &params, Eigen::MatrixXd &V, Eigen::MatrixXi &F, Eigen::MatrixXd &vertex_normals, Eigen::MatrixXd &shape_vel)
    {
        // const int dim = 2;
        // json args;
        // args["mesher"] = "2D_doubly_periodic";
        // args["wire"] = "bistable.obj";

        // IsosurfaceInflator inflator(args["mesher"].get<std::string>(), true, args["wire"].get<std::string>(), 2, 0);

        // const double defaultThickness = 0.07;
        // if (params.size() != inflator.defaultParameters(defaultThickness).size())
        // {
        //     logger().error("Invalid size of shape params, use default params instead!");
        //     params = inflator.defaultParameters(defaultThickness);
        // }

        // inflator.inflate(params);

        // const auto &vertices = inflator.vertices();
        // V.setZero(vertices.size(), dim);
        // for (int i = 0; i < vertices.size(); i++)
        //     for (int d = 0; d < dim; d++)
        //         V(i, d) = vertices[i][d];
        
        // const auto &elements = inflator.elements();
        // F.setZero(elements.size(), elements.back().size());
        // for (int i = 0; i < elements.size(); i++)
        //     for (int j = 0; j < elements[i].size(); j++)
        //         F(i, j) = elements[i][j];

        // const auto normals = inflator.vertexNormals();
        // vertex_normals.setZero(normals.size(), dim);
        // for (int i = 0; i < normals.size(); i++)
        //     for (int d = 0; d < dim; d++)
        //         vertex_normals(i, d) = normals[i][d];

        // const auto &velocity = inflator.normalShapeVelocities();
        // shape_vel.setZero(velocity.size(), vertices.size());
        // for (int i = 0; i < velocity.size(); i++)
        //     for (int p = 0; p < vertices.size(); p++)
        //         shape_vel(i, p) = velocity[i][p];
    }
}