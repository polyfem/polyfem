#include "IsosurfaceInflator.hpp"
#include "Logger.hpp"
#include <polyfem/io/MshReader.hpp>
#include <fstream>

// #include <isosurface_inflator/IsosurfaceInflator.hh>
// #include <isosurface_inflator/MeshingOptions.hh>
// #include <isosurface_inflator/IsosurfaceInflatorConfig.hh>

namespace polyfem::utils
{
    namespace
    {
        template <typename T>
        std::string to_string_with_precision(const T a_value, const int n = 16)
        {
            std::ostringstream out;
            out.precision(n);
            out << std::fixed << a_value;
            return out.str();
        }
    }

    void inflate(const std::string binary_path, const std::string wire_path, const json &options, std::vector<double> &params, Eigen::MatrixXd &V, Eigen::MatrixXi &F, Eigen::MatrixXd &vertex_normals, Eigen::MatrixXd &shape_vel)
    {
        const int dim = 2;

        const std::string sdf_velocity_path = options["dump_shape_velocity"];
        const std::string out_path = "tmp.msh";
        std::string shape_params = "--params \"";
        for (int i = 0; i < params.size(); i++)
            shape_params += to_string_with_precision(params[i], 16) + " ";
        shape_params += "\" ";
        std::string command = binary_path + " " // binary path
                            + std::to_string(dim) + std::string("D_") // dimension
                            + options["symmetry"].get<std::string>() + " " // symmetry
                            + wire_path // wireframe path
                            + " -m meshing.json "
                            + shape_params // shape parameters
                            + " -S " + sdf_velocity_path + " " // dump shape velocity path
                            + out_path; // unit cell mesh path, not used

        {
            json opts = options;
            opts.erase("dump_shape_velocity");
            opts.erase("symmetry");
            std::ofstream out("meshing.json");
            out << opts.dump();
            out.close();
        }

        int return_val;
        try 
        {
            return_val = system(command.c_str());
        }
        catch (const std::exception &err)
        {
            log_and_throw_error("remesh command \"{}\" returns {}", command, return_val);
        }

        logger().info("remesh command \"{}\" returns {}", command, return_val);

        {
            std::vector<std::vector<int>> elements;
            std::vector<std::vector<double>> weights;
            std::vector<int> body_ids;
            std::vector<std::string> node_data_name;
            std::vector<std::vector<double>> node_data;
            io::MshReader::load(sdf_velocity_path, V, F, elements, weights, body_ids, node_data_name, node_data);

            assert(node_data_name.size() == node_data.size());

            vertex_normals = Eigen::Map<Eigen::MatrixXd>(node_data[0].data(), 3, V.rows()).topRows(V.cols()).transpose();
            
            shape_vel.setZero(params.size(), V.rows());
            for (int i = 1; i < node_data_name.size(); i++)
                shape_vel.row(i - 1) = Eigen::Map<RowVectorNd>(node_data[i].data(), V.rows());
        }

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
        // {
        //     for (int d = 0; d < dim; d++)
        //         vertex_normals(i, d) = normals[i][d];
        //     vertex_normals.row(i).normalize();
        // }

        // const auto &velocity = inflator.normalShapeVelocities();
        // shape_vel.setZero(velocity.size(), vertices.size());
        // for (int i = 0; i < velocity.size(); i++)
        //     for (int p = 0; p < vertices.size(); p++)
        //         shape_vel(i, p) = velocity[i][p];
    }
}