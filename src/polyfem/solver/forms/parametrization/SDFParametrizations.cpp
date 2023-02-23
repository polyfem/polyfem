#include "SDFParametrizations.hpp"

#include <polyfem/utils/MatrixUtils.hpp>
#include <polyfem/io/MshReader.hpp>
#include <polyfem/mesh/mesh2D/Mesh2D.hpp>
#include <polyfem/mesh/mesh3D/Mesh3D.hpp>

namespace polyfem::solver
{
    namespace {
        template <typename T>
        std::string to_string_with_precision(const T a_value, const int n = 6)
        {
            std::ostringstream out;
            out.precision(n);
            out << std::fixed << a_value;
            return out.str();
        }
    }

    bool SDF2Mesh::isosurface_inflator(const Eigen::VectorXd &x) const
    {
        if (last_x == x)
            return true;

        std::string shape_params = "--params \"";
        for (int i = 0; i < x.size(); i++)
            shape_params += to_string_with_precision(x(i), 16) + " ";
        shape_params += "\" ";

        std::string command = inflator_path_ + " " + shape_params + " -S " + sdf_velocity_path_ + " " + msh_path_;

        int return_val;
        try 
        {
            return_val = system(command.c_str());
        }
        catch (const std::exception &err)
        {
            logger().error("remesh command \"{}\" returns {}", command, return_val);

            return false;
        }

        logger().info("remesh command \"{}\" returns {}", command, return_val);

        last_x = x;

        return true;
    }

    SDF2Mesh::SDF2Mesh(const std::string inflator_path, const std::string sdf_velocity_path, const std::string msh_path) : inflator_path_(inflator_path), sdf_velocity_path_(sdf_velocity_path), msh_path_(msh_path)
    {

    }
    
    int SDF2Mesh::size(const int x_size) const
    {
        // if (!isosurface_inflator(x))
        // {
        //     logger().error("Failed to inflate mesh!");
        //     return 0;
        // }

        // Eigen::MatrixXd vertices;
        // Eigen::MatrixXi cells;
        // std::vector<std::vector<int>> elements;
        // std::vector<std::vector<double>> weights;
        // std::vector<int> body_ids;
        // std::vector<std::string> node_data_name;
        // std::vector<std::vector<double>> node_data;
        // io::MshReader::load(sdf_velocity_path_, vertices, cells, elements, weights, body_ids, node_data_name, node_data);
        // const int dim = vertices.cols();

        // return vertices.size();
        log_and_throw_error("Not implemented!");
        return 0;
    }
    
    Eigen::VectorXd SDF2Mesh::eval(const Eigen::VectorXd &x) const
    {
        if (!isosurface_inflator(x))
        {
            logger().error("Failed to inflate mesh!");
            return Eigen::VectorXd();
        }

        Eigen::MatrixXd vertices;
        Eigen::MatrixXi cells;
        std::vector<std::vector<int>> elements;
        std::vector<std::vector<double>> weights;
        std::vector<int> body_ids;
        std::vector<std::string> node_data_name;
        std::vector<std::vector<double>> node_data;
        io::MshReader::load(sdf_velocity_path_, vertices, cells, elements, weights, body_ids, node_data_name, node_data);
        const int dim = vertices.cols();

        return utils::flatten(vertices);
    }
    
    Eigen::VectorXd SDF2Mesh::apply_jacobian(const Eigen::VectorXd &grad, const Eigen::VectorXd &x) const
    {
        if (!isosurface_inflator(x))
        {
            logger().error("Failed to inflate mesh!");
            return Eigen::VectorXd();
        }

        Eigen::MatrixXd vertices;
        Eigen::MatrixXi cells;
        std::vector<std::vector<int>> elements;
        std::vector<std::vector<double>> weights;
        std::vector<int> body_ids;
        std::vector<std::string> node_data_name;
        std::vector<std::vector<double>> node_data;
        io::MshReader::load(sdf_velocity_path_, vertices, cells, elements, weights, body_ids, node_data_name, node_data);
        const int dim = vertices.cols();
        
        assert(node_data_name.size() == node_data.size());
        Eigen::MatrixXd shape_velocity_;
        shape_velocity_.setZero(vertices.size(), node_data.size() - 1);
        for (int j = 0; j < vertices.rows(); j++)
            for (int i = 1; i < node_data_name.size(); i++)
                for (int d = 0; d < dim; d++)
                    shape_velocity_(j * dim + d, i - 1) = node_data[0][j * 3 + d] * node_data[i][j];
    
        return grad.transpose() * shape_velocity_;

        // Eigen::VectorXd mapped_grad;
        // mapped_grad.setZero(node_data.size() - 1);
        // for (int j = 0; j < vertices.rows(); j++)
        //     for (int i = 1; i < node_data_name.size(); i++)
        //         for (int d = 0; d < dim; d++)
        //             mapped_grad(i - 1) += node_data[0][j * 3 + d] * node_data[i][j] * grad(j * dim + d);

        // return mapped_grad;
    }
}