#pragma once
#include "MatrixUtils.hpp"
#include <vector>

namespace polyfem::utils
{
    void inflate(const std::string binary_path, const std::string wire_path, const json &options, std::vector<double> &params, Eigen::MatrixXd &V, Eigen::MatrixXi &F, Eigen::MatrixXd &vertex_normals, Eigen::MatrixXd &shape_vel);
}