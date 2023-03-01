#pragma once
#include "MatrixUtils.hpp"
#include <vector>

namespace polyfem::utils
{
    void inflate(std::vector<double> &params, Eigen::MatrixXd &V, Eigen::MatrixXi &F, Eigen::MatrixXd &vertex_normals, Eigen::MatrixXd &shape_vel);
}