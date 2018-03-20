#pragma once

#include "ElasticityUtils.hpp"
#include "AutodiffTypes.hpp"
#include <Eigen/Dense>

namespace poly_fem {
namespace autogen {
void saint_venenant_2d_function(const AutodiffHessianPt &pt, const ElasticityTensor &C, Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> &res);
void saint_venenant_3d_function(const AutodiffHessianPt &pt, const ElasticityTensor &C, Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> &res);

void neo_hookean_2d_function(const AutodiffHessianPt &pt, const double lambda, const double mu, Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> &res);
void neo_hookean_3d_function(const AutodiffHessianPt &pt, const double lambda, const double mu, Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> &res);


}}
